# diffusion schemes
import torch as T
import torch.nn as nn
import numpy as np
from tqdm import tqdm 
from torchdiffeq import odeint
import pytorch_lightning as L
from typing import Union
from torch.utils.data import DataLoader

# internal
from .datamodule.pipeline import Loader
from . import torch_utils as utils
from . import transformations as trans

def generate_gaussian_noise(shape:dict, datatype:str, 
                            eval_ctxt:dict,
                            n_constituents: Union[tuple, int]=None,
                            size:int=None, **kwargs):
    if size is None:
        size = 1e10
    # move ctxt tensor to array
    for i in eval_ctxt:
        if isinstance(eval_ctxt[i], T.Tensor):
            eval_ctxt[i] = eval_ctxt[i].cpu().numpy()
    "generate noisy images for diffusion"
    if "image" in datatype:
        if (len(eval_ctxt)) and (size > len(eval_ctxt.get("images", []))):
            size = len(eval_ctxt["images"])
        mask=None
        gaussian_noise = T.randn(tuple([size]+shape["images"])).numpy()

    elif "pc" in datatype:
        if n_constituents is None:
            raise ValueError("n_constituents has to be defined")
        
        # calculate the n constituents
        if isinstance(n_constituents, tuple):
            n_constituents = np.random.randint(*n_constituents, size)
        elif isinstance(n_constituents, (int, np.int64)):
            n_constituents = np.random.randint(1, n_constituents, size=size)
        
        if size>len(n_constituents):
            size = len(n_constituents)
            
        if not isinstance(n_constituents, np.ndarray):
            raise TypeError("n_constituents has to be a np.array")

        mask = np.zeros([size]+shape["images"][:1])==1
        for nr,i in enumerate(n_constituents[:size]):
            mask[nr, :i] = True
        gaussian_noise = T.randn(tuple([size]+shape["images"])).numpy()
        
        #reduce eval_ctxt size
        for i in eval_ctxt:
            eval_ctxt[i] =eval_ctxt[i][:size]

    return DataLoader(Loader(gaussian_noise,mask=mask,ctxt=eval_ctxt),
                      **kwargs.get("loader_kwargs",
                                   {"batch_size": 512, "num_workers": 4})
                      )

class Solvers(nn.Module):
    def __init__(self, solver_name, verbose=False):
        super().__init__()
        if "heun2d" in solver_name:
            self.do_heun_step=True
        else:
            self.do_heun_step=False
        self.solver = self.heun2d
        self.verbose=verbose
    
    def reverse_diffusion(self, **kwargs):
        raise NotImplementedError("reverse diffusion has to be defined")

    def _train_step(self, **kwargs):
        raise NotImplementedError("train step has to be defined")

    def denoise(self, **kwargs):
        raise NotImplementedError("denoise has to be defined")

    def _shared_step(self, inpt:T.Tensor, ctxt:T.Tensor=None,
                     mask:T.Tensor=None, sigma_noise:T.Tensor=None
                     )->T.Tensor:
        ctxt={} if ctxt is None else ctxt

        # embedding
        sigma_noise = self.time_embedding(sigma_noise)
        
        # add noise embedding to ctxt
        sigma_noise = sigma_noise.squeeze(1)
            
        # predict noise component and calculate the image component using it
        pred = self(inpt, time=sigma_noise, mask=mask, latn = ctxt)

        return pred

    @T.no_grad()
    def heun2d(self, initial_noise:Union[T.Tensor, tuple], diffusion_steps:np.ndarray,
               ctxt:dict=None, mask:T.Tensor=None)->T.Tensor:
        if ctxt is None:
            ctxt = {}

        #heuns 2nd solver
        # scale to correct std
        x = initial_noise*diffusion_steps[0]
        for i in tqdm(range(len(diffusion_steps)-1),
                      disable=not self.verbose):

            # left tangent
            dx = 1/diffusion_steps[i] * (x-self.denoise(x, diffusion_steps[i], ctxt=ctxt,
                                                        mask=mask))
            dt = (diffusion_steps[i+1]-diffusion_steps[i])
            # solve euler
            x_1 = x+dt*dx

            if all(diffusion_steps[i+1]!=0) & self.do_heun_step: # solver heun 2nd
                # right tangent
                dx_ = 1/diffusion_steps[i+1] * (x_1 -self.denoise(x_1, diffusion_steps[i+1],
                                                         ctxt=ctxt,
                                                         mask=mask))

                x = (x+dt*(dx+dx_)*0.5)
            else:
                x = x_1
        return x

class UniformDiffusion(Solvers):
    "from https://keras.io/examples/generative/ddim/"
    def __init__(self, min_signal_rate=0.02, max_signal_rate=1,
                 **kwargs) -> None:
        # super().__init__()
        super().__init__("heun2d")
        self.min_signal_rate=min_signal_rate
        self.max_signal_rate=max_signal_rate

    def uniform_diffusion_time(self, diffusion_times):
        if self.min_signal_rate==self.max_signal_rate:
            noise_rates = signal_rates = T.ones_like(diffusion_times)
        else:
            # diffusion times -> angles
            start_angle = T.acos(T.tensor(self.max_signal_rate))
            end_angle = T.acos(T.tensor(self.min_signal_rate))

            diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

            # angles -> signal and noise rates
            signal_rates = T.cos(diffusion_angles)
            noise_rates = T.sin(diffusion_angles)
            # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def reverse_diffusion(self, noise, ctxt=None, mask=None, n_steps=None):
        if n_steps is None:
            n_steps = self.eval_cfg.n_diffusion_steps
        # reverse diffusion = sampling
        num_images = noise.shape[0]
        step_size = 1.0 / self.eval_cfg.n_diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = noise
        for step in range(n_steps):
            noisy_images = next_noisy_images.detach()

            # separate the current noisy image to its components
            diffusion_times = T.ones([num_images]+[1]*(len(noise.shape)-1),
                                     device=noise.device) - step * step_size

            noise_rates, signal_rates = self.uniform_diffusion_time(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates,
                ctxt=ctxt, mask=mask
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.uniform_diffusion_time(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images.cpu()
    
    # def _shared_step(self, inpt, noise_rates, signal_rates,
    #                  ctxt=None, mask=None):
    #     # the exponential moving average weights are used at evaluation
    #     # inpt = self.inpt_normaliser(inpt, mask, training=training)
        
    #     # embed noise
    #     noise_emb = self.time_embedding(noise_rates, len(inpt.shape))

    #     # add noise embedding to ctxt
    #     if ctxt is not None:
    #         ctxt = T.concat([noise_emb.squeeze(1), ctxt],1)
    #     else:
    #         ctxt = noise_emb.squeeze(1)

    #     # predict noise component and calculate the image component using it
    #     pred_noises = self(inpt, ctxt=ctxt, mask=mask)
        
    #     # remove noise from image
    #     # pred_images = (inpt - noise_rates * pred_noises) / signal_rates

    #     return pred_noises
    
    def _train_step(self, inpt:T.Tensor, ctxt:T.Tensor=None,
                     mask:T.Tensor=None, training:bool=True
                     ):
        # Pass through the normalisers
        inpt = self.inpt_normaliser(inpt, mask, training=training)

        # sample uniform random diffusion times
        diffusion_times = T.rand(
            size=[len(inpt)]+[1]*(len(inpt.shape)-1),
            device=inpt.device
        )

        # sample time
        noise_rates, signal_rates = self.uniform_diffusion_time(diffusion_times)
        
        # generate noise
        noises = T.randn_like(inpt)

        #generate noise
        noisy_inpt = signal_rates * inpt + noise_rates * noises
        
        # run the shared step
        pred_noises = self._shared_step(inpt=noisy_inpt, ctxt=ctxt, mask=mask,  sigma_noise=noise_rates)
        
        pred_inpt = (inpt - noise_rates * pred_noises) / signal_rates
        
        # calculate loss
        # pred_inpt = pred_inpt.detach()

        noise_loss = self.loss(noises[mask], pred_noises[mask])  # used for training

        image_loss = self.loss(inpt[mask], pred_inpt[mask])  # only used as metric
        
        return image_loss

    def denoise(self, inpt:T.Tensor, noise_rates, signal_rates,
                ctxt:T.Tensor=None, mask:T.Tensor=None):

        # get predictions
        pred_noises, pred_inpt =  self._shared_step(inpt, noise_rates, signal_rates, ctxt=ctxt, mask=mask)

        # scale given preconditions
        return pred_noises, pred_inpt


class ElucidatingDiffusion(Solvers):
    def __init__(self, rho=7, s_min=0.002, s_max=80, s_data=1, P_mean=-1.2, P_std=1.2,
                 **kwargs) -> None:
        super().__init__("heun2d")
        # super().__init__()
        self.rho=rho
        self.s_min=s_min
        self.s_max=s_max
        self.s_data=s_data
        self.P_mean=P_mean
        self.P_std=P_std
        self.inv_rho = (1/self.rho)

    def sample_sigma(self, i:int, N:int=20, n_imgs:int=1):
        "sample sigma during generation"
        if self.P_std==-1:
            pass
        else:
            time_step = [(
                self.s_max**self.inv_rho
                +i/(N-1)*(self.s_min**self.inv_rho-self.s_max**self.inv_rho)
                        )**self.rho]*n_imgs
            time_step = T.tensor(np.stack(time_step, 0))
        return utils.append_dims(time_step, target_dims=4)

    
    def precondition(self, sigma):
        
        norm=T.sqrt(sigma**2+self.s_data**2)
        
        c_skip= self.s_data**2/norm**2
        
        c_out = sigma*self.s_data/norm
        
        c_input = 1/norm
        
        c_noise = 1/4*T.log(sigma)
        
        return c_skip.detach(), c_out.detach(), c_input.detach() #, c_noise

    def sample_time(self,size, target_dims):
        if self.P_std==-1:
            pass
        else:
            noise = T.randn(size, requires_grad=True)
            noise = T.exp(noise*self.P_std+self.P_mean)
            noise = T.clip(noise, self.s_min, self.s_max)
        return utils.append_dims(noise, target_dims=target_dims)
    
    def denoise(self, images, sigma_noise, ctxt=None, mask=None):

        # get preconditions
        c_skip, c_out, c_inpt = self.precondition(sigma_noise)

        # get predictions
        pred_inpt =  self._shared_step(inpt=c_inpt*images, ctxt=ctxt, mask=mask,
                                       sigma_noise=sigma_noise)

        # scale given preconditions
        return (c_skip*images+c_out*pred_inpt)
    
    def _train_step(self, inpt:T.Tensor, ctxt:T.Tensor=None,
                     mask:T.Tensor=None, training:bool=True
                     ):

        # Pass through the normalisers
        inpt = self.inpt_normaliser(inpt, mask, training=training)

        # sample noise distribution
        sigma_noise = self.sample_time(len(inpt), len(inpt.shape)).to(inpt.device)

        # calculate preconditions
        c_skip, c_out, c_inpt = self.precondition(sigma_noise)

        #generate noise
        noises = T.randn_like(inpt) * sigma_noise
        
        # mix the images with noises accordingly
        noisy_images = inpt + noises

        # scaled target
        scaled_target = (inpt-c_skip*noisy_images)/c_out
        
        # run the shared step
        pred_target = self._shared_step(inpt=c_inpt*noisy_images, ctxt=ctxt, mask=mask,
                                        sigma_noise=sigma_noise)
        
        # calculate loss
        loss = self.loss(pred_target[mask],scaled_target[mask])
            
        return loss
    
    # def _shared_step(self, inpt:T.Tensor, ctxt:T.Tensor=None,
    #                  mask:T.Tensor=None, sigma_noise:T.Tensor=None
    #                  )->T.Tensor:
    #     ctxt={} if ctxt is None else ctxt

    #     # calculate preconditions
    #     # _, _, c_input = self.precondition(sigma_noise)
        
    #     # embedding
    #     sigma_noise = self.time_embedding(sigma_noise, len(inpt.shape)).to(inpt)
        
    #     # add noise embedding to ctxt
    #     sigma_noise = sigma_noise.squeeze(1)
            
    #     # predict noise component and calculate the image component using it
    #     pred_target = self(x = inpt, ctxt=sigma_noise, mask=mask, latn = ctxt)

    #     return pred_target

    # # Turn off autocast
    # @T.autocast("cuda", enabled=False)  # Dont autocast during integration
    # @T.autocast("cpu", enabled=False)
    def reverse_diffusion(self, noise, ctxt=None, mask=None, n_steps=None):
        
        if n_steps is None:
            n_steps = self.eval_cfg.n_diffusion_steps
        #heuns solver
        
        sigma_steps = self.sample_sigma(
            np.arange(n_steps),N=n_steps,n_imgs=len(noise)
            ).permute(1,0,2,3).to(noise)

        if len(noise.shape)+1!=len(sigma_steps.shape): # noise might need addtional dimensions
            sigma_steps = sigma_steps.unsqueeze(-1)

        return self.solver(initial_noise=noise,
                            diffusion_steps=sigma_steps,
                            ctxt=ctxt, mask=mask)
        
class RectifiedFlows(Solvers):
    def __init__(self, time_sampler:dict = {}) -> None:
        super().__init__('')

        self.time_sampler=time_sampler

        # if isinstance(self.time_sampler, dict):
        self.time_sampler = lambda x: trans.logit_normal(T.randn(x)*time_sampler.get("std", 1)+time_sampler.get("mean", 0))

    def sample_time(self, inpt:T.Tensor) -> T.Tensor:
        "generate time steps for diffusion"
        times =  self.time_sampler(len(inpt))
        return utils.append_dims(times, target_dims=len(inpt.shape))
    
    def _train_step(self, inpt:T.Tensor, ctxt:T.Tensor=None,
                     mask:T.Tensor=None, training:bool=True
                     ):
        # Pass through the normalisers
        inpt = self.inpt_normaliser(inpt, mask, training=training)

        # sample noise distribution
        times = self.sample_time(inpt).to(inpt.device)

        #generate base distribution
        eps = T.randn_like(inpt)
        
        # mix the images with noises accordingly
        zt = (1-times)*inpt+times*eps
        
        # run the shared step to predict velocity
        pred_v = self._shared_step(inpt=zt,ctxt=ctxt, mask=mask,
                                   sigma_noise=times.squeeze(1))
        
        # target velocity
        target_v = (eps-inpt)
        
        # mask that masks out noise from neurals
        _mask = mask.unsqueeze(-1) & T.ones_like(inpt).bool()
        if _mask.shape[-1]>7:
            _mask[...,3:7] = inpt[..., 3:7]!=0
        
        # calculate loss
        loss = self.loss(pred_v[_mask], target_v[_mask])
        
        return loss
    
    # def _shared_step(self, inpt:T.Tensor, ctxt:T.Tensor=None,
    #                  mask:T.Tensor=None, sigma_noise:T.Tensor=None
    #                  )->T.Tensor:
    #     ctxt={} if ctxt is None else ctxt

    #     # embedding
    #     sigma_noise = self.time_embedding(sigma_noise)
        
    #     # add noise embedding to ctxt
    #     sigma_noise = sigma_noise.squeeze(1)
            
    #     # predict noise component and calculate the image component using it
    #     pred_v = self(x = inpt, time=sigma_noise, mask=mask, latn = ctxt)

    #     return pred_v

    # Turn off autocast
    # @T.autocast("cuda", enabled=False)  # Dont autocast during integration
    # @T.autocast("cpu", enabled=False)
    def reverse_diffusion(self, noise:T.Tensor, ctxt:T.Tensor=None,
                          mask:T.Tensor=None, n_steps:int=None,
                          solver_kw:dict=None) -> T.Tensor:
        """Generate a sample by solving the ODE"""
        
        #  x1: T.Tensor, ctxt: T.Tensor, times: T.Tensor,
                #  mask: T.Tensor=None, solver_kw:dict=None

        if solver_kw is None:
            solver_kw = {}

        if n_steps is None:
            n_steps = self.eval_cfg.get('n_diffusion_steps', 10)
            
        times = T.linspace(1,0, n_steps, device=noise.device)

        def ode_fn(t, xt) -> T.Tensor:
            t = t * xt.new_ones([xt.shape[0], 1])
            return self._shared_step(inpt=xt, sigma_noise=t, ctxt=ctxt, mask=mask)

        return odeint(ode_fn, noise, times, method="midpoint")[-1]


if __name__ == '__main__':
    func = lambda x: trans.logit_normal(T.randn(x)*1-1)
    
    import matplotlib.pyplot as plt
    x = func(100_000)
    embedding_dims=8
    pos = Sinusoidal(  embedding_dims= embedding_dims,
  embedding_min_frequency= 0.0002,
  embedding_max_frequency= 1)
    # pos = FourierFeatures(1, embedding_dims)
    px = np.linspace(0.0002, 1, 1001)
    position = pos(T.tensor(px).view(-1,1).float(),2)
    for i in range(embedding_dims):
        plt.plot(px, position[:,i])
        

    plt.figure()
    plt.hist(x.numpy(), bins=100)