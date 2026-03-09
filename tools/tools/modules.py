"Different pytorch modules"
import math
import torch as T
import torch.nn as nn
# import torchvision as TV
import numpy as np
from omegaconf import OmegaConf
from typing import Optional, List, Tuple, Union, Callable, Mapping

# from src.models.transformer import VisionTransformerLayer

# internal 
# from tools import misc
# from tools.transformers.attention import DenseNetwork

class FiLM(nn.Module):
    def __init__(self, in_features:int, out_features:int, dense_config=None, use_on_image:bool=True, simple_linear:bool=True, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.device=device
        self.out_features=2*out_features
        self.use_on_image=use_on_image
        self.hddn_dims = 2*out_features if out_features>in_features else 2*in_features
        self.dense_config=dense_config if dense_config is not None else {}
        if in_features is not None:
            self.network = GatedLinearUnits(self.in_features,
                                       hddn_dims=self.hddn_dims,
                                       out_dims= self.out_features,
                                    )
        else:
            self.network = nn.Linear(self.in_features, self.out_features)
            self.to(device)

    def forward(self,inpt:T.Tensor):
        # thought the network
        film_parameters = self.network(inpt)
        
        # make it Bx2xFeatures
        return film_parameters.reshape(len(film_parameters), -1,1,2)


class Gate(nn.Module):
    def __init__(self, input_shape, gate_shape, act_func="relu",
                 device="cuda"):
        super().__init__()
        self.input_shape=input_shape
        self.gate_shape=gate_shape
        self.act_func=act_func
        self.device=device
        self.get_network()

    def get_network(self):

        # downscale res img to half size
        self.input_info_conv =  nn.Sequential(
            nn.Conv2d(self.input_shape, self.input_shape,
                      kernel_size=1),
            nn.BatchNorm2d(self.input_shape))

        # gate conv the conditional info
        self.gate_info_conv = nn.Sequential(
            nn.Conv2d(self.gate_shape, self.input_shape,
                      kernel_size=1),
            nn.BatchNorm2d(self.input_shape))

        self.gating= nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(self.input_shape,self.input_shape,kernel_size=1),
            nn.Sigmoid(),
            # nn.Upsample(scale_factor=2,mode="nearest")
                                     )
        self.to(self.device)
        
    def forward(self,input_img, __input_img, gate):
        # Do not use __input_img, just a duplicate of input_img
        x = self.input_info_conv(input_img)
        gate = self.gate_info_conv(gate)
        x = self.gating(T.add(x, gate))

        return input_img*x

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, block_depth, dropout=0.1,
                 img_dim=None, zero_init=True) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size=output_size
        self.block_depth = block_depth
        self.zero_init=zero_init
        self.dropout=dropout
        self.img_dim=img_dim
        self.get_network()
        
    def get_layer(self, input_size, output_size):
        return nn.Sequential(
                    nn.BatchNorm2d(input_size, output_size, affine=False), 
                    nn.LeakyReLU(),
                    nn.Dropout(p=self.dropout),
                    nn.Conv2d(input_size, output_size, kernel_size=3, padding="same"))

    def get_network(self):
        self.skip_connection = nn.Conv2d(self.input_size, self.output_size, kernel_size=1)
                
        self.layers = nn.ModuleList([self.get_layer(self.input_size, 
                                                    self.output_size)])
        for _ in range(self.block_depth-1):
            self.layers.append(self.get_layer(self.output_size, self.output_size))

        # if self.img_dim is not None:
        #     self.layers.append(VisionTransformerLayer(
        #         img_shape=self.img_dim,
        #         n_channels=self.output_size, n_patches=8,
        #         attn_heads=8
        #     ))
        # else:
        self.layers[-1][-1].weight.data.fill_(0.00)
        self.layers[-1][-1].bias.data.fill_(0.00)
            
    def forward(self,x, ctxt=None):
        residual_connection = self.skip_connection(x)
        
        # loop over layers and add ctxt
        for layer in self.layers[:-1]:
            # 2d conv
            x = layer(x)
            
            # add ctxt after first conv
            if (ctxt is not None):
                # MxNxCxB times CxB
                x =  (1+ctxt[...,0:1])*x+ ctxt[...,1:2]
            
        # run last layer before skip
        x = self.layers[-1](x)
        
        # output skip connection
        return x+residual_connection

class MinMaxLayer(nn.Module):
    def __init__(self, min_val:np.ndarray, max_val:np.ndarray, feature_range=[0,1]):
        super().__init__()
        if not (max_val.shape==min_val.shape):
            raise ValueError("Max and min values must have the same shape!")

        # Init buffers are needed for saving/loading the layer
        self.register_buffer(
            "min",
            T.tensor(min_val, dtype=T.float32)
            )
        self.register_buffer(
            "max",
            T.tensor(max_val, dtype=T.float32)
            )

        self.register_buffer(
            "feature_range_min",
            T.tensor(feature_range[0], dtype=T.float32)
            )

        self.register_buffer(
            "feature_range_max",
            T.tensor(feature_range[1], dtype=T.float32)
            )

        # self.diff = self.max-self.min
    
    @staticmethod
    def std(inpt, min_val, max_val):
        return (inpt- min_val ) / (max_val-min_val)
    
    @staticmethod
    def scale(inpt, min_val, max_val):
        return inpt * (max_val-min_val) + min_val

    def forward(self, inpt) -> T.Tensor:
        "Apply minmax"
        inpt_std = self.std(inpt, self.min, self.max)
        inpt_scaled = self.scale(inpt_std, self.feature_range_min, self.feature_range_max)
        return inpt_scaled

    def reverse(self, inpt) -> T.Tensor:
        "reverse minmax"
        inpt_std = self.std(inpt, self.feature_range_min, self.feature_range_max)
        inpt_scaled = self.scale(inpt_std, self.min, self.max)
        return inpt_scaled
        
class Norm(T.autograd.Function):
    
    @staticmethod
    def forward(ctx, i, mu, std):
        result = (i - mu) / (std + 1e-8)

        # sort for backwards
        ctx.mark_non_differentiable(mu, std)
        ctx.save_for_backward(result, mu, std)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # result, = ctx.saved_tensors # used for debugging
        _, mu, std = ctx.saved_tensors
        grad_mu = None  # Mark mu as non-differentiable
        grad_std = None  # Mark std as non-differentiables
        return grad_output*std+mu, grad_mu, grad_std

class IterativeNormLayer(nn.Module):
    # from matt
    def __init__(self, in_shape: int, max_iters:int=100_000, ema_sync:float=0.99,
                 extra_dims:Union[tuple, int] = (),
                 track_grad_forward:bool=True, track_grad_reverse:bool=False,
                 redefine_backward:bool=False):
        super().__init__()
        if not isinstance(in_shape, list):
            self.inpt_dim = in_shape
        self.max_iters=max_iters
        self.ema_sync=ema_sync
        self.extra_dims=extra_dims
        self.do_ema=True
        # init: n is the number of samples seen so far
        n = T.tensor(0)

        # Init buffers are needed for saving/loading the layer
        self.register_buffer(
            "means",
            T.zeros(self.inpt_dim, dtype=T.float32)
            )
        self.register_buffer(
            "vars",
            T.ones(self.inpt_dim, dtype=T.float32)
            )
        self.register_buffer("n", n)

        # If the means are set here then the model is "frozen" and never updated
        self.register_buffer(
            "frozen",
            T.as_tensor( self.n > self.max_iters),
            )

        # Gradient tracking options
        self.track_grad_forward = track_grad_forward
        self.track_grad_reverse = track_grad_reverse
        self.redefine_backward=redefine_backward
        if self.redefine_backward:
            self.norm_layer = Norm()

    def __repr__(self):
        return f"IterativeNormLayer({list(self.means.shape)})"

    def __str__(self) -> str:
        return f"IterativeNormLayer(m={self.means.squeeze()}, v={self.vars.squeeze()})"

    def _mask(self, inpt: T.Tensor, mask: Union[T.BoolTensor, None] = None) -> T.Tensor:
        if mask is None:
            return inpt
        return inpt[mask]

    def _unmask(
        self, inpt: T.Tensor, output: T.Tensor, mask: Union[T.BoolTensor, None] = None
    ) -> T.Tensor:
        if mask is None:
            return output
        masked_out = inpt.clone()  # prevents inplace operation, bad for autograd
        masked_out[mask] = output.type(masked_out.dtype)
        return masked_out

    def _check_attributes(self) -> None:
        if self.means is None or self.vars is None:
            raise ValueError(
                "Stats have not been initialised and fit() has not been run!"
            )

    def fit(
        self, inpt: T.Tensor, mask: Union[T.BoolTensor, None] = None, freeze: bool = True
    ) -> None:
        """Set the stats given a population of data."""
        inpt = self._mask(inpt, mask)
        _vars, _means = T.var_mean(
            inpt, dim=(0, *self.extra_dims), keepdim=True
        )
        if (self.means.shape != _means.shape) or (self.vars.shape != _vars.shape):
            print("means shape:", self.means.shape, _means.shape)
            raise ValueError("""New means/vars shapes does not match the shape of the layer!
                             batch dimension might be missing or extra dimensions are present!""")
        else:
            self.means= _means
            self.vars = _vars
        
        self.n += T.tensor(self._n, device=self.means.device)
        # self.m2 = self.vars * self.n
        if freeze:
            self.frozen.fill_(True)
    
    def check_device(self, inpt:T.Tensor):
        "push tensor to device if not same as means"
        if self.means.device!=inpt:
            return inpt.to(self.means.device)
        else:
            return inpt

    def forward(self, inpt: T.Tensor, mask: Union[T.BoolTensor, None] = None,
                training:bool=True) -> T.Tensor:
        """Apply standardisation to a batch of inputs.

        Uses the inputs to update the running stats if in training mode.
        """

        # Save and check the gradient tracking options
        grad_setting = T.is_grad_enabled()
        T.set_grad_enabled(self.track_grad_forward)

        # check that on same device
        inpt = self.check_device(inpt)
        if mask is not None:
            mask = self.check_device(mask)
        
        # define the batch size before mask is applied
        self._n = len(inpt)

        # Mask the inputs and update the stats
        sel_inpt = self._mask(inpt, mask)

        # Only update if in training mode
        if self.training and training:
            self.update(sel_inpt)


        # Apply the mapping
        if self.redefine_backward:
            normed_inpt = self.norm_layer.apply(sel_inpt, self.means, self.vars.sqrt())
        else:
            normed_inpt = (sel_inpt - self.means) / (self.vars.sqrt() + 1e-8)

        # Undo the masking
        normed_inpt = self._unmask(inpt, normed_inpt, mask)

        # Revert the gradient setting
        T.set_grad_enabled(grad_setting)

        return normed_inpt

    def reverse(self, inpt: T.Tensor, mask: Union[T.BoolTensor, None] = None) -> T.Tensor:
        """Unnormalises the inputs given the recorded stats."""

        # check if correct device
        inpt = self.check_device(inpt)
        if mask is not None:
            mask = self.check_device(mask)
            
        # Save and check the gradient tracking options
        grad_setting = T.is_grad_enabled()
        T.set_grad_enabled(self.track_grad_reverse)

        # Mask, revert the inputs, unmask
        sel_inpt = self._mask(inpt, mask)
        unnormed_inpt = sel_inpt * self.vars.sqrt() + self.means
        unnormed_inpt = self._unmask(inpt, unnormed_inpt, mask)

        # Revert the gradient setting
        T.set_grad_enabled(grad_setting)

        return unnormed_inpt

    def update(self, inpt: T.Tensor) -> None:
        """Update the running stats using a batch of data."""

        # Freeze the model if we already exceed the requested stats
        T.fill_(self.frozen, self.n >= self.max_iters)
        if self.frozen:
            return

        # For first iteration, just run the fit on the batch
        if self.n == 0:
            self.fit(inpt, freeze=False)
            return

        # Otherwise update the statistics
        if self.do_ema:
            self._apply_ema_update(inpt)
        else:
            self._apply_welford_update(inpt)

    @T.no_grad()
    def _apply_ema_update(self, inpt: T.Tensor) -> None:
        """Use an exponential moving average to update the means and vars."""
        self.n += self._n
        nm = inpt.mean(dim=(0, *self.extra_dims), keepdim=True)
        self.means = self.ema_sync * self.means + (1 - self.ema_sync) * nm
        nv = (inpt - self.means).square().mean((0, *self.extra_dims), keepdim=True)
        self.vars = self.ema_sync * self.vars + (1 - self.ema_sync) * nv


class GatedLinearUnits(nn.Module):
    def __init__(self, input_dims, hddn_dims,
                 out_dims=None, act_func:str="swish",
                 ctxt_dims:int=0, dropout:float=0.0, norm:bool=True, device="cuda"):
        super().__init__()
        self.input_dims=input_dims
        self.out_dims = out_dims if isinstance(out_dims, (int, np.integer)) else input_dims
        self.ctxt_dims = ctxt_dims if isinstance(ctxt_dims, (int,np.integer)) else 0
        self.hddn_dims=hddn_dims
        self.dropout=dropout
        self.act_func=act_func
        self.device=device
        self.norm=norm
        self.get_network()

        self.to(self.device)

    def get_network(self):
        self.act_func = nn.SiLU()

        if self.norm:
            self.norm = nn.LayerNorm(self.input_dims+self.ctxt_dims)

        self.lin1 = nn.Linear(self.input_dims+self.ctxt_dims,
                                2*self.hddn_dims, bias=True,
                                device=self.device)

        self.lin2 = nn.Linear(self.hddn_dims, self.out_dims, bias=True,
                              device=self.device)
        
        self.drop = nn.Dropout(self.dropout)
    
    def forward(self, x):

        if self.norm:
            x = self.norm(x)

        x0, x1 = self.lin1(x).chunk(2, dim=-1)

        return self.lin2(self.drop(self.act_func(x0) * x1))

        

if __name__ == '__main__':
    pass    