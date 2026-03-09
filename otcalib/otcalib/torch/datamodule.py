"datamodule used for otcalib"


import os
import joblib
import logging
from functools import partial
from typing import Tuple, Union
from glob import glob
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch as T
from sklearn.model_selection import train_test_split
import hydra
from nflows.transforms.base import InputOutsideDomain
from sklearn.preprocessing import StandardScaler

# internal imports
# import src.utils as utils

# ot-framwork imports
from . import torch_utils as ot_utils
# from ..utils import transformations as trans 
from ..utils import misc as misc
from . import loader


# external imports
# from tools import flows, scalers
# from tools.discriminator import DenseNet

def split_to_eval_sample(*data, data_name:list, conds_range:list, conds_name:str,
                         cvx_dim:int, noncvx_dim:int,
                         list_of_cols:list):
    conds_col_nr = np.in1d(list_of_cols, conds_name).argmax()
    if len(data)!=len(data_name):
        raise ValueError("length of data sample should be the same as data_name")
    eval_data = {}

    if conds_name is None:
        eval_data["conds"] = {}
        conds_bins=[-np.inf, -np.inf]
        number_of_events =np.min([len(i) for i in data])
        for sample, name in zip(data, data_name):
            eval_data["conds"][name] = {
                "transport": sample[:number_of_events, : cvx_dim + noncvx_dim],
                "conds": sample[:,:0],
                "sig_mask": T.ones(number_of_events)==1,
            }
    else:
        conds_bins = ot_utils.define_conds_bins(
            data[0][:, :noncvx_dim], conds_name, ranges=conds_range
        )

        for low, high in zip(
            conds_bins[f"{conds_name}"][:-1], conds_bins[f"{conds_name}"][1:]
        ): 
            eval_data[f"{conds_name}_{low}_{high}"] = {}
            number_of_events = np.min([(
                (i[:, conds_col_nr] >= low)
                & (i[:, conds_col_nr] < high)
            ).sum() for i in data])
            for sample, name in zip(data, data_name):
                mask_truth = (
                    (sample[:, conds_col_nr] >= low)
                    & (sample[:, conds_col_nr] < high)
                ).flatten()

                eval_data[f"{conds_name}_{low}_{high}"][name] = {
                    "transport": sample[mask_truth, noncvx_dim : cvx_dim + noncvx_dim][
                        :number_of_events
                    ],
                    "conds": sample[mask_truth, :noncvx_dim][:number_of_events],
                    "sig_mask": sample[mask_truth, -1][:number_of_events].bool(),
                }

    return eval_data, conds_bins

def load_downscaler(downscaler_path:str, device:str='cuda'):
    # load the background classifier
    dense_net_args = misc.load_yaml(f"{downscaler_path}/model_config.yaml")
    dense_net_args["device"]=device
    bkg_downscaler = DenseNet(**dense_net_args)
    bkg_downscaler.load(f"{downscaler_path}/bkg_down_weighting.pth")
        
    return bkg_downscaler


def bkg_removal(data:T.Tensor, downscaler_path:Union[str,T.nn.Module],
                clip:bool=False, device:str="cuda") -> Tuple[np.ndarray,np.ndarray]:
    """Remove background from data using a classifier trained on sig+bkg vs bkg
    
    outputs: mask and probability of being bkg
    """
    # classifier trained sig+bkg vs bkg
    
    # load the background classifier
    if isinstance(downscaler_path, str):
        bkg_downscaler = load_downscaler(downscaler_path, device)
    else:
        bkg_downscaler=downscaler_path.to(device)

    # predict with background classifier
    outpt_lst = []
    for i in data.chunk((len(data)//512)+1):
        i = i.to(bkg_downscaler.device)
        
        # it output the probability of being sig+bkg
        # 1-p_bkg is probability of being bkg
        p_bkg = 1-bkg_downscaler(i)[-1].cpu().detach().numpy()

        output = p_bkg/(1-p_bkg)

        outpt_lst.append(output)

    outpt_lst  = np.ravel(np.concatenate(outpt_lst, 0))

    # clip between [0,1]
    if clip:
        outpt_lst = np.clip(outpt_lst,0,1)
    
    mask = np.random.uniform(0,1,len(outpt_lst)) >= outpt_lst

    return mask, outpt_lst

class TemplateGenerator(loader.Dataset):
    def __init__(self, path, conds, cvx_dim, generate_sig=True, generate_bkg=True,
                 trans_lst=None, 
                 device="cuda",
                 ot_calib:dict=None, **kwargs) -> None:
        self.path = path
        self.device=device
        self.generate_bkg = generate_bkg
        self.generate_sig = generate_sig
        self.trans_lst=trans_lst
        self.cvx_dim=cvx_dim
        self.ot_calib=ot_calib # should be a dict with sig or bkg paths

        # define defaults
        self.classifier=None
        self.dataloader_kwargs = kwargs.get("dataloader_kwargs", {'batch_size': 512,
                                                           'shuffle': False,
                                                           'num_workers': 4,
                                                           'drop_last': False})
        self.duplications = kwargs.get("duplications", 1)
        self.verbose = kwargs.get("verbose", False)

        # define paths
        if self.generate_sig:
            self.sig_flow_path = misc.load_yaml(f"{self.path}/signal_flow_path.yaml")['sig_path']
        if self.generate_bkg:
            self.bkg_flow_path = misc.load_yaml(f"{self.path}/background_flow_path.yaml")['bkg_path']

        self.density_clf_path = f"{self.path}/f_sig"

        # set conds in a vector format and make sure it torch tensor
        if isinstance(conds, T.Tensor) & (len(conds.shape)==1):
            self.conds = conds.float().view(-1,1)
        elif (len(conds.shape)==1):
            self.conds = T.tensor(conds).float().view(-1,1)
        elif not isinstance(conds, T.Tensor):
            self.conds = T.tensor(conds).float()
        else:
            self.conds = conds.float()

        self.n_chunks = len(self.conds) // kwargs.get(
            "chunk_size", 25_000 if self.duplications == 1 else 10_000
        )+1
        self.noncvx_dim = self.conds.shape[1]

        #load flows and density clf
        self.load_models()
        

        # constuct tensor with data size 
        self.sample_wo_labels = T.concat([self.conds,
                                          T.ones((len(self.conds),
                                                  self.cvx_dim))],1
                                 )

        #first sample
        self.sample_data()
    
    @staticmethod
    def load_flows(path, device:str='cuda'):
        ## sig flow
        config = misc.load_yaml(f"{path}/config.yaml")
        flow = hydra.utils.instantiate(config.model.flow)
        
        # last saved model
        model_path = misc.sort_by_creation_time(glob(f"{path}/models/*"))[-1]

        flow.load_state_dict(T.load(model_path)['flow'])

        scaler = joblib.load(f"{path}/scaler.json")
        
        return flow.to(device), scaler
        
    def load_models(self):
        """init all functions used for the template generator"""
        ## sig flow
        if self.generate_sig:
            self.sig_flow, self.sig_scaler = self.load_flows(self.sig_flow_path,
                                                             device=self.device)

        ## bkg flow
        if (self.generate_bkg and (self.sig_flow_path is not None)
            and (len(self.bkg_flow_path)>0)):
            self.bkg_flow, self.bkg_scaler = self.load_flows(self.bkg_flow_path,device=self.device)
        elif self.generate_bkg and self.sig_flow_path is None:
            print("bkg_path is None but boolean for generating bkg is True!")
            self.generate_bkg = False

        ## load clf
        if len(glob(f"{self.density_clf_path}/*"))>0:
            ## sig flow
            config = misc.load_yaml(f"{self.path}/.hydra/config.yaml")
            clf_config = hydra.utils.instantiate(config.model.dense)
            clf_config['input_dim'] = self.noncvx_dim
            clf_config["device"] = self.device

            self.classifier = DenseNet(old_layernorm_version=False, **clf_config)
            
            if len(glob(f"{self.density_clf_path}/models/*"))>0:
                latest_model = sorted(glob(f"{self.density_clf_path}/models/*"), key=os.path.getmtime)[
                    -1
                ]
            else:
                latest_model = f"{self.density_clf_path}/density_clf.pth"
            try:
                self.classifier.load(latest_model, key_name="model_state_dict")
            except KeyError:
                self.classifier.load(latest_model, key_name="model")
            self.classifier = self.classifier.to(self.device)
        
        if self.ot_calib is not None:
            # load ot calibration
            # the calibration network is the last model of the list
            if 'sig' in self.ot_calib:
                self.ot_calib['sig'] = ot_utils.load_model_w_hydra(self.ot_calib['sig'])[-1]
            if 'bkg' in self.ot_calib:
                self.ot_calib['bkg'] = ot_utils.load_model_w_hydra(self.ot_calib['bkg'])[-1]

    def generate(self, flow, conds, scaler):
        "Run flow inference on the conds (conditional distribution) and rescale it"

        out_of_domian_while = True

        while out_of_domian_while:
            try:
                generate = [
                    flow.sample(1, i.to(self.device))[:, 0, :None].cpu().detach()
                    for i in tqdm(conds.chunk(self.n_chunks), disable=not self.verbose)
                ]

                generate = T.concat(generate, 0)

                generate = T.concat([conds.cpu(), generate], 1).cpu().detach().numpy()
                try:
                    generate[:, conds.shape[1] :] = scaler.inverse_transform(
                        generate[:, conds.shape[1] :]
                    )
                except ValueError:
                    generate = scaler.inverse_transform(generate)
                
                out_of_domian_while = False
            except InputOutsideDomain:
                logging.warning("InputOutsideDomain error - Trying again")
                continue

        return T.tensor(generate)

    def generate_data(self):
        "Generate pb,pc,pu given conds_values template"

        # run sig/bkg classifier
        if self.classifier is not None:
            sig_proba = [
                self.classifier(i.to(self.device))[1].cpu().detach()[:, 0]
                for i in tqdm(self.conds.chunk(self.n_chunks),
                            disable=not self.verbose)
            ]
            sig_proba = T.concat(sig_proba, 0)
            if (sig_proba>1).any():
                sig_proba = self.classifier.sigmoid_layer(sig_proba)
        else:
            sig_proba = T.ones((len(self.conds),))

        sig_mask = (sig_proba >= T.rand(len(sig_proba)) if self.generate_bkg 
                    else T.ones_like(sig_proba).bool())
        sampled_data = self.sample_wo_labels.clone()

        if self.generate_sig:

            sampled_data[sig_mask] = T.tensor(
                self.sig_scaler.transform(sampled_data[sig_mask])
                ).float() # this has dimension issues with more conds and trans input

            sampled_data[sig_mask] = self.generate(
                flow=self.sig_flow, scaler=self.sig_scaler,
                conds=sampled_data[:,:self.noncvx_dim][sig_mask].clone()
            )

        else:
            sig_mask = T.ones((len(self.conds),))==0 # all false

        if self.generate_bkg & T.any(~sig_mask):
            sampled_data[~sig_mask] = T.tensor(self.bkg_scaler.transform(sampled_data[~sig_mask])).float()
            sampled_data[~sig_mask] = self.generate(
                flow=self.bkg_flow, scaler=self.bkg_scaler,
                conds=sampled_data[:,:self.noncvx_dim][~sig_mask].clone()
            )
        # IMPORTANT: precision of the conds might change due to all the norms
        sampled_data[:, 0] = self.sample_wo_labels.clone()[:,0]

        data = T.concat([sampled_data,sig_mask.view(-1,1)*1], 1).float().detach()
        return data
    
    def sample_data(self):
        data=[]
        for _ in tqdm(range(self.duplications), disable=self.duplications==1):
            data.append(self.generate_data())

        self.data = T.concat(data,0)

        # log squash the distributions
        if self.trans_lst is not None:
            self.data[:, :self.noncvx_dim+self.cvx_dim] = transform_data(
                self.data[:, :self.noncvx_dim+self.cvx_dim], self.trans_lst)

        # apply ot calibration on sig or bkg if needed        
        if self.ot_calib is not None:
            if 'sig' in self.ot_calib:
                self.data[:, self.noncvx_dim:self.noncvx_dim+self.cvx_dim] = self.ot_calib['sig'].chunk_transport(
                    self.data[:, self.noncvx_dim:self.noncvx_dim+self.cvx_dim], self.data[:, :self.noncvx_dim], sig_mask=self.data[:,-1].bool())
            if 'bkg' in self.ot_calib:
                self.data = self.ot_calib['bkg'].chunk_transport(
                    self.data[:, self.noncvx_dim:self.noncvx_dim+self.cvx_dim], self.data[:, :self.noncvx_dim], sig_mask=self.data[:,-1]!=1)

        self.dataset = T.utils.data.DataLoader(self.data.float(), **self.dataloader_kwargs)
        self._iterator = iter(self.dataset)
        

    def _reset(self):
        "init iterator of the dataloader again"
        # create new data
        self.verbose=False
        flow_error=True
        while flow_error:
            try:
                self.sample_data()
                flow_error=False
            except AssertionError as e:
                print(e)
                print(" AssertionError from flow - try one more time ".center(20, "~"))

        # make it to iterator
        self._iterator = iter(self.dataset)

class OTDataModule:
    """DataModule for the ftag calibration
    
    Should be used instead of load_ftag_data!
    """
    def __init__(
        self,
        source_sample: partial,
        target_sample: partial,
        transportnames:list,
        condnames:list=None,
        template_args: Union[None, dict]=None,
        device="cuda",
        apply_weights:bool=False,
        use_bkg_weighting:str="all",
        maxevents: int=None,
        **kwargs,
    ) -> None:
        """
        Load the ftag data and init
        
        """
        
        self.template_args = {} if template_args is None else template_args
        self.condnames=[] if condnames is None else condnames
        self.transportnames=transportnames
        self.device=device
        self.kwargs=kwargs
        self.apply_weights=apply_weights
        self.use_bkg_weighting=use_bkg_weighting

        # unpack kwarsg' information
        self.bootstraps_iters = kwargs.get("bootstraps_iters", 0)
        self.maxevents = maxevents
        self.valid_maxevents = kwargs.get("valid_maxevents", maxevents)
        self.trans_lst= kwargs.get('trans_lst')
        self.valid_size = self.kwargs.get("valid_size", 0.1)
        self.conds_range = self.kwargs.get("conds_range", [0,1])
        self.noncvx_dim = kwargs.get("nonconvex_dim", len(self.condnames))

        if isinstance(use_bkg_weighting, bool):
            raise ValueError("use_bkg_weighting should be a string")

        if "*" in self.template_args.get("path", ''):
            self.template_args["path"] = glob(self.template_args["path"])[-1]

        if self.valid_size is not None and self.valid_size > 1:
            raise ValueError("valid_size should be a fraction of the total sample size")

        # initialize target sample
        self.target_sample = target_sample()
        
        # if target sample is tuple, then it is a valid sample
        if isinstance(self.target_sample, tuple):
            self.target_sample, self.valid_target_sample = self.target_sample
            self.valid_target_sample = T.tensor(self.valid_target_sample).float()[:self.maxevents]
        
        if not isinstance(self.target_sample, T.Tensor):
            self.target_sample = T.tensor(self.target_sample).float()[:self.maxevents]
        self.target_sample = self.target_sample[:self.maxevents]

        # initialize source sample
        self.source_sample = source_sample()

        # if source sample is tuple, then it is a valid sample
        if isinstance(self.source_sample, tuple):
            self.source_sample, self.valid_source_sample = self.source_sample
            self.valid_source_sample = T.tensor(self.valid_source_sample).float()[:self.maxevents]

        if not isinstance(self.source_sample, T.Tensor):
            self.source_sample = T.tensor(self.source_sample).float()
        self.source_sample = self.source_sample[:self.maxevents]

        # set transportnames to None if target space is very large
        if self.transportnames is not None:
            self.cvx_dim = kwargs.get("convex_dim", len(self.transportnames))
        else:
            # if the transportnames are not given, loop over target dimensions
            self.transportnames = [f"latn{i}" for i in range(self.target_sample.shape[1])]
            self.cvx_dim = len(self.transportnames)

        self.setup()
        
    def setup(self):
        """setup the dataloaders and validation data
        
        apply background calibration if needed
        
        apply background downscaler if needed
        
        in the end create eval data
        """
        
        # split in train/valid if needed
        if self.valid_size is None and not hasattr(self, 'valid_source_sample'):
            self.valid_target_sample = self.target_sample
        elif not hasattr(self, 'valid_source_sample'):
            (self.target_sample, self.valid_target_sample) = train_test_split(
                self.target_sample, test_size = self.valid_size)

            (self.source_sample, self.valid_source_sample) = train_test_split(
                self.source_sample,test_size = self.valid_size)
        
        # create training template generators
        self.source_sampler, self.target_sampler = self.create_tp_samplers()
        
        # create validation template generators
        self.valid_source_sampler, self.valid_target_sampler = self.create_tp_samplers(
            self.valid_target_sample, source_sample=self.valid_source_sample)

        # calibrate bkg events if needed
        # assuming self.source_sampler.bkg_calib_model is the correct model to use
        # assuming pT first columns and proba [1:4]
        if isinstance(self.template_args.get("bkg_calib"), str):
            mask_bkg = self.source_sample[:, -1] == 0
            self.source_sample[mask_bkg, 1:4] = self.source_sampler.bkg_calib_model.chunk_transport(
                self.source_sample[mask_bkg, 1:4],
                self.source_sample[mask_bkg, :1],
                n_chunks=mask_bkg.sum().numpy()//1024+1)
            
        # remove overpredicted bkg
        if self.use_bkg_weighting != 'all':
            self.original_mask_downscaler = self.apply_bkg_downscaler(self.source_sample)
            
            # apply mask
            self.source_sample = self.source_sample[self.original_mask_downscaler]

        # create eval data
        self.eval_data, self.conds_bins = self.create_eval_data(
            {"truth":self.valid_target_sampler.data,
             "Flow": self.valid_source_sampler.data,
             "MC": self.source_sample
             })
    
    def __call__(self):
        return self.source_sampler, self.target_sampler, self.eval_data, self.conds_bins
        
    def get_samples(self, datamodule:dict, sample_args:dict=None,
                    sample_to_size:int=None, bootstraps_iters:int=0) -> T.Tensor:
        "get samples from get_data"
        
        raise NotImplementedError("get_samples should be implemented in the child class in target_sample and source_sample")

    def create_tp_samplers(self, sample:T.Tensor=None, shuffle:bool=False, source_sample:T.Tensor=None) -> Tuple[TemplateGenerator, loader.Dataset]:
        "Create source and target sampler"
        
        # get sample
        if sample is None:
            sample = self.target_sample.clone()

        # shuffle target sample
        if shuffle:
            sample = sample[T.randperm(len(sample))]
        
        if self.use_bkg_weighting !='all': # used to remove bkg from data
            mask = self.apply_bkg_downscaler(sample)
            sample = sample[mask]
        
        # transformation for target
        target_sample = sample.clone()
        if self.trans_lst is not None:
            target_sample[:, :self.noncvx_dim+self.cvx_dim] = transform_data(
            target_sample[:, :self.noncvx_dim+self.cvx_dim], self.trans_lst)
        
        # create target iterator
        target_sampler = loader.Dataset(
            target_sample.clone(),
            dataloader_kwargs=self.kwargs["dataloader_kwargs"],
            device=self.device,
            cvx_dim=self.cvx_dim,
            noncvx_dim=self.noncvx_dim,
        )

        if len(self.template_args)>0:
            # use TemplateGenerator to create source sampler
            if 'trans_lst' not in self.template_args:
                self.template_args['trans_lst'] = self.trans_lst

            source_sampler = TemplateGenerator(
                conds=sample[:, :self.noncvx_dim].cpu().clone(),
                device=self.device,
                dataloader_kwargs=self.kwargs["dataloader_kwargs"],
                cvx_dim=self.cvx_dim,
                **deepcopy(self.template_args)
            )
            # check if transformation list is the same between sample and flow
            trans_lst = misc.load_yaml(
                f"{source_sampler.sig_flow_path}/config.yaml"
                )['data'].get('trans_lst')

            if self.trans_lst != trans_lst:
                # TODO dont think this is correct!!! need to check the list 
                raise ValueError('List of transformation has to be the same as the one used in the flow')
        else:
            if self.noncvx_dim>0:
                logging.warning("No template generator is used and noncvx_dim is not 0"
                                " - might have misaligned conditions"
                                )
            # create source sampler from source distribution
            source_sampler = loader.Dataset(
                self.source_sample.clone() if source_sample is None else source_sample,
                dataloader_kwargs=self.kwargs["dataloader_kwargs"],
                device=self.device,
                cvx_dim=self.cvx_dim,
                noncvx_dim=self.noncvx_dim
                )

        return source_sampler, target_sampler
            
    def create_eval_data(self, samples:dict):
        
        # setup evaluate data
        if self.kwargs.get("same_length", True):
            min_size = np.min([len(j) for i,j in samples.items()])
            for i,j in samples.items():
                samples[i] = j[: min_size]

        eval_data, conds_bins = split_to_eval_sample(
            *samples.values(),
            data_name=samples.keys(),
            conds_range=self.conds_range,
            conds_name=self.condnames[0] if len(self.condnames)>0 else None,
            cvx_dim=self.cvx_dim,
            noncvx_dim=self.noncvx_dim,
            list_of_cols=self.condnames+self.transportnames,
            )
    
        return eval_data, conds_bins


    def apply_bkg_downscaler(self, sample: T.Tensor) -> T.Tensor:

        logging.info("Load Background Downscaler")
        downscaler_path = self.template_args["path"]+"/bkg_weighting"
        
        # 
        bkg_calib = self.template_args.get("bkg_calib")
        if bkg_calib is not None:
            bkg_calib_name = bkg_calib.split("/")[-1]
            downscaler_path = downscaler_path.replace("bkg_weighting", f"bkg_weighting{bkg_calib_name}")

        # run bkg remover
        mask, out_lst = bkg_removal(sample[:, :4], downscaler_path, clip=False)
        
        # apply mask
        if self.use_bkg_weighting == 'rm_over_pred':
            # still sample bkg cause only overpredicted bkg is removed
            mask = out_lst<=1
            self.template_args["generate_bkg"]=True
        else:
            # do not sample bkg if all bkg is removed
            self.template_args["generate_bkg"]=False
        
        self.template_args["duplications"]=1

        return mask

