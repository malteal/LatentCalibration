from copy import deepcopy
import logging
from re import L

import numpy as np
import torch as T
import types
from typing import Union, List, Callable
from sklearn.base import BaseEstimator
from torch.nn.functional import pad
from omegaconf import OmegaConf
from torch.utils.data.dataloader import default_collate
# from tools.physics import numpy_locals_to_mass_and_pt


def torch_log_squash(x: T.Tensor, eps: float = 1) -> T.Tensor:
    """Logarithmically squash the input tensor."""
    return T.sign(x)*T.log(T.abs(x) + eps)
def np_log_squash(x: T.Tensor, eps: float = 1) -> T.Tensor:
    """Logarithmically squash the input tensor."""
    return np.sign(x)*np.log(np.abs(x) + eps)


def batch_preprocess(jet_dict: list,
                     fn: List[Union[BaseEstimator, Callable]],
                     scalar_fn: List[Union[BaseEstimator, Callable]]=None,
                     inpt_name:str="csts",
                     shape:str='batch'
                     ) -> dict:
    """Preprocess the entire batch of jets.
ctxt_cfg
    Preprocesing over the entire batch is much quicker than doing it per jet
    """
    # Colate the jets into a single dictionary
    if shape != 'batch':
        jet_dict = default_collate(jet_dict)

    # Load the constituents
    csts = jet_dict[inpt_name]
    mask = jet_dict["mask"]

    for f in fn:
        # used for normaliation
        if isinstance(f, BaseEstimator):
            # Check if the number of features is the same, else add nans
            if (feat_diff := f.n_features_in_ - csts.shape[-1]) > 0:
                csts = np.pad(csts, ((0, 0), (0, 0), (0,feat_diff)), mode='constant')
            # Replace the impact parameters with the new values
            # csts[mask] = T.from_numpy(f.transform(csts[mask])).float()
            csts[mask] = f.transform(csts[mask]).astype(csts.dtype)
            # Trim the padded features
            if feat_diff > 0:
                csts = csts[..., :-feat_diff].astype(csts.dtype)
        else:
            # Replace the impact parameters with the new values
            if isinstance(f.func, types.FunctionType):
                csts = f(csts, mask)
            else:
                csts = f.transform(csts, mask)
            # csts = T.from_numpy(f.transform(csts.numpy(), mask.numpy())).float()

    # Replace the neutral impact parameters with gaussian noise
    # They are zero padded anyway so contain no information!!!
    if jet_dict[inpt_name].shape[-1] > 3:
        neutral_mask = mask & (jet_dict["csts_id"] == 0) | (jet_dict["csts_id"] == 2)

        csts[..., -4:][neutral_mask] = 0

    csts_id = jet_dict.pop('csts_id', None)

    if csts_id is not None:
        # hot encode the csts_id
        hot_encoding = np.eye(np.max(csts_id)+1)[csts_id].astype(csts.dtype)
        
        # concatenate the hot encoding to the constituents
        csts = np.concatenate([csts, hot_encoding], -1).astype(csts.dtype)
    
    # Replace with the new constituents
    jet_dict[inpt_name] = csts

    # Add the context to the dictionary
    scalar_ctxt = jet_dict.pop('jets', None)
    # jet_dict["ctxt"] = deepcopy(jet_dict)
    # scalar_ctxt = jet_dict["ctxt"].pop('jets')

    # If there is a hlvs function, apply it
    if scalar_fn is not None:
        if exp_jets := (scalar_ctxt.ndim == 1):  # Must work on batched and single jets
            scalar_ctxt = scalar_ctxt[None, ...]
        scalar_ctxt = scalar_fn.transform(scalar_ctxt).astype(scalar_ctxt.dtype)
        if exp_jets:
            scalar_ctxt = scalar_ctxt[0]
    elif scalar_ctxt is not None:
        scalar_ctxt = np_log_squash(scalar_ctxt)

    if scalar_ctxt is not None:
        jet_dict['scalars'] = scalar_ctxt
    
    return jet_dict


    