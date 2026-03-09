"Utils for PICNN in torch"
import os
import pickle
from glob import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import hydra
import copy
import logging
from json import JSONDecodeError
from functools import partial

from ..utils import misc
from . import PICNN

from torch.optim.lr_scheduler import CosineAnnealingLR


def create_eval_dict(
    *distribution,
    names: list,
    cvx_dim: int,
    noncvx_dim: int,
    percentiles=None,
    with_total=False,
):
    "this only work for noncvx = 1"
    # eval data
    if percentiles is not None:
        percentiles = np.round(
            np.percentile(distribution[0][:, :1].detach().numpy(), percentiles), 3
        )
        # percentiles[-1]+=percentiles[-1]/100
    else:
        percentiles = []

    # create conds bins
    eval_data = {}
    if noncvx_dim == 1:
        for nr in range(len(percentiles) - 1):
            e_data = {}
            for dist, name in zip(distribution, names):
                mask = torch.flatten(
                    (dist[:, :1] >= percentiles[nr])
                    & (dist[:, :1] < percentiles[nr + 1])
                )
                sig_mask = (
                    torch.ones((torch.sum(mask),)).bool()
                    if dist.shape[1] == 1 + cvx_dim
                    else dist[:, 1 + cvx_dim][mask]
                )  # if there is label in last column
                dist = dist.detach()
                dist.requires_grad = True
                e_data[name] = {
                    "conds": dist[:, :1][mask],
                    "transport": dist[:, 1 : cvx_dim + 1][mask],
                    "sig_mask": sig_mask.bool(),
                }
            eval_data[f"conds_{percentiles[nr]}_{percentiles[nr+1]}"] = e_data

    # create total conds
    if (len(percentiles) == 0) or with_total:
        eval_data["total"] = {}
        for dist, name in zip(distribution, names):
            if dist.shape[1] == noncvx_dim + cvx_dim:
                sig_mask = torch.ones((len(dist),))
            else:
                sig_mask = dist[:, noncvx_dim + cvx_dim]
            dist = dist.detach()
            dist.requires_grad = True
            eval_data["total"][name] = {
                "conds": dist[:, :noncvx_dim],
                "transport": dist[:, noncvx_dim : noncvx_dim + cvx_dim],
                "sig_mask": sig_mask.bool(),
            }
    return eval_data


def load_ot_model(
    path: str, device="cpu", file_type="yml", index_to_load=-1, missing_kwargs=None,
    verbose=False,
):
    """
    Load an OT model given a path to the model.

    Args:
        path (str): The path to the model.
        device (str, optional): The device to load the model on. Defaults to "cpu".
        file_type (str, optional): The file type of the model configuration file. Defaults to "yml".
        index_to_load (int, optional): The index of the best model to load. Defaults to -1.
        missing_kwargs (dict, optional): Additional keyword arguments that are missing from the model configuration file. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        tuple: A tuple containing the loaded discriminator (f) and generator models (g).
    """
    if missing_kwargs is None:
        missing_kwargs = {}
    # load logs
    # log = misc.load_json(f"{path}/log.json")
    # find best index
    if file_type in ("yml", "yaml"):
        try:
            model_args = misc.load_yaml(f"{path}/model_config.{file_type}")
        except:
            model_args = misc.load_yaml(f"{path}/.hydra/config.yaml")
            model_args = model_args.model
    elif file_type == "json":
        model_args = misc.load_json(f"{path}/model_config.{file_type}")
    model_args["device"] = device

    # getting the model
    if any(["model_config" in i for i in glob(f"{path}/*.y*ml")]): 
        model_args = misc.load_yaml(f"{path}/model_config.{file_type}")
        model_args["device"] = device
        w_disc = PICNN(**model_args)
        generator = PICNN(**model_args)
    else: # hydra/omegaconf
        model_args = misc.load_yaml(glob(f"{path}/.hydra/config*")[0]).model
        model_args["device"] = device
        model_args["verbose"] = verbose
        w_disc = hydra.utils.instantiate(copy.deepcopy(model_args))
        generator = hydra.utils.instantiate(copy.deepcopy(model_args))

    # load trained model
    path_to_best_model = sorted(
        glob(f"{path}/training_setup/*"), key=os.path.getmtime
    )[index_to_load]
    if verbose:
        print(f"Best OT model: {path_to_best_model}")
    parameters = torch.load(path_to_best_model, map_location=model_args["device"])
    w_disc.load_state_dict(parameters["f_func"])
    generator.load_state_dict(parameters["g_func"])

    return w_disc, generator

def load_model_w_hydra(path:str, index_to_load:int=-1, device:str=None) -> Tuple[torch.nn.Module, torch.nn.Module]:
    "load model that has been trained using hydra"
    model_cfg = misc.load_yaml(f"{path}/.hydra/config.yaml")
    
    f_network = hydra.utils.instantiate(model_cfg.model,
                                        device=device)
    g_network = hydra.utils.instantiate(model_cfg.model,
                                        device=device)

    # load log
    try:
        log = misc.load_json(path+"/log.json")
        if np.max(log["loss_f_abs_log"][-25:]) > 5:
            index_to_load = np.argmin(log["AUC"])
            logging.warning(f"Model seems to have diverged, loading model with lowest AUC instead of lowest loss. That is index: {index_to_load}")
    except (JSONDecodeError, KeyError):
        logging.warning(f"Could not find log.json in {path}, running with index_to_load={index_to_load}")        

    checkpoint = sorted(glob(f"{path}/training_setup/*"),
                        key=os.path.getctime)[index_to_load]
    _checkpoint = sorted(glob(f"{path}/training_setup/*"),
                        key=os.path.getctime)[index_to_load]

    if checkpoint!=_checkpoint:
        raise ValueError(f"Checkpoint: {checkpoint} is not the same as _checkpoint: {_checkpoint}. Modification is different from creation time.....")

    print(f"Checkpoint: {checkpoint}")
    parameters = torch.load(checkpoint,
                            map_location=model_cfg.device if device is None else device)

    if isinstance(f_network, partial):
        # get convex dimension
        cvx_dim = parameters["f_func"]['layer_zz.0.linear_layer'].shape[-1]

        # get non-convex dimension
        noncvx_dim=0
        if "layer_uutilde.0.weight" in parameters["f_func"]:
            noncvx_dim = parameters["f_func"]['layer_uutilde.0.weight'].shape[-1]

        f_network = f_network(noncvx_dim=noncvx_dim, cvx_dim=cvx_dim)
        g_network = g_network(noncvx_dim=noncvx_dim, cvx_dim=cvx_dim)

    f_network.load_state_dict(parameters["f_func"])
    g_network.load_state_dict(parameters["g_func"])
    return f_network, g_network

def get_optimizer(parameters, optim_args: dict, sch_args: dict = None):
    """build the optimisers
        This should properly be a loop instead of hard code twice
    Parameters
    ----------
    init_new : bool, optional
        if you want to init a new optimizer during training, by default False
    """
    if ("name" not in optim_args) or ("args" not in optim_args):
        raise ValueError("optim_args missing either name or args")

    if optim_args["name"].lower() == "adamw":
        optimizer = torch.optim.AdamW(parameters, **optim_args["args"])
    elif optim_args["name"].lower() == "nadam":
        optimizer = torch.optim.NAdam(parameters, **optim_args["args"])
    elif optim_args["name"].lower() == "sgd":
        optimizer = torch.optim.SGD(parameters, **optim_args["args"])
    else:
        optimizer = torch.optim.Adam(parameters, **optim_args["args"])

    if sch_args is not None:
        scheduler = get_scheduler(optimizer, sch_args)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    return optimizer, scheduler

class CosineAnnealingLRContinueFlat(CosineAnnealingLR):
    """
    Custom learning rate scheduler extending CosineAnnealingLR. Continues with a flat learning rate after the cosine annealing schedule.

    Args:
        Refer to CosineAnnealingLR for detailed arguments.

    Methods:
        get_lr(): Computes the learning rate at each step.
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            return (self.eta_min if self.eta_min==0 
                    else [group['lr'] for group in self.optimizer.param_groups])
        else:
            return super().get_lr()

def get_scheduler(optimizer, sch_args):
    """Setting the correct lr scheduler

    Raises
    ------
    AttributeError
        If the lr scheduler is unknown
    """
    if ("name" not in sch_args) or ("args" not in sch_args):
        raise ValueError("sch_args missing either name or args")

    if sch_args["name"].lower() == "exponentiallr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, **sch_args["args"]
        )
    elif sch_args["name"].lower() == "onecyclelr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **sch_args["args"])

    elif sch_args["name"].lower() == "cyclelr":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **sch_args["args"])
    elif sch_args["name"].lower() == "CosineAnnealingWarmRestarts".lower():
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, **sch_args["args"]
        )
    elif sch_args["name"].lower() == "CosineAnnealingLR".lower():
        if "T_max" not in sch_args["args"]:
            raise ValueError("T_max missing from lr_scheduler")
        scheduler = CosineAnnealingLRContinueFlat(
            optimizer=optimizer, **sch_args["args"]
        )
    elif sch_args["name"].lower() == "ReduceLROnPlateau".lower():
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **sch_args["args"]
        )
    else:
        raise ValueError("Unknown scheduler")
    return scheduler


def define_conds_bins(
    conds: np.ndarray, names: list, ranges: list = [0, 0.25, 0.5, 0.75, 1]
):
    conds_bins = {}
    if not isinstance(names, list):
        names = [names]

    if not isinstance(ranges, torch.Tensor):
        ranges = torch.tensor(ranges).float()
    for col, name in enumerate(names):
        if ranges[-1]>1:
            values = ranges
        else:
            values = list(np.round(torch.quantile(conds[:, col], ranges).numpy(), 3))
            values[0] -= values[0] / 10
            values[-1] += values[-1] / 10
        conds_bins[name] = np.round(values, 3)
    return conds_bins


# def conds_mask(conds, conds_bins, output_bins=False):
#     """This function expect the columns order to be the same
#     as the name order from the conds_bins"""
#     if conds.shape[1] == 0:
#         if output_bins:
#             if len(conds_bins.keys()) == 0:
#                 dummy_key = ["dummy"]
#             else:
#                 dummy_key = list(conds_bins.keys())[0]
#             yield torch.ones(len(conds)) == 1, -1000, 1000, dummy_key
#         else:
#             yield torch.ones(len(conds)) == 1
#     for col, name in enumerate(conds_bins.keys()):
#         for conds_low, conds_high in zip(conds_bins[name][:-1], conds_bins[name][1:]):
#             # if len(conds_bins)>2:
#             if conds.shape[1] > 0:
#                 mask_conds = (conds[:, col] >= conds_low) & (conds[:, col] < conds_high)
#             else:
#                 mask_conds = torch.ones((len(conds))) == 1
#             if output_bins:
#                 yield mask_conds, conds_low, conds_high, name
#             else:
#                 yield mask_conds
#     # else:
#     #     yield torch.ones((len(conds), 1)) == 1


def save_config(outdir: str, values: dict, drop_keys: list, file_name: str):
    """save the model and training config files and helps with creating folders

    Parameters
    ----------
    outdir : str
        path to the output folder
    values : dict
        config values in dict form - also the function will remove all drop_keys
    drop_keys : list
        drop_keys should all non-saveable parameters in values
    file_name : str
        Name of the new saved file
    """
    for folder in ["", "models", "plots"]:
        if not os.path.exists(outdir + "/" + folder):
            os.mkdir(outdir + "/" + folder)
    for drop in drop_keys:
        values.pop(drop, None)

    misc.save_yaml(values, f"{outdir}/{file_name}.yml", hydra=True)
