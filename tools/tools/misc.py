import os
import re
import io
from itertools import tee
import logging

from tqdm import tqdm
import yaml
import h5py
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
import json
from pathlib import Path

log = logging.getLogger(__name__)

def create_gif(imgs_paths, outdir_name, **kwargs):
    imgs = (Image.open(f) for f in imgs_paths)
    img = next(imgs)  # extract first image from iterator
    if not ".gif" in outdir_name:
        outdir_name += ".gif"
    img.save(fp=outdir_name, format='GIF', append_images=imgs,
             save_all=True, duration=kwargs.get("time_pr_image", 250), loop=0)
    
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it
    Can be used for log WANDB images
    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def replace_symbols(string:str) -> str:
    return re.sub("\W+", "", string)

def prepare_save_path(save_path:str, replace_symbols_bool:bool=True) -> str:
    """
    Prepare the save path by replacing symbols and removing "mathrm" if specified.

    Args:
        save_path (str): The original save path.
        replace_symbols_bool (bool, optional): Whether to replace symbols in the file name. Defaults to True.

    Returns:
        str: The prepared save path.
    """
    
    if replace_symbols_bool:
        save_path = save_path.split("/")
        file_name = save_path[-1].split(".")
        file_name[0] = replace_symbols(file_name[0])
        save_path[-1] = ".".join(file_name)
        save_path = "/".join(save_path)
        save_path = save_path.replace("mathrm", "")
        
    return save_path

def save_fig(fig, save_path, title:str=None, close_fig=True,
             save_args={}, replace_symbols_bool=True, **kwargs):
    """
    Save a matplotlib figure to a file.

    Parameters:
    - fig: The matplotlib figure object to save.
    - save_path: The path to save the figure to.
    - title: Optional title for the figure.
    - close_fig: Whether to close the figure after saving.
    - save_args: Additional arguments to pass to the `savefig` function.
    - replace_symbols_bool: Whether to replace symbols in the save path.
    - **kwargs: Additional keyword arguments to pass to the `savefig` function.

    Returns:
    None
    """
    if 'None' in save_path:
        raise ValueError("None in save_path - pply not a valid path")
    
    save_path = prepare_save_path(save_path, replace_symbols_bool)

    if title is not None:
        plt.title(title)
    if kwargs.get("tight_layout", True):
        plt.tight_layout()

    logging.info(f"Saving to {save_path}")

    plt.savefig(save_path, dpi=500, facecolor='white', transparent=False, **save_args)

    if close_fig:
        plt.close(fig)

def gaussian2d(mean, cov, graph_size):
    return np.random.multivariate_normal(mean, cov, graph_size)

def save_json(dict, name):
    with open(name, "w+") as fp:
        json.dump(dict,fp)

def load_hdf(filename):
    return h5py.File(filename,'r')   
        
def load_json(name):
    with open(name, "r") as fp:
        data = json.load(fp)
    return data

def save_h5(values:dict, path:str) -> None:
    """Save a dictionary as an h5 file.

    Parameters:
    values (dict): The dictionary to be saved.
    path (str): The path and filename of the h5 file.
    """
    with h5py.File(path, 'w') as f:
        for key, value in values.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        raise ValueError("Only two levels of nesting are supported.")
                    f.create_dataset(f"{key}/{subkey}", data=subvalue)
            f.create_dataset(key, data=value)

def load_h5(path , keys:list=None) -> dict:
    """Load an h5 file.

    Parameters:
    path (str): The path and filename of the h5 file.

    Returns:
    dict: The loaded data from the h5 file.
    """
    with h5py.File(path, 'r') as f:
        data = {}
        for key in f.keys() if keys is None else keys:
            data[key] = f[key][()]
    return data


def save_yaml(dict, name, hydra=True):
    """save data as yaml    

    Parameters
    ----------
    dict : data
        data that should be saved
    name : str
        path.yml where it should be saved
    """
    if hydra:
        # dumps to file:
        with open(name, "w") as f:
            OmegaConf.save(dict, f)
    else:
        with open(name, 'w') as outfile:
            yaml.dump(dict, outfile, default_flow_style=False)

def load_yaml(path, hydra_bool:bool=True):
    """load yaml file with OmegaConf or not"""
    if hydra_bool:
        data = OmegaConf.load(path)
        OmegaConf.set_struct(data, False)
    else:
        with open(path, 'r') as stream:
            data = yaml.safe_load(stream)
    return data

def save_pkl(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_dict_keys(dictionary:dict):
    "Takes in a dict and return all keys - sub keys separated with ..."
    result = []
    for key, value in dictionary.items():
        if isinstance(value, dict): # nested loop
            new_keys = get_dict_keys(value)
            result.append(key)
            for innerkey in new_keys:
                result.append(f'{key}...{innerkey}')
        else:
            result.append(key)
    return result

def get_data_from_dict(data:dict, keys:list):
    "Given a dict and list of keys - return the value in keys"
    result = data.copy()
    for i in keys:
        result = result[i]
    return result

def sort_by_creation_time(lst):
    return sorted(lst, key=os.path.getctime)

def generate_idx_given_probs(weights:np.ndarray, size:int=None):

    if isinstance(weights, list):
        weights = np.array(weights)
        
    if size is None:
        size = len(weights)

    return np.random.choice(np.arange(len(weights)), size,p=weights/np.sum(weights))

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def pairwise(iterable, verbose:bool=False): #-> zip[tuple]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ...
    in python 3.10 itertools has a pairwise function
    """
    a, b = tee(iterable)
    next(b, None)
    return tqdm(zip(a, b), disable=not verbose,
                total=len(iterable)-1)

def save_declaration(path: str = "done.txt") -> None:
    """
    FROM matts MLTOOLS
    Save a simple file which declares a job to be finished.

    This is typically called at the end of training for pipelines managers such as
    snakemake to be able to track the progress of the pipeline.
    """
    with open(Path(path).with_suffix(".txt"), "w") as f:
        # Add some text describing the job
        for text in [
            "This file was created to declare the job as finisehd!\n",
            "This is often called at the end of training for workflow managers\n",
            "like snakemake to be able to track the progress of the pipeline.\n"
        ]:
            f.write(text)

def merge_dicts(old_cfg, new_cfg, swap_priority_keys=[], verbose:bool=False
                ):
    """Merges old_cfg into new_cfg, prioritizing values from new_cfg unless specified.

    Args:
        old_cfg (DictConfig): The older configuration.
        new_cfg (DictConfig): The newer configuration.
        swap_priority_keys (list): A list of keys where the priority should be swapped.

    Returns:
        DictConfig: The merged configuration.
    """
    
    # make sure that old_cfg and new_cfg are not struct
    if isinstance(old_cfg, DictConfig):
        OmegaConf.set_struct(old_cfg, False)

    if isinstance(new_cfg, DictConfig):
        OmegaConf.set_struct(new_cfg, False)

    for key, value in new_cfg.items():
        if verbose: log.info(key, value)
        if key in swap_priority_keys:
            # Swap priority for specified keys
            if key in old_cfg and isinstance(value, (DictConfig, dict)) and isinstance(old_cfg[key], (DictConfig, dict)):
                merge_dicts(value, old_cfg[key])
            else:
                old_cfg.update({key: value})
        else:
            # Default behavior: prioritize values from new_cfg
            if key in old_cfg and isinstance(value, (DictConfig, dict)) and isinstance(old_cfg[key], (DictConfig, dict)):
                merge_dicts(old_cfg[key], value)
            else:
                old_cfg.update({key: value})

    return old_cfg
