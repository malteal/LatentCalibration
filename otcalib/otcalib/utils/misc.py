import re
import os
import h5py
import json
import numpy as np

import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf

def load_json(name):
    with open(name, "r") as fp:
        data = json.load(fp)
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
        with open(name, "w") as outfile:
            yaml.dump(dict, outfile, default_flow_style=False)


def load_yaml(path, hydra_bool: bool = True):
    if hydra_bool:
        data = OmegaConf.load(path)
        OmegaConf.set_struct(data, False)
    else:
        with open(path, "r") as stream:
            data = yaml.safe_load(stream)
    return data

def load_h5(path) -> dict:
    """Load an h5 file.

    Parameters:
    path (str): The path and filename of the h5 file.

    Returns:
    dict: The loaded data from the h5 file.
    """
    with h5py.File(path, 'r') as f:
        data = {}
        for key in f.keys():
            data[key] = f[key][()]
    return data


def replace_symbols(string:str) -> str:
    string = re.sub("\W+", "", string)
    return string

def save_fig(
    fig,
    save_path:str,
    title: str = None,
    close_fig:bool=True,
    save_args:dict={},
    replace_symbols_bool:bool=True,
    **kwargs,
) -> None:
    if replace_symbols_bool:
        save_path = save_path.split("/")
        file_name = save_path[-1].split(".")
        file_name[0] = replace_symbols(file_name[0])
        save_path[-1] = ".".join(file_name)
        # for i,j in replace_symbols.items():
        #     save_path[-1] = save_path[-1].replace(i, j)
        save_path = "/".join(save_path)
    if title is not None:
        plt.title(title)
    if kwargs.get("tight_layout", True):
        plt.tight_layout()
    plt.savefig(save_path, dpi=500, facecolor="white", transparent=False, **save_args)
    if close_fig:
        plt.close(fig)
        
def sort_by_creation_time(lst):
    return sorted(lst, key=os.path.getctime)
    
def generate_idx_given_probs(weights:np.ndarray, size:int=None):

    if isinstance(weights, list):
        weights = np.array(weights)
        
    if size is None:
        size = len(weights)

    return np.random.choice(np.arange(len(weights)), size,p=weights/np.sum(weights))

