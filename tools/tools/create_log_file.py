import numpy as np
import json
from glob import glob

def load_log(path):
    "Load log file"
    with open(path, "r+") as fp:
        log_file = json.load(fp)
    return log_file


class CreateLog():
    """This class is used to create a log file
    """
    def __init__(self, path:str, keys:list, types:list=None, wandb=None, path_old_log=None):
        """init log file

        Parameters
        ----------
        path : str
            path where to save it
        keys : list
            the keys with the log file
        types : list, optional
            IMPORTANT! This should not be a list of string but list of direct types. EXAMPLE: [[], "", np.array([])]
        wandb : _type_, optional
            To activate wandb or not, by default not
        """
        if types is None:
            types = [[] for i in range(len(keys))]
        self.path = path
        if path_old_log is not None:
            self.log = load_log(path_old_log)
        else:
            self.log = dict(zip(keys,types))
        self.wandb = wandb
    
    def _run_wandb(self, logging_values, nested=False):
        "if WandB is activated"
        if self.wandb is not None:
            if nested:
                pass
            self.wandb.log(logging_values)

    def keys(self):
        "Not working for nested keys" #TODO need to work for nested keys
        return list(self.log.keys())

    def save(self):
        "save the log file in the path"
        with open(f"{self.path}/log.json", "w") as fp:
            json.dump(self.log,fp)

    def update_log(# pylint: disable=too-many-arguments
        self,
        logging_values,
        nested=False
    ) -> dict:
        """Saving training information

        Parameters
        ----------
        logging_values : dict
            dict of lists of values that should be logged

        Returns
        -------
        dict
            return dict where the new metrics have been added
        """
        self._run_wandb(logging_values, nested)
        if "epoch" in logging_values.keys():
            logging_values.pop("epoch")

        keys = np.array(list(logging_values.keys()))

        sub_keys = keys[np.in1d(keys, list(self.log.keys()))]
        mask_dict = np.array([isinstance(logging_values[i], dict) for i in sub_keys])
        if (len(sub_keys) > 0) and any(mask_dict):
            for i in sub_keys[mask_dict]:
                sub_keys_logging = np.array(list(logging_values[i].keys()))
                sub_keys_logging = sub_keys_logging[~np.in1d(sub_keys_logging, list(self.log[i].keys()))]
                for j in sub_keys_logging:
                    self.log[i][j] = []

        add_keys = keys[~np.in1d(keys, list(self.log.keys()))]
        for i in add_keys:
            if isinstance(logging_values[i], dict):
                self.log[i] = {}
                for j in logging_values[i].keys():
                    self.log[i][j] = []
            else:
                self.log[i] = []

        
        for i,j in logging_values.items():
            if isinstance(j, dict):
                for j,k in j.items():
                    if isinstance(k, list):
                        self.log[i][j].append(k)
                    else:
                        self.log[i][j].append(np.float64(k))
            else:
                self.log[i].append(np.float64(j))


