"evaluate ot ftag calibration"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import matplotlib.pyplot as plt
import numpy as np
from typing import List
import pandas as pd
import logging
from json import JSONDecodeError

# ot-framework
# internal
from tools.tools import misc
from tools.tools.hydra_utils import hydra_init

log = logging.getLogger(__name__)


def get_model_config(paths:list, log_key_to_load:str="AUC") -> pd.DataFrame:

    model_cfg = []
    
    for nr, path in enumerate(paths):

        try:
            log = misc.load_json(f"{path}/log.json")
            log_nr =  float(log[log_key_to_load][-1])
        except (FileNotFoundError, JSONDecodeError):
            print(f"Could not find log.json in {path}")
            log_nr=None

        model_cfg.append(misc.load_yaml(f"{path}/.hydra/config.yaml")["model"])

        model_cfg[nr][log_key_to_load] = log_nr
        model_cfg[nr]["path"] = path.split("/")[-1]
    
    model_cfg = pd.DataFrame.from_dict(model_cfg)
    
    return model_cfg,log

def plot_log(paths:List, keys:List=["loss_f_abs_log", "loss_g", "g_clip", "f_clip", "lr_f", "lr_g", "AUC"], skip_val:int=0, str_in_key:str=None) -> dict:

    figure_dicts = {}

    if (str_in_key is not None) and (keys is None):
        log = misc.load_json(f"{paths[-1]}/log.json")
        keys = [key for key in log if str_in_key in key]

    for key in keys:
        figure_dicts[key] = plt.subplots(1,1)[-1]

    for path in paths:
        try:
            log = misc.load_json(f"{path}/log.json")
        except JSONDecodeError:
            print(path)
            continue
            
        # loop over keys to plot
        for key in keys:
            figure_dicts[key].plot(log[key][skip_val:],
                                    label=path.split("/")[-1])
            figure_dicts[key].set_xlabel("Epoch")
            figure_dicts[key].set_ylabel(key)
            figure_dicts[key].legend()

    return log

if __name__ == '__main__':
    # %matplotlib widget
    config = hydra_init(str(root/"configs/eval_ot.yaml"), verbose=True)

    # n_bins=10
    selection = config['type_to_eval']
    gen_name = config["evaluation_sample"]
    name_to_eval = config["name_to_eval"]

    # load log and plot log
    paths = config['paths'][selection][name_to_eval]
    model_cfg, log = get_model_config([paths])
    top_models = np.argsort(log['AUC'])[0]
    # model_cfg = model_cfg.iloc[top_models]

    skip_val=0
    log = plot_log([paths], skip_val=skip_val,
            #  keys=['transport_cost'],
            #  str_in_key="conds"
                )
    
    # plt.figure()
    # plt.plot(1-np.array(log['lr_f'][:]), log['conds']['Flow']['transport_cost'][:-1])
    # plt.figure()
    # plt.plot(log['conds']['Flow']['transport_cost'][skip_val:])
    
    
    
    
    
    
    
    
    
    
    