"flow utils"
import os
import torch as T
from glob import glob
from ..misc import load_json, load_yaml

def load_path_info(model_path, yaml_bool=False):

    model_paths = sorted(glob(model_path + "/models/*"), key=os.path.getctime)
    print(f"Best path {model_paths[-1].split('/')[-1]}")
    # if yaml_bool:
    try:
        data_settings = load_yaml(model_path + "/data_settings.yaml")
        flow_args = load_yaml(model_path + "/flow_args.yaml")
    except FileNotFoundError:
        data_settings = load_json(model_path + "/data_settings.json")
        flow_args = load_json(model_path + "/flow_args.json")

    return data_settings, flow_args, model_paths[-1]

def load_flows(flow, path, device):
    "Load the pretrain sig and bkg flows"
    _, _, model_path = load_path_info(path, yaml_bool=True)
    # load flow
    flow = flow.to(device)
    model_state = T.load(model_path, map_location=device)
    flow.load_state_dict(model_state["flow"])
    return flow