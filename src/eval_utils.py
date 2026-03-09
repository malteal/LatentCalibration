# pylint: skip-file
"utils for evaluate models"
import json
import os

from tqdm import tqdm

import numpy as np
import torch

from src import utils

from tools.tools import misc
from otcalib.otcalib.utils import plotutils
import otcalib.otcalib.utils.transformations as trans
import otcalib.otcalib.torch.torch_utils as ot_utils



def get_styles(path):
    return utils.get_styles(path)

def load_calibration_sample(model_path, data_size, **kwargs):
    try:
        with open(f"{model_path}/commandline_args.txt") as f:
            commandline_args = f.read()
        commandline_args = json.loads(commandline_args)
    except FileNotFoundError:
        commandline_args = misc.load_yaml(f"{model_path}/.hydra/config.yaml")
        commandline_args = commandline_args.data_cfg
    try:
        train_config = misc.load_yaml(f"{model_path}/train_config.yml")
        model_config = misc.load_yaml(f"{model_path}/model_config.yml")
    except:
        train_config = misc.load_yaml(f"{model_path}/train_config.json")
        model_config = misc.load_yaml(f"{model_path}/model_config.json")
    if "device" in kwargs:
        train_config["device"] = kwargs["device"]
    
    if kwargs.get("use_eval_sample", False):
        # torch.save(eval_data, "evaluate/eval_data.pth")
        print("Using standard eval sample")
        eval_data = torch.load("evaluate/eval_data.pth")
    else:
        batch_size = 512
        conds_var = train_config["condsnames"]
        transport_var = train_config["distnames"]
        sample_args = {
            "probnames": transport_var,
            "condnames": conds_var,
            "train_size": 1,
            "valid_size": data_size,
            "sig_bkg_selection": "sig",
        }
        sample_args["maxevents"] = data_size
        # pt_bins = np.array([20, 30, 40, 60, 85, 110, 140, 175, 250, 600])
        # pt_bins = {"jet_pt": np.log(pt_bins)}
        args={}
        if "conds_bins" in kwargs:
            args["conds_bins"] = kwargs["conds_bins"]
        if "conds_range" in kwargs:
            args["conds_range"] = kwargs["conds_range"]

        if "template_kwargs" in commandline_args:
            args["template_kwargs"] = commandline_args["template_kwargs"]
            
        # commandline_args["source_path"] = "/home/users/a/algren/scratch/ftag-otcalib/MC_to_data_all"
        
        args["bkg_weighting"] = False
        _, _, eval_data, _  = utils.load_data(
            sample_args=sample_args,
            source_path=commandline_args["source_path"],
            target_path=commandline_args["target_path"],
            transport_var=transport_var,
            conds_var=conds_var,
            generator_path=kwargs.get("generator_path", commandline_args["generator_path"]),
            device=train_config["device"],
            batchsize=batch_size,
            cut_in_valid_size=False,
            valid_duplications=sample_args["maxevents"]//500_000 if sample_args["maxevents"] is not None else 1,
            number_of_events=None,
            **args
        )
    
    return eval_data, train_config, model_config

def run_calibration(paths:list, data_size, best_index=-1,
                    eval_func_str="g_func", device="cuda", **kwargs):
    bootstrap_path = kwargs.get("bootstrap_path", None)
    evaluate_plot = kwargs.get("evaluate_plot", False)
    eval_data=None
    evaluate_bootstrap_dict={}
    for index, model_path in tqdm(enumerate(paths),total=len(paths)):
        name = model_path.split("/")[-1].casefold()
        if (eval_data is None):
            eval_data, train_config, model_config = load_calibration_sample(
                model_path=model_path,
                data_size=data_size,
                device=device,
                **kwargs,
            )
            if isinstance(bootstrap_path,str):
                evaluate_bootstrap_dict = {key:i.clone().detach().numpy() for key,i in eval_data[list(eval_data.keys())[0]]["Flow"].items()}

            
        # if not kwargs.get("load_data_every_time", True):
        #     total_eval_data = copy.deepcopy(eval_data)
        # if total_eval_data is not None:
        #     eval_data = copy.deepcopy(total_eval_data)
        
        if "device" in kwargs:
            model_config["device"] = kwargs["device"]
        # sys.exit()
        print(
            f"MC evaluation size: {np.sum([eval_data[i]['MC']['transport'].shape[0] for i in eval_data.keys()])}"
        )
        print(
            f"Flow evaluation size: {np.sum([eval_data[i]['Flow']['transport'].shape[0] for i in eval_data.keys()])}"
        )

        # if best_index  == -1:
        #     model_checkpoint_path = sorted(glob(
        #         f"{model_path}/training_setup/checkpoint_*.pth"
        #     ), key=os.path.getmtime)[-1]
        # else:
        #     model_checkpoint_path = (
        #         f"{model_path}/training_setup/checkpoint_{best_index}.pth"
        #     )
        # config_path = f"{model_path}/model_config.yml"
        # auc = misc.load_json(f"{model_path}/log.json")["AUC"]
        # best_index = best_index if auc[best_index]<0.6 else np.argmin(auc[25:]) # 25 burning in phase
        f_func, g_func = ot_utils.load_ot_model(model_path, index_to_load=best_index,
                                                device=device)
        # print(f"Running path {model_checkpoint_path}".center(50, "-"))
        # continue
        f_func.eval()
        g_func.eval()
        # sys.exit()
        if eval_func_str == "f_func":
            eval_func = f_func
        elif eval_func_str == "g_func":
            eval_func = g_func
        else:
            raise ValueError(f"Unknown eval_func_str {eval_func_str}")
        # eval_func.scaler = source.sig_scaler

        if (index not in evaluate_bootstrap_dict) and isinstance(bootstrap_path,str):
            evaluate_bootstrap_dict[index] = {}
        for name, values in eval_data.items():
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
            for sub_name, sub_values in values.items():
                if sub_name.lower() == "truth":
                #     if (
                #         index not in evaluate_bootstrap_dict
                #     ) and isinstance(bootstrap_path,str):
                #         evaluate_bootstrap_dict[sub_name] = {
                #             name: i.clone().detach().numpy()
                #             for name, i in sub_values.items()
                #         }
                    continue
                output_dir = (
                    f"{model_path}/{eval_func_str}_{sub_name}_eval_testing/"
                )
                
                # if "MC" in sub_name:
                    # continue
                
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                n_chunks = len(sub_values["conds"])//25_000+1
                cal_ten = eval_func.chunk_transport(sub_values["conds"],
                                                    sub_values["transport"],
                                                    sub_values["sig_mask"],
                                                    n_chunks=n_chunks,
                                                    )
                cal_ten = trans.logit(
                    trans.probsfromlogits(cal_ten)
                )
                sub_values["eval_transport"] = cal_ten.cpu().clone().detach()
                
                print(f"Size of data: {len(sub_values['eval_transport'])}, {len(values['truth']['transport'])}")
                if len(sub_values["eval_transport"]) == len(values["truth"]["transport"]):
                    dl1r_trans = trans.dl1r(trans.probsfromlogits(sub_values["eval_transport"]))
                    dl1r_origin = trans.dl1r(trans.probsfromlogits(sub_values["transport"]))
                    dl1r_truth = trans.dl1r(trans.probsfromlogits(values["truth"]["transport"]))
                    print(sub_name, name)
                    print(f"DL1r origin: {torch.abs(dl1r_origin.sort(0)[0]-dl1r_truth.sort(0)[0]).mean()}")
                    print(f"DL1r transport: {torch.abs(dl1r_trans.sort(0)[0]-dl1r_truth.sort(0)[0]).mean()}")
                # continue
                if isinstance(bootstrap_path,str):
                    # if sub_name not in evaluate_bootstrap_dict[index]:
                    #     evaluate_bootstrap_dict[index] = {}
                    evaluate_bootstrap_dict[index][sub_name] = sub_values["eval_transport"].clone().detach().numpy()#{
                        # name: i.clone().detach().numpy() for name, i in sub_values.items()
                    # }
                elif evaluate_plot:
                    probnames, condnames = ot_utils.get_names(
                        datatype="cern",
                        convex_dimensions=train_config.cvx_dim,
                        nonconvex_dimensions=train_config.noncvx_dim,
                    )
                    plotutils.plot_callback(
                        g_func=eval_func,
                        sources={i: j.clone() for i,j in sub_values.items()},
                        targets=values["truth"],
                        nonconvex_inputs=[
                            "log_" + x if "pt" in x else x for x in condnames
                        ],
                        convex_inputs=[i.replace("$", "") for i in probnames],
                        writer=None,
                        epoch=f"best_better_{best_index}",
                        outfolder=output_dir,
                        prefix=f"{name}_{best_index}",
                        correlation_bool=False,
                    )


        # keys = misc.get_dict_keys(eval_data)
        # input_data = misc.get_data_from_dict(eval_data, keys[2].split("...")).detach().numpy()
        # misc.input_value_into_dict(eval_data, input_data, keys=keys[2].split("..."))

        # save prediction
        if not isinstance(bootstrap_path, str):
            total_eval_path = f"{model_path}/model_transport_from_best_model.npy"
            for conds_key in eval_data: # WTF!!!! jezu
                for model_key in eval_data[conds_key]:
                    for sub_key in eval_data[conds_key][model_key]:
                        eval_data[conds_key][model_key][sub_key] = (
                            eval_data[conds_key][model_key][sub_key].clone().cpu().detach().numpy()
                            )
            np.save(total_eval_path, eval_data)
    if isinstance(bootstrap_path, str):
        np.save(bootstrap_path, evaluate_bootstrap_dict)
        return evaluate_bootstrap_dict
    else:
        return eval_data
