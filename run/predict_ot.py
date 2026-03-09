"run calibration for ot path"
from copy import deepcopy
import pyrootutils


root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import os
import hydra
import numpy as np
import torch as T
from typing import List
from glob import glob
import pandas as pd
import h5py
import logging
from tqdm import tqdm

# ot-framework
from otcalib.otcalib.torch.torch_utils import load_model_w_hydra

# internal
from tools.tools import misc

log = logging.getLogger(__name__)

def pack_data(*args, names:List[str], 
              transport_names:List[str], conds_names:List[str],
              weights=None, additional_info:dict=None) -> dict:
    data={}
    cvx_dims = len(transport_names)
    noncvx_dims = len(conds_names)

    for nr, (i,j) in enumerate(zip(args, names)):
        data[j] = {}
        eval_transport = None

        if isinstance(i, list):
            i, eval_transport = i
            key = j.split('_')

            # making sure eval_transport is not overwritten
            if len(key)==2:
                key = f"eval_transport_{key[-1]}"
            else:
                key = "eval_transport"
            data[key]={}
        
        if isinstance(i, T.Tensor):
            i = i.cpu().detach().numpy()

        transport = i[:, noncvx_dims:noncvx_dims+cvx_dims]
        for col in range(cvx_dims):
            data[j][transport_names[col]] = transport[:, col]

        conds = i[:, :noncvx_dims]
        for col in range(noncvx_dims):
            data[j][conds_names[col]] = conds[:, col]

        if noncvx_dims+cvx_dims+1==i.shape[1]:
            data[j]["sig_mask"] = i[:, noncvx_dims+cvx_dims]==1
        else:
            data[j]["sig_mask"] = np.ones(len(i), dtype=bool)

        if weights is not None:
            data[j]["weights"] = weights[nr]

        if eval_transport is not None:
            eval_transport = eval_transport.cpu().detach().numpy()
            for col in range(cvx_dims):
                data[key][transport_names[col]] = eval_transport[:, col]
            data[key] = pd.DataFrame(data[key])
        
        if (additional_info is not None) and (not 'Flow' in names):
            for name, info in additional_info[nr].items():
                data[j][f"additional_{name}"] = info
        data[j] = pd.DataFrame(data[j])

    return data

def save_prediction_to_h5(data: dict, path: str) -> None:
    with h5py.File(path, 'w') as hf:
        for key, value in tqdm(data.items(), desc='Saving H5', leave=False):
            if isinstance(value, pd.DataFrame):
                # Convert DataFrame to NumPy array
                value_array = value.values
                # Create dataset
                dataset = hf.create_dataset(key, data=np.float64(value_array))
                # Save column names as an attribute
                dataset.attrs['columns'] = value.columns.to_list()
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    # Flatten the structure by combining the keys
                    hf.create_dataset(f'{key}/{subkey}', data=np.float64(subvalue))
            else:
                hf.create_dataset(key, data=np.float64(value))

def load_prediction_from_h5(path: str) -> dict:
    data = {}
    with h5py.File(path, 'r') as hf:
        for key in hf.keys():
            group = hf[key]
            if isinstance(group, h5py.Group):
                data[key] = {}
                for subkey in group.keys():
                    dataset = group[subkey]
                    if 'columns' in dataset.attrs:
                        columns = dataset.attrs['columns']
                        data[key][subkey] = pd.DataFrame(dataset[:], columns=columns)
                    else:
                        data[key][subkey] = dataset[:]
            else:
                dataset = group
                if 'columns' in dataset.attrs:
                    columns = dataset.attrs['columns']
                    data[key] = pd.DataFrame(dataset[:], columns=columns)
                else:
                    data[key] = dataset[:]
    return data

class PredictionHandler:
    def __init__(self, model_path:str, sample_name:str, bkg_scaler_str:str,
                 transport_names:list=None,
                 conds_names:list=None,
                 force_prediction:bool=False,force_new_sample:bool=False, maxevents:int=None, allow_prediction:bool=True, shuffle:bool=False, save_sample_path:str=None,
                 additional_ot:dict=None, **kwargs) -> None:
        self.conds_names=conds_names
        self.transport_names=transport_names
        self.model_path = model_path
        self.force_prediction=force_prediction
        self.maxevents=maxevents
        self.bkg_scaler_str=bkg_scaler_str
        self.allow_prediction=allow_prediction
        self.shuffle = shuffle
        self.force_new_sample=force_new_sample
        self.norm_func = kwargs.get("norm_func", None)
        self.additional_ot = additional_ot
        self.save_sample_path=save_sample_path
        self.index_to_load=kwargs.get('index_to_load', -1) # -1 to get the last
        self.duplications = kwargs.get("duplications", 1)
        self.additional_columns = kwargs.get("additional_columns", [])
        self.loader_args = kwargs.get("loader_args", {})
        self.glob_path = kwargs.get("glob_path", None)
        self.glob_samples = kwargs.get("glob_samples", None)
        
        self.prediction_path = f"{model_path}/transport_files/"
        os.makedirs(self.prediction_path, exist_ok=True)
        
        # load model config
        self.model_cfg = misc.load_yaml(f"{self.model_path}/.hydra/config.yaml")

        # check if same norm_path otherwise define norma path
        if self.norm_func is not None:
            norm_path = self.norm_func.keywords.get('norm_path')
            if norm_path != self.model_cfg.data.target_sample.norm_path:
                self.norm_func.keywords['norm_path'] = self.model_cfg.data.target_sample.norm_path
                log.info(f"Norm path not the same as model path, using {self.model_cfg.data.target_sample.norm_path} instead")

        # if save_sample_path not defined, try to get it from the glob_path
        if self.save_sample_path is None: 
            
            # TODO not a good way to do this
            sample_path = self.model_cfg.data.source_sample.data[0].path
            save_sample_path = sample_path.split(self.glob_samples)[-1].split('/')[0]
            
            self.save_sample_path=f"{self.glob_samples}/{save_sample_path}"
            log.info(f"Save sample path not defined, using {self.save_sample_path} instead")

        # files to load
        self.name = ""
        self.name += f"{sample_name}_"
        self.name += self.bkg_scaler_str 
        self.sample_name=sample_name.split("_")[0] # remove the "_uniform"

        self.addi_ot_model=None
        if "mc" in self.sample_name.lower():
            self.duplications=1

        # For hadronic recoil - load ot model
        if (self.additional_ot is not None) and ('mc' in self.sample_name.lower()):
            self.addi_ot_model = load_model_w_hydra(self.additional_ot['path'], index_to_load=-1)[-1]
            
        
        self.check_for_predictions()
        
        if (not self.predictions_exists and self.allow_prediction) | self.force_prediction:
            log.info("Predicting...")

            self.create_prediction()

    def check_for_predictions(self):
        self.predictions_exists = os.path.exists(f"{self.prediction_path}/{self.name}.h5")
    
    def save_prediction(self, data:dict) -> None:
        
        # save the prediction to save_path
        save_path = f"{self.prediction_path}/{self.name}.h5"

        logging.info(f"Saving prediction to {save_path}")

        save_prediction_to_h5(data, save_path)
    
    def load_prediction(self) -> dict:
        try:
            return load_prediction_from_h5(f"{self.prediction_path}/{self.name}.h5")
        except FileNotFoundError:
            possible_paths = glob(f'{self.prediction_path}/'+self.name.split("_")[0]+"*")
            raise FileNotFoundError(f"No prediction found for {self.name}.\n Possible {self.name.split('_')[0]} paths: {possible_paths}")
    
    def create_datasets(self, model_path:str=None):
        
        if model_path is None:
            model_path = self.model_path

        # load model config
        model_cfg = deepcopy(self.model_cfg)

        if 'condnames' in model_cfg.data:
            self.conds_names = model_cfg.data.condnames
        
        if 'transportnames' in model_cfg.data:
            self.transport_names = model_cfg.data.transportnames
        
        # setting for the template generator
        if 'template_args' in model_cfg.data:
            # deactivate any bkg downscaling
            model_cfg.data.template_args.use_bkg_weighting = self.bkg_scaler_str
            model_cfg.data.template_args.duplications = 1
            model_cfg.data.template_args.downscale_bkg_ratio = 1
        
            # set the number of events to evaluate
            model_cfg.data.sample_args.maxevents = self.maxevents
        
            # ensure bootstraps are not used
            model_cfg.data.bootstraps_iters=0
        
        if self.save_sample_path is not None:
            save_path = f"{self.save_sample_path}/{self.name}.h5"
        else:
            os.makedirs(f"{model_path}/transport_files/samples/", exist_ok=True)
            save_path = f"{model_path}/transport_files/samples/{self.name}.h5"
        
        # get samples
        if not os.path.exists(save_path) or self.force_new_sample:
            if "Flow"  in self.name and 'all' in self.name:
                model_cfg.data.template_args.duplications=self.duplications
            dataset = hydra.utils.instantiate(
                model_cfg.data,
                # valid_size=None,
                get_eval_data=False,maxevents=self.maxevents,
                shuffle=self.shuffle,
                use_bkg_weighting = self.bkg_scaler_str,
                additional_columns=self.additional_columns,
                **self.loader_args
                )

            source, target, _, _ = dataset()
            
            # generated uniform pT samples
            if "all" not in self.name:
                raise NotImplementedError("You have to implement your own conditional sampling")
            
            target_weights = np.ones(len(target.data))
            if hasattr(dataset, 'target_weights'):
                target_weights = dataset.target_weights
            
            # save flow generated samples
            data = {"Data": {"sample": target.data.numpy(),
                                "weights": target_weights},
                    'Data_valid': {"sample": dataset.valid_target_sample.numpy(),
                                "weights": np.ones(len(dataset.valid_target_sample))}
                    }
            
            if "mc" in self.name.lower():
                valid_source_original = dataset.original_sample.data
                data["MC"] =  {"sample":valid_source_original.data.numpy(), 
                                "weights": dataset.original_weights}

                if len(self.additional_columns):
                    # duplicate the target data same as flows is doing
                    csv_target = pd.concat([dataset.original_df], ignore_index=True)
                    
                    for i in self.additional_columns:
                        data["MC"][i] = csv_target[i].values
            else:
                if np.all(target_weights != 1):
                    raise ValueError("Weights are not all 1 - not supported for flow atm. when duplications are used we assume that all weights are 1")

                data[self.sample_name] = {
                    "sample": source.data.numpy(),
                    "weights": np.ones(len(source.data))}

                data[f"{self.sample_name}_valid"] = {
                    "sample": dataset.valid_source_sample.numpy(),
                    "weights": np.ones(len(dataset.valid_source_sample))
                    }

            if len(self.additional_columns) and (not np.in1d('Flow', self.sample_name)):
                # duplicate the target data same as flows is doing
                csv_target = pd.concat([dataset.target_df], ignore_index=True)

                for i in self.additional_columns:
                    data["Data"][i] = csv_target[i].values

            # specific for hadronic recoil
            if self.addi_ot_model is not None:
                sample = T.tensor(data["MC"]["sample"]).float()[:,:2]
                data["MC"]["sample"][:, 1:2] = self.addi_ot_model.chunk_transport(
                    totransport = sample[:, 1].reshape(-1,1),
                    conditionals = sample[:, 0].reshape(-1,1),
                    sig_mask = T.tensor(data["MC"]["sample"])[:, -1].bool(),
                    n_chunks=len(data["MC"]["sample"])//1024,
                    pbar_bool=True)
            
            save_prediction_to_h5(data, save_path)

        return save_path
 
    
    def create_prediction(self, new_data_sample:dict=None) -> None:
        
        # load ot model
        ot_model = load_model_w_hydra(self.model_path, 
                                      index_to_load=self.index_to_load)[-1]

        # predict new data
        if new_data_sample is None:
            # create datasets for eval
            save_path = self.create_datasets()
            
            # load the eval datasets
            data = load_prediction_from_h5(save_path)

            # load the target data
            target = [T.tensor(data[i]["sample"]).float() for i in data if "Data" in i]
            target_weights = [ data[i]["weights"] for i in data if "Data" in i]
            target_keys = [i for i in data if "Data" in i]
            
            # load the transport data
            trans_dist = [T.tensor(data[i]["sample"]).float() for i in data if self.sample_name in i]
            weights = [ data[i]["weights"] for i in data if self.sample_name in i]
            trans_keys = [i for i in data if self.sample_name in i]


        else:
            raise NotImplementedError("You have to implement your own conditional sampling")
            # load data from previous prediction
            # source = T.tensor(np.c_[new_data_sample[self.sample_name]["conds"],
            #                new_data_sample[self.sample_name]["transport"],
            #                new_data_sample[self.sample_name]["sig_mask"]*1]).float()

        noncvx_dim = ot_model.noncvx_dim
        cvx_dim = ot_model.cvx_dim
        
        if self.transport_names is None and cvx_dim>8:
            self.transport_names = [f"latn{i}" for i in range(cvx_dim)]
        elif self.transport_names is None:
            raise ValueError("You have to specify the transport and maybe conds names")

        if (self.model_cfg.data.get('trans_lst', None) is not None
            and 'mc' in self.name.lower()):
            if 'undo' not in self.model_cfg.data.trans_lst[0][0]:
                trans_dist[:, :noncvx_dim+cvx_dim] = transform_data(trans_dist[:, :noncvx_dim+cvx_dim],
                self.model_cfg.data.trans_lst)
        
        # transport flow
        transport = []
        for i in range(len(trans_dist)):
            if trans_dist[i].shape[-1] == noncvx_dim+cvx_dim:
                sig_mask = T.ones(len(trans_dist[i]))==1
            else:
                sig_mask = trans_dist[i][:, -1].bool()
            transport.append(
                ot_model.chunk_transport(
                totransport = trans_dist[i][:, noncvx_dim:noncvx_dim+cvx_dim],
                conditionals = trans_dist[i][:, :noncvx_dim],
                sig_mask = sig_mask,
                n_chunks=(len(trans_dist[i])//10240+1),
                pbar_bool=True)
                )

        # first list is source+transport, second is target
        args = [[trans_dist[i], transport[i]] for i in range(len(trans_dist))]+target

        names = trans_keys+target_keys

        weights = weights+target_weights

        if new_data_sample is None:

            # pack the newly predicted data

            additional_info = [
                {i:j for i,j in data[i].items() if not np.isin(i, ["sample", "weights"])} for i in names
            ]

            data = pack_data(*args, names=names,
                             transport_names = self.transport_names,
                             conds_names = self.conds_names,
                             weights=weights,
                             additional_info=additional_info)
        else:
            # update the data with the new predictions
            data[self.sample_name]["eval_transport"] = transport.numpy()
        
        if self.norm_func is not None:
            columns = data['eval_transport'].columns
            for i,j in data.items():
                if len(j)==0: continue
                data[i].loc[:,columns] = np.float32(self.norm_func(j[columns].values))

        # save predictions
        self.save_prediction(data)


@hydra.main(config_path=str(root / "configs/"), config_name="predict_ot", version_base=None)
def main(config):
    # %matplotlib widget

    transport_handler = hydra.utils.instantiate(config)

    # load the prediction
    data = transport_handler.load_prediction()
    
    # print predictions
    print(data)
    
if __name__ == '__main__':
    main()
