
"pipeline"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from copy import deepcopy
from glob import glob
import math
import os

import hydra
import numpy as np
import pandas as pd
import awkward as ak
import uproot
import torch as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from omegaconf import OmegaConf,DictConfig

from otcalib.otcalib.torch import loader
from src import utils

from tools.discriminator import DenseNet
from tools.visualization import general_plotting as plot
from tools import misc, hydra_utils

def get_dnn_weights(sample:np.ndarray, dense_network:DenseNet, scaler=None) -> np.ndarray:
    sample = utils.log_squash(sample)
    
    if scaler is not None:
        if isinstance(scaler,str):
            # Load the scaler object from a file
            with open(f'{scaler}/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

        # Fit and transform on train dataset
        sample = scaler.transform(sample)

    # evaluate the weights
    target_loader = T.utils.data.DataLoader(np.float32(sample),
                                            batch_size=1024,
                                            shuffle=False,)
    output=[]
    for i in target_loader:
        output.append(1/dense_network(i.to(dense_network.device))[0].detach().cpu().numpy())

    output = np.concatenate(output)

    return output   

def DAOD_calculation(batch, columns, file=None, hist_columns=None):
    try:
        batch = np.stack([ak.to_numpy(batch[i].array()) for i in columns],1)
    except: 
        batch = np.stack([ak.to_numpy(batch[i]) for i in columns],1)
    batch = pd.DataFrame(batch, columns=columns)

    if "isReco" in columns:
        mask = batch["isReco"]==1

        if not any(mask):
            return None

        if file is not None:
            if hist_columns is not None:
                hist_columns = np.intersect1d(hist_columns, file.keys())
                if len(hist_columns)>1:
                    raise ValueError("Multipe hist_columns in keys of file")
                hist_columns = hist_columns[0]
            batch["evtWeight"] = (batch["evtWeight"]*file[hist_columns].values()[-1]
                                  /batch[mask]["evtWeight"].sum())
                                    # *np.sum(batch["evtWeight"])
                                    # /np.abs(batch["evtWeight"]).sum())

        batch = batch.drop(['isReco'], axis=1)[mask]

    elif file is not None:
        batch["evtWeight"] = (batch["evtWeight"]
                                *file[hist_columns[0]].values()[-1]
                                /batch["evtWeight"].sum()
                                # *np.sum(batch["evtWeight"])
                                # /np.abs(batch["evtWeight"]).sum()
                                )
    if "dilep_pTx" in batch:
        batch["pT_ll"] = np.sqrt(np.sum(batch[["dilep_pTx", "dilep_pTy"]]**2,1))

    batch= batch[batch["sumEtUE"]>0]

    return batch

def load_DAOD_hist(paths, columns, hist_columns, shuffle:bool=False, verbose=True):
    data = []
    n_drop_evt:int = 0
    if verbose:
        print(f"Number of paths {len(paths)}")
    for path in tqdm(paths, total=len(paths), disable=not verbose):
        file = uproot.open(path)
        batch = file['MicroTree;1']["microtree;1"]
        if ak.to_numpy(batch["evtWeight"].array()).sum()==0:
            continue
        columns = np.intersect1d(columns, batch.keys())
        batch = DAOD_calculation(batch, columns, file, hist_columns)

        if batch is None:
            continue

        batch["DSID"] = np.int64(path.split("5TeV.")[-1].split(".")[0])

        data.append(batch)
    data = pd.concat(data, 0)
    if verbose:
        print(f"Number of dropped events {n_drop_evt}")
        print(f"Size of accpected evetns {data.shape}")
        print(f"Total events {n_drop_evt+data.shape[0]}")
    if "_data_" in paths[0].casefold():
        data["evtWeight"]+=1
    if shuffle:
        data = data.sample(frac = 1)
    return data.reset_index(drop=True)
 

def load_DAOD(paths, columns, chunk_size = 100_000, shuffle:bool=False, verbose=False):
    data = []
    n_drop_evt=0
    if verbose:
        print(f"Number of paths {len(paths)}")
    for batch in tqdm(uproot.iterate(paths,columns,
                                    step_size=chunk_size,
                                    # filter_name=branches,
                                    # aliases=aliases
                                    ), disable=not verbose):
        batch = DAOD_calculation(batch, columns)
        if batch is None:
            continue
        data.append(batch)
    data = pd.concat(data, 0)
    if verbose:
        print(f"Number of dropped events {n_drop_evt}")
        print(f"Size of accpected evetns {data.shape}")
        print(f"Total events {n_drop_evt+data.shape[0]}")
    if "_data_" in paths[0].casefold() and any(data["evtWeight"]==0):
        data["evtWeight"]+=1

    if shuffle:
        data = data.sample(frac = 1)
    return data.reset_index(drop=True)

def rotate_vector(vector, angle):
    x = vector[:, 0] * math.cos(angle) - vector[:, 1] * math.sin(angle)
    y = vector[:, 0] * math.sin(angle) + vector[:, 1] * math.cos(angle)
    return np.c_[x, y]

class Pipeline(loader.Dataset):
    def __init__(self, paths, columns, **kwargs) -> None:
        self.paths = paths
        self.columns=columns
        self.quantiles=None
        self.data_columns=columns[1:]
        self.correct_weight=kwargs.get("correct_weight", True)
        self.log_squash=kwargs.get("log_squash",False)
        
        self.bkg_paths =kwargs.get("bkg_paths", [])
        self.bkg_paths = [] if self.bkg_paths is None else self.bkg_paths
        # self.bkg_columns=kwargs.get("bkg_columns", [])
        
        self.sig_data={}
        self.data=None
        self.verbose=kwargs.get("verbose", True)
        self.shuffle = kwargs.get("shuffle", True)
        self.bootstrap = kwargs.get("bootstrap", False)
        self.indices=None

        print("Load signal events:")
        self.sig_data = self.load_data(self.paths, True)
        if len(self.bkg_paths)>0:
            print("Load bkg events:")
            self.bkg_data = self.load_data(self.bkg_paths, False)
        self.combine_data()
        # super().__init__() # should init Dataset when using otcalib

    def combine_data(self):
        self.data = pd.concat(self.sig_data, 0)
        if len(self.bkg_paths)>0:
            self.data = pd.concat([
                self.data,
                pd.concat(self.bkg_data, 0)], 0)

        # bootstraps
        if self.bootstrap and (self.indices is None):
            self.indices = np.random.choice(np.arange(len(self.data)), len(self.data))
            self.data = self.data.iloc[self.indices]

        # scale to log GeV
        self.col_energy = [i for i in self.data.columns
               if (("pt" in i.lower()) or ("et" in i.lower()))]

        self.data.loc[:,self.col_energy] = self.data.loc[:, self.col_energy]/1000

        if ((("boson_pTy_reco" in self.columns) and ("boson_pTx_reco" in self.columns)) or
            (("boson_pTy_truth" in self.columns) and ("boson_pTx_truth" in self.columns))):
            cols=["uT", "u_perp", "u_para", "bias"]
            self.data[cols] = 0 

            if (("boson_pTx_truth" in self.data)
                & (len(self.data["boson_type"].unique())==3)):
                # for Z
                mask_z = np.in1d(self.data["boson_type"].values, ["Zee", "Zmumu"])
                self.data[mask_z] = self.calculate_recoil(self.data[mask_z], type_to_select="reco")
                
                # for W
                self.data[~mask_z] = self.calculate_recoil(self.data[~mask_z], type_to_select="truth")
            else:
                self.data = self.calculate_recoil(self.data, type_to_select="reco")
            self.col_energy+=cols
                
        if self.log_squash:
            self.data.loc[:,self.col_energy] = utils.log_squash(self.data.loc[:,self.col_energy])

        if self.shuffle:
            self.data = self.data.sample(frac = 1)

        self.data = self.data.reset_index(drop=True)

        
    def load_data(self, paths, sig_bool):
        data=[]
        for path in tqdm(paths):
            bkg_label = "" if isinstance(path, str) else "_bkg"
            if isinstance(path, str):
                path = glob(path)
            try:
                save_path, name = path[0].split("/pT")
            except:
                print(path[0])
                sys.exit()
            name = name.split("/")[0]+bkg_label
            if len(self.bkg_paths)>0:
                
                hist_columns = [
                    "WminusmunuSelectionCutFlow",
                    "WplusmunuSelectionCutFlow",
                    'WminusenuSelectionCutFlow',
                    'WplusenuSelectionCutFlow',
                    "ZmumuSelectionCutFlow",
                    "ZeeSelectionCutFlow"]
                hist_columns = [f"{i};1" for i in hist_columns]

                samples = load_DAOD_hist(path, columns=self.columns,
                                         hist_columns=hist_columns, verbose=False,
                                         shuffle=self.shuffle)
            else:
                samples = load_DAOD(path, columns=self.columns, verbose=False,
                                    shuffle=self.shuffle)
                
            # account for negative weights
            if np.any(samples["evtWeight"]<0) & self.correct_weight:
                #calculate the absolute value of the weights
                
                if "DSID" in samples:
                    samples["abs_evtWeight"] = deepcopy(samples["evtWeight"])
                    #correct negative weights pr DSID
                    for dsid in samples["DSID"].unique():
                        mask_dsid = samples["DSID"]==dsid
                        samples.loc[mask_dsid, "abs_evtWeight"] = (
                            np.abs(samples.loc[mask_dsid, "evtWeight"])
                            *np.sum(samples.loc[mask_dsid, "evtWeight"])/np.abs(samples.loc[mask_dsid, "evtWeight"]).sum()
                            )
                else:
                    samples["abs_evtWeight"] = (
                        np.abs(samples["evtWeight"])
                        *np.sum(samples["evtWeight"])/np.abs(samples["evtWeight"]).sum()
                        )
                
                # ff_str = f"ff_corrected_{name}"
                # samples = self.calculate_recoil(samples, type_to_select="truth" if "boson_pTx_truth" in samples else "reco")
                # reweight_cols = ["pT_ll", "sumEtUE"] #, "u_perp", "bias"]
                # if (not os.path.exists(f"{save_path}/{ff_str}.h5")) | False:
                #     style = {"bins": [100, 40,
                #                     #   40,40
                #                       ], "range": [[0, 100], [0, 500],
                #                                 #    [-60, 60], [-60, 60]
                #                                    ]}   
                #     # style = {"bins": list(np.linspace(0, 20,21))+list(np.arange(25, 105, 5))+[1000]}

                #     # density of correct weights
                #     counts, edges = np.histogramdd([samples["pT_ll"]/1_000,
                #                                     samples["sumEtUE"]/1_000,
                #                                             # samples["u_perp"]/1_000,
                #                                             # samples["bias"]/1_000
                #                                             ],
                #                             weights=samples["evtWeight"], **style)
                    
                #     # density of abs weights
                #     counts_new, edges = np.histogramdd([samples["pT_ll"]/1_000,
                #                                             samples["sumEtUE"]/1_000,
                #                                             # samples["u_perp"]/1_000,
                #                                             # samples["bias"]/1_000
                #                                             ],
                #                             weights=samples["abs_evtWeight_not_correct"], **style)
                    
                #     # density between the abs and correct weights
                #     k = np.nan_to_num(counts_new/counts,1,1)
                #     fractions={"edges": [i.tolist() for i in edges], "k": k.tolist(),
                #                "counts": counts.tolist(), "counts_new": counts_new.tolist()}

                #     misc.save_json(fractions, f"{save_path}/{ff_str}.json")

                #     print(f"Create new fractions: {save_path}/{ff_str}.json")
                # else:
                #     fractions = misc.load_json(f"{save_path}/{ff_str}.json")
                # fractions = {key: np.array(val) for key, val in fractions.items()}
                # fractions["k"][fractions["k"]<0]=np.inf

                # # density between the abs and correct weights
                # bins = []
                # for nr_i, col in enumerate(reweight_cols):
                #     bins.append(list(
                #                 np.digitize(samples[col]/1_000, fractions["edges"][nr_i][1:-1]))
                #                 )
                #     # y_b = np.digitize(samples["sumEtUE"]/1_000, fractions["yedges"][1:-1])
                
                # # correct the abs weights
                # samples["abs_evtWeight"] = (samples["abs_evtWeight_not_correct"]
                #                             /fractions["k"][bins])
            else:
                samples["abs_evtWeight"] = samples["evtWeight"]
                

            samples["boson_type"] = ("Zmumu" if "zmumu" in path[0].casefold()
                                        else "Zee" if "zee" in path[0].casefold()
                                        else "W")
            if sig_bool:
                label = pd.DataFrame(np.ones((len(samples),1)),
                                     columns=["label"])
            else:
                label = pd.DataFrame(np.zeros((len(samples),1)),
                                     columns=["label"])

            data.append(pd.concat([samples, label],1))

        return data

    @staticmethod
    def calculate_recoil(data, type_to_select="truth"):

        uT = data.loc[:,["boson_pTx_reco", "boson_pTy_reco"]].values*1
        data.loc[:,"uT"] = np.sqrt(np.sum(uT**2, 1))
        phi = np.arctan2(uT[:, 0], uT[:, 1])
        if ("dilep_pTx" in data) or ("boson_pTy_truth" in data):

            if type_to_select=="reco":
                if ("dilep_pTx" in data):
                    dilep_cols=["dilep_pTx", "dilep_pTy"]
                else:
                    dilep_cols=["boson_pTx_reco", "boson_pTy_reco"]

            if type_to_select=="truth": # for W
                dilep_cols=["boson_pTx_truth", "boson_pTy_truth"]
            
            for rotate, name in zip([np.pi/2, np.pi], ["u_perp", "u_para"]):

                bosonperp = data.loc[:,dilep_cols].values*1
                bosonperp = rotate_vector(bosonperp, rotate)
                

                u_perpendicular = (np.sum(uT*bosonperp, 1) / np.linalg.norm(bosonperp, axis=1))

                data.loc[:,name] = u_perpendicular
            
            data.loc[:,"pT_ll"] = np.sqrt(np.sum(data.loc[:,dilep_cols]**2,1))

            data = utils.calculate_bias(data)

        data["uT_c_phi"] = np.cos(phi)
        data["uT_s_phi"] = np.sin(phi)

        return data

def get_bkg_paths(path, particle_version:str="z"):
    if path is None:
        return []
    if "13TeV" in path:
        energy_str = "13"
    else:
        energy_str = "5"
    
    bkg_files = glob(f"/home/users/a/algren/scratch/hadronic_recoil/lists/{particle_version}*{energy_str}*")

    if "z" in particle_version.lower():
        bkg_files_zmumu = np.loadtxt(bkg_files[0], dtype="str")
        bkg_files_zee = np.loadtxt(bkg_files[1], dtype="str")

        # get bkg files
        zee_bkg_path = glob(f"/home/users/a/algren/scratch/hadronic_recoil/{path}/pTZanalysis_zee*MC*")[0]
        bkg_files_zee = [f"{zee_bkg_path}/Nominal/{i}" for i in bkg_files_zee]

        zmumu_bkg_path = glob(f"/home/users/a/algren/scratch/hadronic_recoil/{path}/pTZanalysis_zmumu*MC*")[0]
        bkg_files_zmumu = [f"{zmumu_bkg_path}/Nominal/{i}" for i in bkg_files_zmumu]
        return [bkg_files_zee, bkg_files_zmumu]

    elif "w" in particle_version.lower():
        bkg_files_w = np.loadtxt(bkg_files[0], dtype="str")
        if "wm" in particle_version.lower():
            bkg_path = f"{path}/pTWanalysis_wminusmunu_MC_5TeV/"
        elif "wp" in particle_version.lower():
            bkg_path = f"{path}/pTWanalysis_wplusmunu_MC_5TeV/"
        bkg_files_w = [f"/home/users/a/algren/scratch/hadronic_recoil/{bkg_path}/Nominal/{i}"
                       for i in bkg_files_w]

    return [bkg_files_w]

if __name__ == "__main__":
    # %matplotlib widget
    import sys
    config = hydra_utils.hydra_init(str(root)+"/configs/ot/config.yaml")
    test_config = hydra_utils.hydra_init(str(root)+"/configs/ot/data/sumETUE/sum_ETUE_5TeV_sherpa.yaml")
    # config.data.update(test_config)
    # cfg = OmegaConf.merge(config.data, test_config)
    OmegaConf.update(config, 'data', test_config, merge=True)

    source = hydra.utils.instantiate(config.data.target_sample)(
        **config['data']
    )

    # target\
    if "target_path" in config.data:  # TODO compare to data or target
        target_path = config.target_path
        target = Pipeline(target_path, columns,log_squash=False )
    else:
        target = source
    

    if "gensyst" in config.data.target_sample.paths[0].lower():
        labels= ["Sherpa", "Pythia"]
    else:
        labels= ["MC", "Data"]

    plt.figure()
    style = {"bins": 100, "range": [0, 100], "histtype": "step"}   
    
    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    bins=100
    for col in ['evtWeight', 'abs_evtWeight']:
        _, bins, _ = ax1.hist(source.data[col], label=col, bins=bins,
                              histtype="step")
        _, _, _ = ax2.hist(source.data["pT_ll"],
                              weights=source.data[col], **style)
    ax1.legend()
    ax2.legend()
    ax1.set_yscale("log")

    # plot data
    mass_space=100
    mass_bins = np.arange(0, 100+mass_space,  mass_space)
    for col, lim in zip([
        "pT_ll",
        "sumEtUE",
        # "uT",
        "u_perp", "u_para", "bias",
                # "uT_s_phi",
                # "uT_c_phi",
                # "boson_pT_reco", "sumEtUE"
                # 'dilep_pTx', 'dilep_pTy',
                ],
                   [
                       [0, 100], [0, 400], [-60,60],
                    [-60,60],[-60,60]
                    #    [0, 6], [0, 6], [-6,6],
                    # [-6,6],[-6,6]
                    ]):#+config.columns_u_cal:
        for i in range(len(mass_bins)-1):
            mask_conds =  ((mass_bins[i]<=target.data["pT_ll"])
                        & (mass_bins[i+1]>target.data["pT_ll"]))
            mask_conds_source =  ((mass_bins[i]<=source.data["pT_ll"])
                        & (mass_bins[i+1]>source.data["pT_ll"]))
            # mask_conds =  np.ones_like(target.data[col])==1
            # mask_conds_source =  np.ones_like(source.data[col])==1
            # print(f"Data: {np.sum(mask_conds)}")
            # print(f"MC: {np.sum(mask_conds_source)}")

            fig, (ax_1, ax_2) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6), sharex="col"
                )
            ax_2.set_xlabel(f"{col} [GeV]")
            counts, _ = plot.plot_hist(
                target.data[col][mask_conds],
                # target.data[col][mask_conds],
                # source.data[col][mask_conds_source],
                # source.data[col][mask_conds_source],
                source.data[col][mask_conds_source],
                # source.data[col][mask_conds_source],
                ax=ax_1, style={"bins":30, "range": lim},
                # percentile_lst = [0.1,99.9],
                weights = [
                                target.data["abs_evtWeight"].values[mask_conds],
                                # target.data["abs_evtWeight"].values[mask_conds],
                                source.data["evtWeight"].values[mask_conds_source],
                                # source.data["abs_evtWeight"].values[mask_conds_source],
                                # source.data["abs_evtWeight_not_correct"].values[mask_conds_source],
                                # source.data["dnn_weight"].values[mask_conds_source],
                                    ],
                            dist_styles=[
                                {"label":labels[1], "color": "black"},
                                # {"label":f"abs: {labels[1]}", "ls": "dashed"},
                                {"label":labels[0], "color": "blue"},
                                # {"label":f"abs: {labels[0]}", "ls":"dashed"},
                                # {"label":f"incorrect: {labels[0]}", "ls":"dashed"},
                                # {"label":f"DNN: {labels[0]}", "ls":"dashed"},
                                ],
                            normalise=True,
                            legend_kwargs={"title": f"pT: [{mass_bins[i]}, {mass_bins[i+1]}]"}
                            )
            plot.plot_ratio(counts, truth_key="dist_0", ax=ax_2,
                            ylim=[0.8, 1.2],
                            # ylim=[0.98, 1.02],
                            zero_line_unc=True, normalise=True)
            
            if False:
                os.makedirs(f"/home/users/a/algren/scratch/hadronic_recoil/plots/{col}", exist_ok=True)
                misc.save_fig(fig,
                              f"/home/users/a/algren/scratch/hadronic_recoil/plots/{col}/{col}_{mass_bins[i]}_{mass_bins[i+1]}.png")
    print("MC")
    print("Zee: ", source.data[source.data.boson_type=="Zee"].evtWeight.sum())
    print("Zmumu: ", source.data[source.data.boson_type=="Zmumu"].evtWeight.sum())
    print("total: ", source.data.evtWeight.sum())
    print("Data")
    print("Zee: ", target.data[target.data.boson_type=="Zee"].evtWeight.sum())
    print("Zmumu: ", target.data[target.data.boson_type=="Zmumu"].evtWeight.sum())
    print("total: ", target.data.evtWeight.sum())
    if False: # bootstrapping
        counts=[]
        for col in [
            "pT_ll",
            "sumEtUE",
            # "uT",
            "u_perp", "u_para", 
                    # "uT_s_phi",
                    # "uT_c_phi",
                    # "boson_pT_reco", "sumEtUE"
                    ]:#+config.columns_u_cal:
            target_data = target.data
            for i in range(30):
                bootstrap_w_replacements = np.random.choice(len(target_data), len(target_data))
                data = target_data.iloc[bootstrap_w_replacements]

                # mask_conds =  ((mass_bins[i]<=target.data["pT_ll"])
                #             & (mass_bins[i+1]>target.data["pT_ll"]))
                # mask_conds_source =  ((mass_bins[i]<=source.data["pT_ll"])
                #             & (mass_bins[i+1]>source.data["pT_ll"]))
                mask_conds =  np.ones_like(data[col])==1
                mask_conds_source =  np.ones_like(source.data[col])==1
                print(f"Data: {np.sum(mask_conds)}")
                print(f"MC: {np.sum(mask_conds_source)}")
                ax_2.set_xlabel(f"{col} [GeV]")
                count, _ = plot.plot_hist(data[col][mask_conds],
                                        source.data[col][mask_conds_source],
                                ax=ax_1, style={"bins":50, "range": [0, 100]
                                                }, percentile_lst = [0,99.9],
                                weights = [data["evtWeight"].values[mask_conds],
                                        source.data["evtWeight"].values[mask_conds_source]],
                                dist_styles=[{"label":"Data"}, {"label":"MC"}],
                                # normalise=True
                                plot_bool=False
                                )
                counts.append(count["dist_0"]["counts"][-1])

        fig, (ax_1, ax_2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6), sharex="col"
            )
        count, _ = plot.plot_hist(target.data[col][mask_conds],
                                source.data[col][mask_conds_source],
                        # ax=ax_1,
                        style={"bins":100, "range": [0, 100]
                                        }, percentile_lst = [0,99.9],
                        weights = [data["evtWeight"].values[mask_conds],
                                source.data["evtWeight"].values[mask_conds_source]],
                        # dist_styles=[{"label":"Data"}, {"label":"MC"}],
                        # normalise=True
                        plot_bool=False
                        )
        count["dist_0"]["unc"] = np.std(counts, 0)
        plot.plot_hist_1d(counts_dict=count,
                        ax=ax_1,
                        # style={"bins":100, "range": [0, 100]
                        #                 }, percentile_lst = [0,99.9],
                        weights = [data["evtWeight"].values[mask_conds],
                                source.data["evtWeight"].values[mask_conds_source]],
                        dist_styles=[{"label":"Data"}, {"label":"MC"}],
                        # normalise=True
                          )

        plot.plot_ratio(count, truth_key="dist_0", ax=ax_2, ylim=[0.9, 1.1],
                        zero_line_unc=True)