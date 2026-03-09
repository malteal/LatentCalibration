"functions for uncertainty plotting"
"evaluate ot ftag calibration"
from copy import deepcopy
from tkinter import font
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from asyncore import loop
import json
import sys
import os
from typing import List
from omegaconf import ListConfig

from glob import glob
from scipy.optimize import curve_fit

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch as T
import pandas as pd

from tools.tools import misc
from tools.tools.visualization import general_plotting as plot
import tools.tools.visualization.plot_utils as plot_utils 
import tools.tools.visualization.atlas_utils as atlas_utils

from otcalib.otcalib.utils import plotutils
import src.eval_utils as eval_utils
import otcalib.otcalib.utils.transformations as trans

import logging

logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

default_pt = [20,30,40,60,85,110,140,175,250,400]

legend_kwargs = {"prop":{'size': 25}, "frameon": False, "title_fontsize":25}


# Define the Gaussian function
def gauss_function(x, a, x0, sigma):
    return a* np.exp(-(x - x0)**2 / (2 * sigma**2))

def higher_dim_selection(samples:list, eff_cut:dict):
    mask_list = []

    # run through all samples
    for sample in tqdm(samples,
                        leave=False):

        # get list of transformations
        trans_funcs = eff_cut["trans_func"]

        # name of non transformation ex: dl1rb, dl1rc
        keys = [i for i in eff_cut if not np.isin(i, ["trans_func", "name"])]

        # dummy mask
        mask = np.ones(len(sample), dtype=bool)

        for nr, key in enumerate(keys):
            
            # project to trans_funcs[nr]
            projection = trans_funcs[nr](sample)
            
            if isinstance(eff_cut[key], list):
                _mask = np.ravel((projection>=eff_cut[key][0])
                                & (projection<eff_cut[key][1]))
            else:
                _mask = np.ravel(projection>=eff_cut[key])
                
            mask = mask & _mask

        mask_list.append(mask)

    return mask_list


class EstimateUncertainty:
    def __init__(self, nominal:dict,
                 path,
                 bootstraps:dict=None,
                 systematics:dict=None,
                 modelling:dict=None,
                 evaluation_sample = "Flow",
                 save:bool = False,
                 b_jet_calib:bool=True,
                 **kwargs):
        self.nominal = nominal
        self.bootstraps = bootstraps
        self.modelling = modelling
        self.systematics = systematics
        self.save=save
        self.evaluation_sample = evaluation_sample
        self.systematic_counts=None
        self.systematic_weights=None
        self.systematic_conds=None
        self.save_path=None
        self.systematic_sigma={}
        self.activate_weights = False
        self.bkg_scaler_str = kwargs.get('bkg_scaler_str', None)
        self.ATLAS_style = kwargs.get('ATLAS_style', True)

        # define names and labels
        (self.style_source,
         self.style_trans,
         self.style_target) = eval_utils.get_styles(path.lower())

        self.style_target.pop("linewidth", None)
        self.style_target["linestyle"] = "none"

        # define sample namings
        if 'rm_over_pred'==self.bkg_scaler_str:
            self.dist_name = (
                "$p^\prime_{\mathrm{sim, ds}}$" if "flow" in evaluation_sample.lower()
                else "$p_{\mathrm{sim, ds}}$")
            
            self.dist_bkg_name = (
                "$p^\prime_{\mathrm{bkg, ds}}$" if "flow" in evaluation_sample.lower()
                else "$p_{\mathrm{bkg, ds}}$")
        else:
            self.dist_name = (
                "$p^\prime_{\mathrm{sim}}$" if "flow" in evaluation_sample.lower()
                else "$p_{\mathrm{sim}}$")
            
            self.dist_bkg_name = (
                "$p^\prime_{\mathrm{bkg}}$" if "flow" in evaluation_sample.lower()
                else "$p_{\mathrm{bkg}}$")

        # bjets_label = self.dist_name
        # non_bjets_label = f"{self.dist_name}
        self.trans_name = "$\widehat{T}_{\#}$"
        
        self.calib_name = f"{self.trans_name} {self.dist_name}"

        # self.signal_name = bjets_label if b_jet_calib else non_bjets_label
        # self.bkg_name = non_bjets_label if b_jet_calib else bjets_label

        self.style_trans["label"] = self.calib_name
        self.style_source["label"] = self.dist_name
        self.style_source["zorder"] = 1
        self.style_trans["zorder"] = 2
        self.style_target["zorder"] = 3

        self.style_source_bkg = self.style_source.copy()
        self.style_source_bkg["alpha"] = 0.4
        self.style_source_bkg["label"] = 'Background'
        self.style_source_bkg["linestyle"] = "dashed"

    @staticmethod
    def dummy_func(x):
        return x
    
    @staticmethod
    def apply_transformation(data, trans_func: list):
        cols = None
        if isinstance(data, pd.DataFrame):
            cols = data.columns

        for nr, func_list in enumerate(trans_func):
            if cols is not None:
                col_name = cols[nr]
                if isinstance(func_list, (list, ListConfig)):
                    for func in func_list:
                        data.loc[:, col_name] = func(data.loc[:, col_name])
                else:
                    data.loc[:, col_name] = func_list(data.loc[:, col_name])
            else:
                if isinstance(func_list, (list, ListConfig)):
                    for func in func_list:
                        data[:, nr] = func(data[:, nr])
                else:
                    data[:, nr] = func_list(data[:, nr])

        return data
            
    
    def unpack_dicts(self, transport_names:list, integral_conds_name:str=None, func=None, conds_func=None, calib_col:str="eval_transport",
                     conds_names:str=None) -> None:
        # unpack dicts
        self.integral_conds_name =integral_conds_name
        self.conds_names =conds_names

        if func is None:
            func = [self.dummy_func]*len(transport_names)
        if conds_func is None and conds_names is not None:
            conds_func = [self.dummy_func]*len(conds_names)
            
        self.calib_col = calib_col

        # for truth
        self.truth_counts = self.apply_transformation(
            self.nominal["truth"][transport_names], func)
        self.truth_conds = self.apply_transformation(
            self.nominal["truth"][conds_names], conds_func) if conds_names is not None else None

        if "weights" in self.nominal["truth"]:
            self.truth_weights = self.nominal["truth"]["weights"].values
            self.activate_weights = True
        else:
            self.truth_weights = np.ones(len(self.truth_counts))
        
        # for source
        self.source_sig_mask = self.nominal[self.evaluation_sample][
            "sig_mask"
            ].values==1

        self.source_counts = self.apply_transformation(
            self.nominal[self.evaluation_sample][transport_names], func
            )
        self.source_conds = self.apply_transformation(
            self.nominal[self.evaluation_sample][conds_names], conds_func
            ) if conds_names is not None else None
        if self.activate_weights:
            self.source_weights = self.nominal[self.evaluation_sample]["weights"].values
        else:
            self.source_weights = np.ones(len(self.source_counts))

        # for hadronic recoil calibration
        if 'additional_u_perp' in self.nominal[self.evaluation_sample]:
            self.nominal[self.calib_col].loc[:, 'u_perp'] = self.nominal[self.calib_col].loc[:, 'u_perp']*np.sign(self.nominal[self.evaluation_sample]['additional_u_perp'])
            self.source_counts.loc[:, 'u_perp'] = self.source_counts.loc[:, 'u_perp']*np.sign(self.nominal[self.evaluation_sample]['additional_u_perp'])
            
            self.truth_counts.loc[:, 'u_perp'] = self.truth_counts.loc[:, 'u_perp']*np.sign(self.nominal["truth"]['additional_u_perp'])
            
            
        # for center prediction
        self.center_counts = self.apply_transformation(
            self.nominal[self.calib_col], func
            )
        
        self.center_conds = self.apply_transformation(self.nominal[self.evaluation_sample][conds_names], conds_func) if conds_names is not None else None
        self.center_sig_mask = np.ones(len(self.center_counts))==1
        if self.activate_weights:
            self.center_weights = self.nominal[self.evaluation_sample]["weights"].values
            self.center_weights = np.ones(len(self.center_counts))
        else:
            self.center_weights = np.ones(len(self.center_counts))

        # for bootstrap
        if self.bootstraps is not None:
            self.bootstrap_counts = np.array(
                [
                    func(self.bootstraps[key][self.evaluation_sample])
                    for key in self.bootstraps if isinstance(key, int)
                ]
            )
            self.bootstrap_conds = np.ravel(self.apply_transformation(self.bootstraps[conds_names], conds_func)) if conds_names is not None else None
            self.bootstrap_weights = self.bootstraps["weights"]
            self.bootstrap_sig_mask = self.bootstraps["sig_mask"]

        #for systematics
        if self.systematics is not None: 
            if len(self.systematics)==0:
                self.systematics=None
            else:
                self.systematic_counts = {key:
                        self.apply_transformation(self.systematics[key][self.evaluation_sample][self.calib_col], func)
                        for key in self.systematics}
                self.systematic_conds = {key:
                        self.apply_transformation(self.systematics[key][self.evaluation_sample][conds_names], conds_func)
                        for key in self.systematics
                        }
                self.systematic_sig_mask = {key:
                        self.systematics[key][self.evaluation_sample]["sig_mask"]
                        for key in self.systematics
                        }
                self.systematic_weights = {key:
                        self.systematics[key][self.evaluation_sample]["weights"]
                        for key in self.systematics
                        }
        
        # modelling
        self.modelling_keys=[]
        if self.modelling is not None:
            if self.systematic_counts is None:
                self.systematic_counts={}
                self.systematic_conds={}
                self.systematic_sig_mask={}
                self.systematic_weights={}
            for key in self.modelling:
                self.systematic_counts[key] = self.apply_transformation(self.modelling[key][self.calib_col], func)
                self.systematic_conds[key] = self.apply_transformation(self.modelling[key][self.evaluation_sample][conds_names],
                                                                       conds_func)
                self.systematic_sig_mask[key] = self.modelling[key][self.evaluation_sample]["sig_mask"]==1
                self.systematic_weights[key] = self.modelling[key][self.evaluation_sample]["weights"].values
            
            self.modelling_keys = list(self.modelling.keys())

        if self.systematic_counts is not None:
            self.sys_keys = np.array(list(self.systematic_counts.keys()))

        self.calib_col_names = list(self.center_counts.columns)

    def eff_conds(self, conds_bins, eff_cut=None, plot_kwargs={}, only_sig=True,
                  calculate_sf:bool=False, plot_individual_points=False,
                  fig_name=None):
        if self.center_counts.shape[1] > 1:
            raise ValueError("The dimension of the eff distribution should be 1")

        if isinstance(eff_cut, dict) and plot_individual_points:
            raise NotImplementedError("Not implemented for higher dimensions")
        center_counts =(self.center_counts if not only_sig
                        else self.center_counts[self.center_sig_mask])
        base_conds = [
            self.center_conds if not only_sig else self.center_conds[self.center_sig_mask],
            self.truth_conds,
            self.source_conds if not only_sig
            else self.source_conds[self.source_sig_mask]]
        base_dist = [
            center_counts,
            self.truth_counts,
            self.source_counts if not only_sig
            else self.source_counts[self.source_sig_mask]]
        base_weight = [
            self.center_weights if not only_sig else self.center_weights[self.center_sig_mask],
            self.truth_weights,
            self.source_weights if not only_sig
            else self.source_weights[self.source_sig_mask]]

        styles = deepcopy([self.style_trans, self.style_target, self.style_source])
        base_labels = [
            self.center_sig_mask if not only_sig else self.center_sig_mask[self.center_sig_mask],
            np.ones(len(base_dist[1]), dtype=bool), 
            self.source_sig_mask if not only_sig
            else self.source_sig_mask[self.source_sig_mask]
            ]

            
        uncertainty_eff=np.zeros(len(conds_bins)-1)
        # calculate stat uncertainty from bootstraps
        if (self.bootstraps is not None) and (not calculate_sf):
            all_eff_dict = {}
            transport_conds=self.bootstrap_conds
            for i in tqdm(range(self.bootstrap_counts.shape[0])):
                boot_st = self.bootstrap_counts[i]
                mask_from_eff_cut = None
                sig_labels = self.bootstrap_sig_mask
                weights = self.bootstrap_weights
                
                if isinstance(eff_cut, dict):
                    mask_from_eff_cut = higher_dim_selection([self.bootstraps[0][self.evaluation_sample]],eff_cut)

                if only_sig:
                    transport_conds=self.bootstrap_conds[sig_labels]

                    boot_st = self.bootstrap_counts[i][sig_labels]
                    
                    weights = weights[sig_labels]
                    
                    if mask_from_eff_cut is not None:
                        mask_from_eff_cut = [i[sig_labels] for i in mask_from_eff_cut]

                    sig_labels = sig_labels[sig_labels]

                all_eff_dict[i] = plotutils.efficiency_plots(
                    boot_st,
                    eff_cut=eff_cut,
                    conds_bins_lst=conds_bins,
                    conditions=[transport_conds],
                    labels=["bootstrap"],
                    weights=[weights],
                    sig_labels=[sig_labels],
                    calculate_sf=calculate_sf,
                    mask_from_eff_cut=mask_from_eff_cut,
                )

            # take the std of them
            uncertainty_eff = np.std([val["bootstrap"]
                                    for _, val in all_eff_dict.items()], 0)

        # calculate systematics uncertaintiess
        # if len(self.systematic_counts)>0:
        if (self.systematic_counts is not None) and (not calculate_sf):
            all_eff_sys = {}
            nr=0
            nr_ls=0
            ls_lst = [['dotted', 'dashed'], ['solid', 'dashed'], ['dotted', 'solid']]
            for dist, conds, kwargs, key, mask_from_eff_cut in self.loop_systematics(only_sig=only_sig, eff_cut=eff_cut):
                labels=["up", "down"] if ("down" in key) else ["Source",key]
                if plot_individual_points:
                    if ("down" in key):
                        style = [{"label": key.replace("down", "up")},
                                {"label": key}]
                    else:
                        style = [{"label": "Source"}, {"label": key}]
                    style[0].update({'color':plotutils.COLORS[3+nr],
                                    'linestyle': ls_lst[nr_ls][0]})
                    style[1].update({'color':plotutils.COLORS[3+nr],
                                    'linestyle': ls_lst[nr_ls][1]})
                    if nr==len(plotutils.COLORS)-4:
                        nr_ls+=1
                        nr=0
                    base_dist.extend(dist)
                    base_conds.extend(conds)
                    base_weight.extend(kwargs['weights'])
                    base_labels.extend(kwargs["mask_sig"])
                    styles.extend(style)
                else:
                    eff_sys = plotutils.efficiency_plots(
                        *dist,
                        eff_cut = eff_cut,
                        conds_bins_lst=conds_bins,
                        conditions=conds,
                        labels=labels,
                        weights=kwargs["weights"],
                        sig_labels=kwargs["mask_sig"],
                        calculate_sf=calculate_sf,
                        mask_from_eff_cut=mask_from_eff_cut,
                        plot_bool=plot_individual_points,
                        styles=[{'label': key} for key in labels]
                        )
                    if np.isin(key, self.modelling_keys):
                        all_eff_sys[key] = ((eff_sys["Nominal"]-eff_sys[key]))**2
                    else:
                        all_eff_sys[key] = ((eff_sys["up"]-eff_sys["down"])/2)**2
                nr+=1
                        
            total_sigma = np.sqrt(uncertainty_eff**2
                                +np.sum([i for _,i in all_eff_sys.items()],0)
                                )
            target_total_sigma=None
            
        else:
            target_total_sigma=None
            total_sigma=None
            
        labels=deepcopy([i['label'] for i in styles])
        indx_to_remove = [nr for nr, i in enumerate(labels) if 'nominal' in i.lower()] # remove nominal from labels
        mask = None
        if isinstance(eff_cut, dict):
            samples = [
                self.nominal[self.evaluation_sample][self.calib_col],
                self.nominal["truth"]["transport"], 
                self.nominal[self.evaluation_sample]["transport"]]

            if only_sig:
                samples = [samples[nr][i]
                            for nr, i in enumerate(
                                [
                                    self.center_sig_mask,
                                    np.ones(len(self.truth_counts), dtype=bool),
                                    self.source_sig_mask
                                    ]
                                )]
                
            mask = higher_dim_selection(samples,eff_cut)

        total_sigma_lst=[total_sigma, target_total_sigma, None]+[None]*(len(labels)-3)

        if only_sig:
            indx_to_remove.append(1)
            fig_name = "eff_figure_" if fig_name is None else fig_name
        elif calculate_sf:
            for i in range(len(labels)):
                labels[i] += f" - {self.dist_bkg_name}"
                styles[i]['label'] += f" - {self.dist_bkg_name}"
            fig_name = "our_sample_sf_"if fig_name is None else fig_name
        else:
            fig_name = "passing_b_jet_cut_figure_"if fig_name is None else fig_name
        
        # remove indx from all lists in reverse order to avoid index out of range error
        for indx in sorted(indx_to_remove, reverse=True):
            base_conds.pop(indx)
            base_dist.pop(indx)
            base_weight.pop(indx)
            labels.pop(indx)
            total_sigma_lst.pop(indx)
            styles.pop(indx)
            if isinstance(eff_cut, dict):
                mask.pop(indx)
            
        plot_kwargs["styles"] = styles
        eff_dict,_ = self.plot_eff(
            base_dist, base_conds, eff_cut=eff_cut, labels=labels,
            uncertainty_eff=uncertainty_eff,total_sigma=total_sigma_lst,
            plot_kwargs=plot_kwargs, fig_name=fig_name, calculate_sf=calculate_sf,
            conds_bins=conds_bins, base_weight=base_weight, only_sig=only_sig, mask=mask,
            sig_labels=base_labels)
        eff_dict["stat_unc"] = uncertainty_eff
        eff_dict["full_unc"] = total_sigma
        eff_dict["bins"] = eff_dict["bins"][0]
        return eff_dict




    def plot_eff(self, base_dist:list, conds_lst:list, base_weight:list, eff_cut:int,
                 labels: List[str], uncertainty_eff:list, total_sigma:list, plot_kwargs,
                 conds_bins:list, fig_name=None, calculate_sf=False,
                 only_sig:bool=False, mask=None,
                 sig_labels:list=None):

        if (total_sigma is not None) and (len(base_dist) != len(total_sigma)):
            raise ValueError("The length of the base_dist and total_sigma should be the same")
        
        if isinstance(eff_cut, dict):
            save_name = f"{self.save_path}/{fig_name}_{eff_cut['name']}.pdf"
        else:
            save_name = f"{self.save_path}/{fig_name}_{eff_cut}.pdf"
        # print("Save name: ", save_name)
        
        eff_dict, fig = plotutils.efficiency_plots(
                                            *base_dist,
                                            weights = base_weight,
                                            conditions=conds_lst,
                                            eff_cut = eff_cut,
                                            labels=labels,
                                            plot_bool=True,
                                            transport_unc=uncertainty_eff,
                                            total_unc=total_sigma,
                                            conds_bins_lst=conds_bins,
                                            calculate_sf=calculate_sf,
                                            sig_labels=sig_labels,
                                            truth_key=labels[1] if only_sig else labels[0],
                                            mask_from_eff_cut=mask,
                                            **plot_kwargs
                                            )
        if (self.save) & (fig_name is not None):
            misc.save_fig(fig, save_name)
        
        return eff_dict, fig

    def calculate_stat_in_hist(self, xlabels, conds_bins=None, plot_individual_points=False,
                               save_path=None, **kwargs):

        if conds_bins is None:
            conds_bins = [-np.inf, np.inf]

        xlabels = kwargs.get("xlabels", xlabels)
        variables_names = kwargs.get("variables_names", self.truth_counts.keys())
        
        for low, high in tqdm(zip(conds_bins[:-1], conds_bins[1:]),
                              desc="Conds bins loop",
                              total=len(conds_bins)-1):

            for i, (name, xlabel) in enumerate(zip(variables_names, xlabels)):
                style = deepcopy(kwargs.get("hist_kw", 
                                            [{"style": {"bins": 50}}]*len(xlabels)))[i]
                #conds bins
                if self.integral_conds_name is None:
                    mask_center = np.ones(len(self.center_counts), dtype=bool)
                    mask_truth = np.ones(len(self.truth_counts), dtype=bool)
                    mask_source = np.ones(len(self.source_counts), dtype=bool)
                else:
                    mask_center = np.ravel(
                        (self.center_conds[self.integral_conds_name]>=low)
                        & (self.center_conds[self.integral_conds_name]<high))

                    mask_truth = np.ravel(
                        (self.truth_conds[self.integral_conds_name]>=low)
                        & (self.truth_conds[self.integral_conds_name]<high))
                    mask_source = np.ravel(
                        (self.source_conds[self.integral_conds_name]>=low)
                        & (self.source_conds[self.integral_conds_name]<high))
                
                ##### testing bins on data #####
                self.target_hist,_ = plot.plot_hist(
                    self.truth_counts.loc[mask_truth, name], plot_bool=False,
                    weights=[self.truth_weights[mask_truth]],
                    **deepcopy(style)
                )
                
                bins = self.target_hist["bins"]
                counts = self.target_hist["dist_0"]["counts"][0]

                # merge bins with low statistics in data
                if kwargs.get("merge_bins_threshold") is not None:
                    style["style"]["bins"] = plot_utils.merge_bins(bins, counts,
                                                                   kwargs.get("merge_bins_threshold"))[0]
                
                # nominal
                mean = self.center_counts.loc[mask_center, name]
                self.mean_counts_hist,_ = plot.plot_hist(
                    mean,
                    mask_sig=[self.center_sig_mask[mask_center]],
                    weights=[self.center_weights[mask_center]],
                    plot_bool=False,
                    **deepcopy(style)
                )
                
                #bootstraps
                if self.bootstraps is not None:
                    bootstrap_counts_cut = np.copy(self.bootstrap_counts)
                    
                    mask_bootstrap = np.ravel((self.bootstrap_conds>=low)
                                              & (self.bootstrap_conds<high))

                    self.bootstrap_hist,_ = plot.plot_hist(
                        *bootstrap_counts_cut[:, mask_bootstrap, i],
                        mask_sig=[self.bootstraps["sig_mask"][mask_bootstrap]] *len(bootstrap_counts_cut),
                        weights=[self.bootstrap_weights[mask_bootstrap]] *len(bootstrap_counts_cut),
                        plot_bool=False,
                        style={"bins": self.mean_counts_hist["bins"]},
                    )
                self.target_hist,_ = plot.plot_hist(
                    self.truth_counts.loc[mask_truth, name], plot_bool=False,
                    style={"bins": self.mean_counts_hist["bins"]},
                    weights=[self.truth_weights[mask_truth]],
                )
                self.source_hist,_ = plot.plot_hist(
                    self.source_counts.loc[mask_source, name],
                    plot_bool=False,
                    mask_sig=[self.source_sig_mask[mask_source]],
                    weights=[self.source_weights[mask_source]],
                    style={"bins": self.mean_counts_hist["bins"]},
                )
                # run systematics
                if self.systematic_counts is not None:
                    self.systematic_sigma={}
                    # sys_keys = np.array(list(self.systematic_counts.keys()))
                    for dist,_, _kwargs, key, _ in self.loop_systematics(name, low, high):
                        sys_hist,_ = plot.plot_hist(
                            *dist,
                            plot_bool=False,
                            style={"bins": self.mean_counts_hist["bins"]},
                            normalise=True,
                            **_kwargs
                        )

                        # if "dist_1" in list(sys_hist.keys()):
                        if np.isin(key, self.modelling_keys) & plot_individual_points:
                            self.systematic_sigma[key] = sys_hist["dist_1"]["counts"][-1]
                        elif plot_individual_points:
                            self.systematic_sigma[key] = {
                                "up": sys_hist["dist_0"]["counts"][-1]
                                    /sys_hist["dist_0"]["counts"][-1].sum(),
                                "down": sys_hist["dist_1"]["counts"][-1]
                                    /sys_hist["dist_1"]["counts"][-1].sum(),
                                                        }
                        else:
                            if np.isin(key, self.modelling_keys):
                                #dist_0 nominal
                                #dist_1 modelling
                                self.systematic_sigma[key.split("_down")[0]] = (
                                    np.abs(sys_hist["dist_0"]["counts"][-1]
                                        -sys_hist["dist_1"]["counts"][-1]))
                            else:
                                #dist_0 up
                                #dist_1 down
                                self.systematic_sigma[key.split("_down")[0]] = (
                                    np.abs(sys_hist["dist_0"]["counts"][-1]
                                        -sys_hist["dist_1"]["counts"][-1])/2)

                if save_path is not None:
                    integral_conds_name = '' if self.integral_conds_name is None else self.integral_conds_name
                    if self.conds_names is not None:
                        name_str = f'{name}_hist_{low}{high}'
                    else:
                        name_str = f'{name}'
                    save_path_name = f"{save_path}/{integral_conds_name}/{name_str}.pdf"
                else:
                    save_path_name=None
                    
                print(f"Plotting histogram: {[low, high]}")
                self.plot_histogram(pt_lst=[low, high],
                                    plot_individual_points=plot_individual_points,
                                    save_path = save_path_name,
                                    xlabel=xlabel,
                                    **kwargs)
                
    def loop_systematics(self, col=None, low=None, high=None, only_sig=False, eff_cut:dict=None):
        for ((key, counts),
                (_, conds)) in tqdm(zip(self.systematic_counts.items(),
                                    self.systematic_conds.items()),
                                    total=len(self.systematic_counts),
                                    leave=False, desc="Systematics loop: ",):
            # if ((("up" in key) and (not ("pileup" in key.lower())))
            if (("down" in key) or ("1down" in key)):
                key_down = self.sys_keys[np.array([
                    key.replace("down", "up") in i for i in self.sys_keys
                    ])]
                if len(key_down)==0:
                    key_down = self.sys_keys[np.array([
                        key.replace("1down", "1up") in i for i in self.sys_keys
                        ])][0]
                else:
                    key_down = key_down[0]
                if key_down != key.replace("down", "up"):
                    raise("Incorrect key found!")
                down_counts = self.systematic_counts[key_down]
                down_conds = self.systematic_conds[key_down]
                if low is not None:
                    mask_sys = np.ravel((conds>=low) & (conds<high))
                    mask_sys_down = np.ravel((down_conds>=low) & (down_conds<high))
                else:
                    mask_sys = np.ones((len(conds),))==1
                    mask_sys_down = np.ones((len(down_conds),))==1
                
                if col is not None:
                    dist = (counts[mask_sys, col], down_counts[mask_sys_down, col])
                else:
                    dist = (counts[mask_sys], down_counts[mask_sys_down])
                conds = (conds[mask_sys], down_conds[mask_sys_down])

                mask_sig=[self.systematic_sig_mask[key][mask_sys],
                            self.systematic_sig_mask[key_down][mask_sys_down]]
                weights = [self.systematic_weights[key][mask_sys],
                           self.systematic_weights[key_down][mask_sys_down]]
                
                if all(weights[0]!=weights[1]):
                    raise ValueError("Weights are not the same")

                mask_lst=None
                if isinstance(eff_cut, dict):
                    mask_lst = higher_dim_selection(
                        [
                            self.systematics[key][self.evaluation_sample][self.calib_col],
                            self.systematics[key_down][self.evaluation_sample][self.calib_col]
                         ],
                        eff_cut=eff_cut)
                
                if only_sig:
                    if mask_lst is not None:
                        mask_lst = [i[j] for i,j in zip(mask_lst, mask_sig)]

                    dist = [i[j] for i,j in zip(dist, mask_sig)]
                    conds = [i[j] for i,j in zip(conds, mask_sig)]
                    weights = [i[j] for i,j in zip(weights, mask_sig)]
                    mask_sig = [i[j] for i,j in zip(mask_sig, mask_sig)]


                print(f"Systematics: {key} & {key_down}")

        
                yield dist, conds, {"mask_sig":mask_sig, "weights":weights}, key, mask_lst
            elif "up" in key:
                continue
            else:
                # nominal_dist = self.source_counts
                # nominal_conds =  self.source_conds
                # if only_sig:
                #     base_dist = [self.source_counts[self.source_sig_mask]]
                #     base_conds = [self.source_conds[self.source_sig_mask]]
                if low is not None:
                    mask_sys = np.ravel((conds>=low) & (conds<high))
                    mask_sys_nominal = np.ravel((self.center_conds>=low) &
                                        (self.center_conds<high))
                else:
                    mask_sys = np.ones((len(conds), ))==1
                    mask_sys_nominal = np.ones((len(self.center_conds), ))==1
                
                if col is not None:
                    dist = counts.loc[mask_sys, col].values
                    nominal_dist = self.center_counts.loc[mask_sys_nominal, col].values
                else:
                    dist = counts[mask_sys]
                    nominal_dist = self.center_counts[mask_sys_nominal]

                conds = conds[mask_sys].values
                nominal_conds =  self.center_conds[mask_sys_nominal].values
                
                mask_sig = self.systematic_sig_mask[key][mask_sys]
                mask_sig_nominal = self.center_sig_mask[mask_sys_nominal]

                weigths = self.systematic_weights[key][mask_sys]
                nominal_weigths = self.center_weights[mask_sys_nominal]

                mask_lst=None
                if isinstance(eff_cut, dict):
                    mask_lst = higher_dim_selection(
                        [self.nominal[self.evaluation_sample][self.calib_col],
                         self.modelling[key][self.evaluation_sample][self.calib_col]],
                        eff_cut=eff_cut)

                if only_sig:
                    if mask_lst is not None:
                        mask_lst = [i[mask] for i, mask in zip(mask_lst,
                                                             [mask_sig_nominal,
                                                              mask_sig])]

                    dist = dist[mask_sig]
                    conds = conds[mask_sig]
                    weigths = weigths[mask_sig]
                    mask_sig = mask_sig[mask_sig]
                    
                    #nominal
                    nominal_dist = nominal_dist[mask_sig_nominal]
                    nominal_conds = nominal_conds[mask_sig_nominal]
                    nominal_weigths = nominal_weigths[mask_sig_nominal]
                    mask_sig_nominal = mask_sig_nominal[mask_sig_nominal]
                    

                print(f"Modelling: {key}")
                

                kwargs = {"mask_sig":[mask_sig_nominal, mask_sig],
                          "weights":[nominal_weigths, weigths]}

                yield ([nominal_dist, dist], [nominal_conds, conds],
                       kwargs, key, mask_lst)


    def plot_histogram(self, pt_lst, plot_individual_points=False, **kwargs):
        self.style_trans.pop("drawstyle", None)
        self.style_target.pop("drawstyle", None)
        self.style_source.pop("drawstyle", None)
        self.style_source_bkg.pop("drawstyle", None)


        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(8, 8),
            sharex="col",
        )
        
        #plot target
        centers = (self.mean_counts_hist["bins"][1:] + self.mean_counts_hist["bins"][:-1]) / 2
        xerr = (self.mean_counts_hist["bins"][1:] - self.mean_counts_hist["bins"][:-1]) / 2
        
        # for target/data
        target_error = plot.binary_uncertainty(counts=self.target_hist["dist_0"]["counts"][-1])
        p_target= self.target_hist["dist_0"]["counts"][-1]/self.target_hist["dist_0"]["counts"][-1].sum()
        
        
        ax.errorbar(centers, p_target, yerr=target_error, xerr=xerr,
                    **self.style_target)

        # transport
        centers_transport = self.mean_counts_hist["dist_0"]["counts"][-1]

        if plot_individual_points: # just plot the up down of the uncertainties
            system_style = self.style_trans.copy()
            centers_transport = centers_transport/np.sum(centers_transport)
            system_style.pop("color", None)
            total_counts = {}
            for nr, (k, up_down) in enumerate(self.systematic_sigma.items()):
                system_style["label"] = k
                if isinstance(up_down, dict):
                    for _, i in up_down.items():
                        plot.plot_stairs(
                            i,
                            self.mean_counts_hist["bins"],
                            style=system_style,
                            ax=ax,
                        )
                        total_counts[system_style["label"]]={"counts": [i],
                                                            #  "unc": [np.sqrt(i)]
                                                                }
                        system_style["label"] = system_style["label"].replace("up", "down")
                else:
                    plot.plot_stairs(
                            up_down,
                            self.mean_counts_hist["bins"],
                            style=system_style,
                            ax=ax,
                        )
                    total_counts[system_style["label"]]={"counts": [up_down],
                                                        #  "unc": [np.sqrt(i)]
                                                            }
            plot.plot_stairs(
                centers_transport,
                self.mean_counts_hist["bins"],
                style=self.style_trans,
                ax=ax,
            )
            ## plotting ratio plot
            #     "source": {
            #         "counts": [self.source_hist["dist_0"]["counts"][1]],
            #         "unc": [np.sqrt(self.source_hist["dist_0"]["counts"][1])],
            #     },
            total_counts["transport"] = {"counts": [centers_transport],
                                        #  "unc": [stat_sigma],
                                        #  "total_unc": [total_sigma]
                                         }
            total_counts["truth"]= {"counts": [self.target_hist["dist_0"]["counts"][0]]}
            total_counts["bins"] = self.mean_counts_hist["bins"]
            plot.plot_ratio(
                total_counts,
                truth_key="truth",
                ylim=[0.8, 1.2],
                ax=ax_ratio,
                # styles=[self.style_source.copy(), self.style_trans.copy()],
                normalise=True,
                zero_line_unc=False,
            )
        else:
            if self.bootstraps is not None:
                bootstrap_sigma = np.std(
                    [
                        self.bootstrap_hist[key]["counts"][-1]/self.bootstrap_hist[key]["counts"][-1].sum()
                        for key in self.bootstrap_hist.keys()
                        if "bins" not in key
                    ],
                    0,
                )
                # stat_sigma = bootstrap_sigma*centers_transport.sum()
            else: 
                bootstrap_sigma = self.mean_counts_hist["dist_0"]["unc"][-1]/self.mean_counts_hist["dist_0"]["counts"][-1].sum()
                
            # calculate total uncertainty
            total_sigma = np.sqrt(
                bootstrap_sigma**2 
                +np.sum([i**2 for _,i in self.systematic_sigma.items()],0)
                                )*centers_transport.sum()

            # plot transport
            plot.plot_stairs(
                centers_transport,
                self.mean_counts_hist["bins"],
                sigma=bootstrap_sigma,
                total_sigma=total_sigma/centers_transport.sum(),
                style=self.style_trans,
                ax=ax,
            )
            
            # plot source distribution
            plot.plot_stairs(
                self.source_hist["dist_0"]["counts"][-1],
                self.mean_counts_hist["bins"],
                (np.sqrt(self.source_hist["dist_0"]["counts"][-1])
                    /self.source_hist["dist_0"]["counts"][-1].sum()),
                style=self.style_source,
                ax=ax,
            )
            # plot bkg sample if any
            if len(self.source_hist["dist_0"]["counts"])>1:
                plot.plot_stairs(
                    self.source_hist["dist_0"]["counts"][0] / self.source_hist["dist_0"]["counts"][-1].sum(),
                    self.mean_counts_hist["bins"],
                    style=self.style_source_bkg,
                    ax=ax,
                    normalise=False,
                )

            ## plotting ratio plot
            total_counts = {
                "source": {
                    "counts": [self.source_hist["dist_0"]["counts"][-1]],
                    "unc": [np.sqrt(self.source_hist["dist_0"]["counts"][-1])],
                },
                "transport": {"counts": [centers_transport],
                              "unc": [bootstrap_sigma*centers_transport.sum()],
                            "total_unc": [total_sigma]},
                "truth": {"counts": [self.target_hist["dist_0"]["counts"][0]],
                          "unc": [np.sqrt(self.target_hist["dist_0"]["counts"][-1])]},
                "bins": self.mean_counts_hist["bins"],
            }
            plot.plot_ratio(
                total_counts,
                truth_key="transport",
                ylim=kwargs.get('ratio_ylim', [0.8, 1.2]),
                ax=ax_ratio,
                styles=[self.style_source.copy(), self.style_trans.copy(),
                        self.style_target.copy()],
                normalise=True,
                zero_line_unc=True,
                overwrite_uncertainty=False
            )

        # histogram settings
        ax.set_ylabel(kwargs.get("ylabel", "Normalised entries"), fontsize=plot.FONTSIZE)
        # ax.set_yticks(fontsize=plot.FONTSIZE)
        ax.set_xlim([total_counts["bins"][0], total_counts["bins"][-1]])
        
        # ratio plot
        ax_ratio.set_xlabel(kwargs.get("xlabel", "DL1r"), fontsize=plot.FONTSIZE)
        ax_ratio.set_ylabel(kwargs.get("ylabel_ratio", f"{self.dist_name}/Data"))
        ax.tick_params(axis='y', labelsize=plot.LABELSIZE)
        ax_ratio.tick_params(axis='both', labelsize=plot.LABELSIZE)
    
        if kwargs.get("yscale", "log")=="log":
            ax.set_yscale("log")
        
        if self.integral_conds_name is not None:
            ATLAS_kwargs = {"eta": pt_lst} if "eta" in self.integral_conds_name else {"pt": pt_lst}

        if self.ATLAS_style:
            atlas_utils.ATLAS_setup(ax=ax, **ATLAS_kwargs)
        else:
            ax.legend(**legend_kwargs)
        
        ylim_hist = kwargs.get("ylim_hist") 
        if ylim_hist is not None:
            ax.set_ylim(ylim_hist)

        if isinstance(kwargs.get("save_path", None), str) & self.save:
            misc.save_fig(fig, kwargs['save_path'])

    def conds_closure(self, conds_percentile = None, mean_bool=True,
                      truth_key="Data", single_figure=True,
                      fit_bool=True, **kwargs):
        distributions={
            self.style_target['label']: {
                'conds': self.truth_conds.values,
                'dist': self.truth_counts.values,
                'weights': self.truth_weights,
            },
            self.style_trans['label']: {
                'conds': self.center_conds.values,
                'dist': self.center_counts.values,
                'weights': self.center_weights,
            },
            self.style_source['label']: {
                'conds': self.source_conds.values,
                'dist': self.source_counts.values,
                'weights': self.source_weights,
            }
        }

        if conds_percentile is None:
            conds_percentile = np.round(np.percentile(distributions[self.style_target['label']]["conds"],
                                                      np.arange(0, 100, 4), 0),3).T

        conds_dist_styles = [self.style_target, self.style_trans, self.style_source]

        for i in conds_dist_styles:
            i["marker"] = "."

        if single_figure:
            fig, ax_all = plot.generate_figures(1, len(self.conds_names))

        ylim_lst = kwargs.get("ylim", [[0.95, 1.05]]*len(self.calib_col_names))

        for nr_conds, (bin_conds, conds_name) in enumerate(zip(conds_percentile,
                                                             self.conds_names)):
            for nr_cal, cal_name in enumerate(self.calib_col_names):
                dists = {}
                if not single_figure:
                    fig, ax = plt.subplots(
                        2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6),sharex="col"
                        )
                else:
                    ax = [ax_all[2*nr_conds, nr_cal], 
                          ax_all[2*nr_conds+1, nr_cal]]

                for key, values in distributions.items():

                    if key not in dists:
                        dists[key] = {"counts": [[]], "unc": [[]]}

                    for low, high in zip(bin_conds[:-1], bin_conds[1:]):
                        mask_conds =  ((low<=values["conds"][:,nr_conds])
                                    & (high>values["conds"][:,nr_conds]))

                        values_in_conds = values["dist"][mask_conds][:,nr_cal]
                        if "weights" in values:
                            weights_in_conds = values["weights"][mask_conds]
                        else:
                            weights_in_conds = np.ones(len(values_in_conds))

                        if kwargs.get("lim"):
                            lim = kwargs.get("lim")
                            values_in_conds=values_in_conds[(values_in_conds>lim[0])
                                                            & (values_in_conds<lim[1])]
                        if fit_bool:

                            lim = [
                                np.mean(values_in_conds)-(2*np.sqrt(np.mean(values_in_conds**2))),
                                np.mean(values_in_conds)+(2*np.sqrt(np.mean(values_in_conds**2)))
                                ]
                            fit_lim = (values_in_conds>lim[0]) & (values_in_conds<lim[1])
                            weights_in_conds=weights_in_conds[fit_lim] 
                            values_in_conds=values_in_conds[fit_lim]
                            
                            # Create a weighted histogram from the data
                            y, bin_edges = np.histogram(values_in_conds, bins=30,
                                                        density=True, weights=weights_in_conds)
                            x_hist = (bin_edges[1:] + bin_edges[:-1]) / 2  # calculate bin centers

                            # Calculate the weighted mean and standard deviation
                            mean = np.average(values_in_conds, weights=weights_in_conds)
                            sigma = np.sqrt(np.average((values_in_conds - mean)**2, weights=weights_in_conds))

                            # Use curve_fit to fit the Gaussian function to your data
                            popt, pcov = curve_fit(gauss_function, x_hist, y, p0=[1, mean, sigma])
                            _, central_mu, central_std = popt
                            
                            # mu, std = norm.fit(values_in_conds)
                        else:
                            central_mu= np.median(values_in_conds)
                            central_std = np.diff(np.percentile(values_in_conds, [25,75]))[0]/1.349
                        unc_val=0
                        if mean_bool:
                            mean_bool_str="mean"
                            center_val = central_mu

                            if len(values.get("bootstraps", []))>0:
                                unc_val+=np.std([np.median(i[mask_conds][:, nr_cal]) for i in values["bootstraps"]])

                            if len(values.get("systematics", []))>0:
                                unc_val+=np.std([np.median(i[mask_conds][:, nr_cal]) for i in values["systematics"]])

                            if unc_val==0:
                                unc_val = (np.std(values_in_conds)
                                    /np.sqrt(np.sum(mask_conds)))*(np.sqrt(np.pi/2))
                        else:
                            mean_bool_str="width"
                            center_val = central_std
                            if len(values.get("bootstraps", []))>0:
                                unc_val += np.std([np.diff(np.percentile(i[mask_conds][:, nr_cal], [25,75]))[0]/1.349 for i in values["bootstraps"]])

                            if len(values.get("systematics", []))>0:
                            # if "systematics" in values:
                                unc_val += np.std([np.diff(np.percentile(i[mask_conds][:, nr_cal], [25,75]))[0]/1.349 for i in values["systematics"]])

                            if unc_val==0:
                                unc_val = 1.573*np.sqrt(center_val**2/np.sum(mask_conds))

                        dists[key]["counts"][0].append(center_val)
                        dists[key]["unc"][0].append(unc_val)

                dists["bins"] = bin_conds

                # histogram
                counts, _ = plot.plot_hist_1d(counts_dict = dists.copy(),
                                            ax=ax[0], normalise=False,
                                            dist_styles=conds_dist_styles,
                                            xerr_on_errorbar=True,
                            # style={"bins":utils.default_bins()},
                            legend_kwargs=kwargs.get("legend_kwargs", {})
                            )

                # ratio plot
                plot.plot_ratio(counts, truth_key=truth_key, ax=ax[1],
                                ylim=ylim_lst[nr_cal],
                                styles=conds_dist_styles, zero_line_unc=True, 
                                overwrite_uncertainty=False, normalise=False)
                if fit_bool and mean_bool:
                    ylabel_name = r"$_{fit mu}$ [GeV]"
                elif fit_bool and not mean_bool:
                    ylabel_name = r"$_{fit sigma}$ [GeV]"
                elif not fit_bool and mean_bool:
                    ylabel_name = r"$_{median}$ [GeV]"
                else:
                    ylabel_name = r"$_{IQR}$ [GeV]"

                ax[0].set_ylabel(cal_name+ylabel_name)
                ax[1].set_xlabel(rf"{conds_name} [GeV]")
                if (self.save_path is not None) & (not single_figure):
                    misc.save_fig(fig, f"{self.save_path}/{cal_name}_vs_{conds_name}_{mean_bool_str}.pdf")
        if (self.save_path is not None) & (single_figure):
            misc.save_fig(fig, f"{self.save_path}/closure_in_conditions_{mean_bool_str}_{bin_conds}.pdf")

            
# evaluate performance of calibration

def load_calibration(path, new_predictions=False, **kwargs):
    if kwargs.get("parent_folder", False):
        file_name = glob(f"{kwargs['parent_folder']}/*ps.npy")
        bootstrap_path=f"{kwargs['parent_folder']}/bootstraps.npy"
    else:
        file_name = glob(f"{path}/*.npy")
        bootstrap_path=None
    if (len(file_name) == 0) or new_predictions:
        if not isinstance(path, list):
            path = [path]
        calibration = eval_utils.run_calibration(path,
                                                 bootstrap_path=bootstrap_path,
                                                 best_index=-1,
                                                 bkg_weighting=False,
                                                 **kwargs)
    else:
        try:
            calibration = np.load(file_name[-1], allow_pickle=True).item().copy()
        except:
            calibration = np.load(file_name[-1], allow_pickle=True)["arr_0"].item().copy()
    key=list(calibration.keys())
    if (len(key)>1) and (bootstrap_path is None):
        raise Warning(f"More than one key: {key}")
    elif bootstrap_path is not None:
        return calibration
    else:
        return calibration[key[0]]

def get_systematics_calibration(sys_paths, data_size, new_predictions=False,
                                generator_path=None):
    systematics={}
    for path in tqdm(sys_paths):
        try:
            with open(f"{path}/commandline_args.txt") as f:
                commandline_args = f.read()
            commandline_args = json.loads(commandline_args)
            name = commandline_args["generator_path"].split("-")[-1].split("_")[1:]
            name = "_".join(name)
        except:
            commandline_args = misc.load_yaml(f"{path}/.hydra/config.yaml")
            # name = commandline_args["data_cfg"]["generator_path"].replace("*", "_").split("-")[-1].split("_")[1:]
            name = commandline_args["data_cfg"]["generator_path"].replace("*", "").split("/")[-1]
        systematics[name]=load_calibration(path, data_size=data_size,
                                           new_predictions=new_predictions,
                                           generator_path=generator_path,
                                           use_eval_sample=True # TODO maybe not hardcoded
                                           )
    return systematics

def init_dl1r(dl1r_c:bool, logit=True, fc=None):
    def dl1r(x):

        if logit:
            x = trans.probsfromlogits(x)

        x = trans.dl1r(x, dl1r_c=dl1r_c, fc=fc) # page 23
        return x.reshape(-1,1)
    return dl1r

        
if __name__ == "__main__":
    sys.path.insert(0, "/home/users/a/algren/.local/lib/python3.7/site-packages")
    sys.path.insert(0, "/home/users/a/algren/work/root_stuff")
    from test_h5_files import load_h5_file, get_systematics, pack_data

    # %matplotlib widget
    data_size = 1_000_000
    PATH = "/home/users/a/algren/scratch/trained_networks/ftag_calib/full_run_large_model/"
    SPATH = "/home/users/a/algren/scratch/trained_networks/ftag_calib/paper_runs/"
    generator_path = "/home/users/a/algren/scratch/trained_networks/ftag_calib/sample_template/2023-03-11_16-54-55-933067_testing_mc_data"
    device = "cpu"
    evaluation_sample = "MC" # Flow MC
    
    save=False
    plot_unc_points=False
    only_sig=False # cannot be same as calculate_sf
    calculate_sf=True
 
    if False:
        global_path = "/home/users/a/algren/scratch/ftag-otcalib/"
        truth_label=[5]
        # %matplotlib widget
        path = f"/{global_path}/root_files/ttbar_sel/merged/*.h5"

        paths = glob(path)
        test_distribution = [paths,
                            f"/{global_path}/root_files/ttbar_sel/FTAG2_physics_Main/FTAG2_physics_Main_nominal.h5"
                            ]
        systematics_path = "/home/users/a/algren/scratch/ftag-otcalib/systematics"
        uncertainties = [
            "EtaIntercalibration", 
            "Flavor_Composition", "Flavor_Response",
            "JER_DataVsMC_MC16", "JER_EffectiveNP_1", "Pileup_OffsetMu",
            "Pileup_RhoTopology"
                            ]
        systematics_paths = {i:glob(f"{systematics_path}/{i}*/merged/") for i in uncertainties}
        systematics_paths["hdamp_modelling"] = ["/home/users/a/algren/scratch/ftag-otcalib/root_files/user_alopezso_410482_PhPy8EG_/merged/"]
        systematics_paths["herwig_modelling"] = ["/home/users/a/algren/scratch/ftag-otcalib/MC_to_MC_only_ttbar/merged/"]
        columns = ['jet_pt', "jet_DL1r_pb", "jet_DL1r_pc", "jet_DL1r_pu", "jet_truthflav"]

        real_data = load_h5_file(test_distribution[1],
                                sample_n_events=data_size,
                                cols=columns[:-1]).values
        real_data =np.c_[real_data, np.ones(len(real_data))]

        data = load_h5_file(test_distribution[0],
                            sample_n_events=data_size,
                            weight_name="abs_event_xsec",
                            cols=columns
                            ).values
        data[:,-1] = np.isin(data[:, -1], truth_label)*1
        
        #get systematics
        bootstraps=None
        systematics = get_systematics(paths=systematics_paths,columns=columns,
                                      truth_label=truth_label,
                                      pack_samples=True, data_size=data_size,
                                      weight_name="abs_event_xsec")
        
        nominal = pack_data(data,real_data,
                        names=["MC", "truth"], cvx_dims=3,
                        noncvx_dims=1)

        conds_func = lambda x: x
        calib_col=None
        ratio_ylim = [0.75,1.15]
        

    else:
        paths={
            "nominal": '/home/users/a/algren/scratch/trained_networks/ftag_calib/gridsearch/OT_gridsearch_PICNN_05_29_2023_19_57_19_966542_MC_data/05_29_2023_19_59_21_249825_cern_prob_3_1_MC_to_data/',

            # #uncertainties
            # "bootstraps": glob(f"{PATH}/bootstraps/*/"),

            # "systematic": glob(f"{PATH}/systematics/2023_08_*"),

            # "modelling": glob(f"{PATH}/modelling/*hdamp*")+glob(f"{PATH}/modelling/*2023_08_29_11_39_16_213833_lights_nom*")+glob(f"{PATH}/modelling/*08_24*herwig_v_pythia*")

            ##### ############# old
            # "systematic": glob(f"{PATH}/systematics/2023_08_27*")+glob(f"{PATH}/systematics/2023_08_28*"),
            # old
            # "modelling": glob(f"{PATH}/modelling/*_08_31_19_06_24_916368_herwig_v_pythia*"),
            # "modelling": glob(f"{PATH}/modelling/*09_07_14_48_22_662452_herwig_v_pythia*"),
            # "modelling": [
            #     f"{PATH}/modelling/herwig_v_pythia_2023_06_25_16_14_21_031136/",
            #     f"{PATH}/modelling/2023_06_22_18_34_14_857158_hdamp/",
            #     f"{PATH}/modelling/2023_07_20_16_24_45_150417_lights_nominal/",
            #                 ]
            }
        if calculate_sf:
            paths.pop("bootstraps", None)

        new_predictions=False
        # sys.exit()
        systematics={}
        if "systematic" in paths:
            print(" Systematic ".center(20, "~"))
            systematics = get_systematics_calibration(paths["systematic"],
                                                    new_predictions=new_predictions,
                                                    data_size=data_size,
                                                    generator_path=generator_path)
        # sys.exit()

        if "modelling" in paths:
            print(" Modelling ".center(20, "~"))
            modelling = get_systematics_calibration(paths["modelling"],
                                                    new_predictions=False,
                                                    data_size=data_size,
                                                    generator_path=generator_path)
            systematics.update(modelling)
        bootstraps=None
        if "bootstraps" in paths:
            print(" Bootstraps ".center(20, "~"))
            bootstraps = load_calibration(paths["bootstraps"],
                                        parent_folder=f"{PATH}/bootstraps/",
                                        data_size=data_size,
                                        new_predictions=new_predictions,
                                        device=device,
                                        load_data_every_time=False,
                                        use_eval_sample=True,
                                        )
        
        print(" Nominal ".center(20, "~"))
        nominal = load_calibration(paths["nominal"], data_size=data_size,
                                new_predictions=new_predictions,
                                    use_eval_sample=True,
                                )
        conds_func = np.exp
        calib_col="eval_transport"
        ratio_ylim = [0.85,1.05]


    save_folder = f"{SPATH}/plots/"
    os.makedirs(save_folder, exist_ok=True)
    print("Running class!")
    # {i:np.abs(systematics[i]["Flow"]["eval_transport"]-systematics[i]["Flow"]["transport"]).mean() for i in systematics}
    cls_unce = EstimateUncertainty(nominal=nominal,
                                   bootstraps=bootstraps,
                                   systematics=systematics,
                                   path="mc_to_data",
                                   evaluation_sample=evaluation_sample,
                                   save=save)

    # sys.exit()
    for (disc_name, save_path, xlabels,
            eff_linspace, transform_func, ylabel
            ) in figure_lst_generator(save_path=save_folder,
                                      only_sig=only_sig,calculate_sf=calculate_sf,
                                      save=save, logit=True):

        cls_unce.save_path = save_path
        

        cls_unce.unpack_dicts(conds_func=conds_func, func=transform_func,
                              calib_col="eval_transport")

        # conds_bins=np.percentile(
        #         cls_unce.truth_conds,
        #         np.linspace(0, 100, 50)
        #         )
        # # conds_bins=np.arange(20, 400, 20)
        # conds_bins[-1]+=conds_bins[-1]/100
        # conds_bins = np.round(conds_bins,2)
        # conds_bins = np.array([ 20, 400])
        conds_bins = np.array([ 20, 30, 40, 60, 85,110,140,175,250,400])
        # conds_bins = np.round([ 250, cls_unce.truth_conds.max()], 2)
        print(conds_bins)
        # eff_linspace=[]
        if (disc_name!="") & (len(conds_bins)>2):
            for i in eff_linspace:
                cls_unce.eff_conds(eff_cut=i, conds_bins=conds_bins,
                                                plot_kwargs={"ratio_ylim": ratio_ylim,
                                                             "ylabel":ylabel,
                                                             "legend_title": rf"{xlabels[0]} {get_WP(i)}",
                                                            # "total_unc_label":"JES"
                                                            "total_unc_label":"Stat + JES + modelling"
                                                            },
                                                only_sig=only_sig, plot_individual_points=plot_unc_points,
                                                calculate_sf=calculate_sf)
            sys.exit()
        if (not only_sig) or plot_unc_points: # plot dl1r
            if plot_unc_points:
                save_name+="_individual_errors"
            
            cls_unce.calculate_stat_in_hist(conds_bins=conds_bins, bins=15,
                                            xlabels=xlabels,
                                            save_path=save_name+xlabels[0],
                                            plot_unc_points=plot_unc_points
                                            )