"evaluate OT calibration"
import pyrootutils


root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import os
import hydra
import numpy as np
import pandas as pd
import logging
import corner
from sklearn.decomposition import PCA
import matplotlib.lines as mlines


# from framework
import evaluate.uncertainties as unc 
from run.predict_ot import PredictionHandler
from src.datamodule import get_classifier_output
from src import utils

# internal
from tools.tools.hydra_utils import hydra_init
from tools.tools import misc
import tools.tools.transformations as trans
import tools.tools.visualization.general_plotting as plot
import matplotlib.pylab as pylab

log = logging.getLogger(__name__)
plot.lw=3.5
plot.FONTSIZE=35
plot.LABELSIZE = 35

unc.legend_kwargs = {"prop":{'size': 30}, "frameon": False, "title_fontsize":30,
                      "loc": 'lower center'}

if __name__ == '__main__':
    # %matplotlib widget    
    config = hydra_init(str(root/"configs/eval_ot.yaml"), verbose=True)

    # n_bins=10
    selection = config['type_to_eval']
    gen_name = config["evaluation_sample"]

    # get paths to model
    path_to_eval = config['paths']
    nominal_path = path_to_eval[selection][config['name_to_eval']]

    ######### get nominal prediction #########
    transport_handler = PredictionHandler(nominal_path, gen_name,
                                          config["bkg_scaler_str"],
                                          force_prediction = False,
                                          allow_prediction = False)

    data = transport_handler.load_prediction()
    
    if False:
        # run validation sample
        data['source'] = data.pop('source_valid')
        data['truth'] = data.pop('Data_valid')
        data['eval_transport'] = data.pop('eval_transport_valid')
    else:
        data['truth'] = data['Data']
    
    additional_str = f"_{config.bkg_scaler_str}"
    
    if config.get("run_clf_layers"):
        # run more layer of the classifier
        data = get_classifier_output(data, nominal_path, config.get("run_clf_layers"))
        
        # insert dummy weight variable in again
        for i,j in data.items():
            if i == "eval_transport":
                continue
            dummy = pd.DataFrame(np.ones((len(j),2)), columns=['weights', 'sig_mask'])
            data[i] = pd.concat([j, dummy],axis=1)

    columns = data['eval_transport'].columns

    log.info("Plotting...")
    plotting = unc.EstimateUncertainty(data,path="",
                                evaluation_sample=gen_name.split("_")[0],
                                save=config.save_figures,
                                bkg_scaler_str = config.bkg_scaler_str,
                                )
    
    ## change names of distributions
    plotting.style_target['label'] = "Target"
    plotting.style_source['label'] = "Source"
    plotting.style_trans['label'] = "Calibrated"
    save_path = f"{nominal_path}/plots/{gen_name}{additional_str}/"

    os.makedirs(save_path,exist_ok=True)

    log.info("Plotting histograms")
    config.list_to_run_over['latn']['transport_names'] = list(columns)
    config.list_to_run_over[f'latn_{len(columns)}'] = config.list_to_run_over.pop('latn')
    for disc_name, items in config.list_to_run_over.items():
        # get the save path
        _save_path=None
        if config.save_figures:
            _save_path = f"{save_path}/{disc_name}"
            os.makedirs(f'{_save_path}/',exist_ok=True)

            if config.get('conds_name') is not None:
                os.makedirs(f'{_save_path}/{config.conds_name}/',exist_ok=True)
        kwargs = {}
        # plotting.save_path =_save_path
        
        # setup transformations
        if 'transform_func' in items:
            kwargs['transform_func'] = hydra.utils.instantiate(items['transform_func'])
        
        if 'conds_func' in items:
            kwargs['conds_func'] = hydra.utils.instantiate(items['conds_func'])
        
    
        plotting.unpack_dicts(transport_names = items['transport_names'],
                                # conds_names = items.get("conds_names"),
                                # integral_conds_name=config.get('integral_conds_name'),
                                **kwargs)
        if config.get("discriminators"):
            os.makedirs(f"{_save_path}/1d_disc/", exist_ok=True)
            if len(columns)!=10:
                raise ValueError("Need 10 columns for the discriminator")

            p_target = trans.softmax(data['Data'][columns].values)
            p_target_val = trans.softmax(data['Data_valid'][columns].values)
            
            p_trans = trans.softmax(data['eval_transport'][columns].values)
            p_trans_val = trans.softmax(data['eval_transport_valid'][columns].values)
            
            p_source = trans.softmax(data['source'][columns].values)
            p_source_val = trans.softmax(data['source_valid'][columns].values)
            dist_styles =  [
                plotting.style_trans,
                plotting.style_target,
                plotting.style_source,
                # {"label": "Target", "color": "black"},
                # {"label": "Transport", "color": "red"},
                # {"label": "Source", "color": "blue"},
                # {"label": "Target Val", "color": "red", "linestyle": "--", 'alpha': 0.5},
                # {"label": "Transport", "color": "blue", "linestyle": "--", 'alpha': 0.5},
            ]
            prob_labels = utils.jet_class_labels()
            funcs = {
                prob_labels[2]+'/'+prob_labels[0]: lambda x: np.nan_to_num(np.log(x[:, 2]/(x[:, 0])), nan=999),
                prob_labels[5]+'/'+prob_labels[0]: lambda x: np.nan_to_num(np.log(x[:, 5]/(x[:, 0])), nan=999),
                prob_labels[6]+'/'+prob_labels[0]: lambda x: np.nan_to_num(np.log(x[:, 6]/(x[:, 0])), nan=999),
                prob_labels[6]+'/'+prob_labels[2]: lambda x: np.nan_to_num(np.log(x[:, 6]/(x[:, 2])), nan=999),
            }
            
            for name, func in funcs.items():
                
                bins = np.linspace(*np.percentile(func(p_target_val), [0.1,100]), 20)
                # bins = np.logspace(*np.log10(np.percentile(func(p_target_val), [0.1,99.9])), 20)
                dist_styles[1]['marker'] = 'o'
                dist_styles[1]['linestyle'] = 'none'
                counts_dict, (fig, ax_1, ax_2) = plot.plot_hist(
                                                # func(p_target),
                                                # func(p_trans),
                                                # func(p_source),
                                                func(p_trans_val),
                                                func(p_target_val),
                                                func(p_source_val),
                                                percentile_lst=[0, 100],
                                                style={'bins': bins},
                                                full_plot_bool=True,
                                                ylim=[0.90, 1.1],
                                                dist_styles=dist_styles,
                                                # legend_kwargs = {'title': name}
                                                xerr_on_errorbar=True,
                                                legend_kwargs={"prop":{'size': 25}, 
                                                               "frameon": False, 
                                                               "title_fontsize":25}
                                                        ) # type: ignore
                # ax_2.set_xscale('log')
                ax_2.set_xlabel(f"$\ln(${name}$)$", fontsize=35)
                ax_1.tick_params(axis='y', labelsize=35)
                ax_2.tick_params(axis='both', labelsize=35)
                
                if config.save_figures:
                    name = name.replace('/', '_')
                    misc.save_fig(fig, f"{_save_path}/1d_disc/{name}.pdf",
                                  tight_layout=False)

        elif config.get("hist_kwargs"):
            hist_kw = [{"style": {"bins": 30}, 'percentile_lst': [0.05, 99.999]}] * len(items['xlabels'])
            # hist_kw = [{"style": {"bins":30}, 'percentile_lst': [0, 100]}] * len(items['xlabels'])
            logit_labels = [f'logit {i}' for i in items['xlabels']]
            
            for i in hist_kw:
                i["normalise"] = False
            plotting.calculate_stat_in_hist(hist_kw=hist_kw,
                                            xlabels=logit_labels,
                                            save_path = _save_path,
                                            # plot_individual_points=config.plot_individual_points,
                                            # ylim_hist=[1e-6, 1000],
                                            ylabel_ratio="Ratio",
                                            # merge_bins_threshold=1000,
                                            yscale="log",
                                            ratio_ylim=[0.9, 1.1]
                                            )
        else:

            # raise ValueError("No corner plots")
            target = data['truth'][data['eval_transport'].columns].values
            source = data['source'][data['eval_transport'].columns].values
            transport = data['eval_transport'].values
            logit_labels = [f'logit {i}' for i in items['xlabels']]

            col_size = None
            additional_str = ""

            if len(columns) != 10:
                n_components = 5
                # with PCA
                pca = PCA(n_components=n_components, whiten=True)
                pca_target = pca.fit_transform(target)[:, :n_components]
                pca_transport = pca.transform(transport)[:, :n_components]
                pca_source = pca.transform(source)[:, :n_components]
                additional_str = f"pca_components_{n_components}"
                quantile = [0.01, 0.999]

                labels = [f'PCA {i+1}' for i in range(pca_target.shape[1] if col_size is None else col_size)]
                fontsize=40*len(labels)/4
                label_fontsize= 30
            else:
                additional_str = f"logits_{len(columns)}"
                pca_target = target
                pca_transport = transport
                pca_source = source
                # corner plots between pythia and herwig
                labels = logit_labels
                fontsize=20*len(labels)/(len(labels)/5)
                label_fontsize= 25

            params = {'xtick.labelsize':24, 'ytick.labelsize':24}
            pylab.rcParams.update(params)

            ranges = [tuple(np.quantile(pca_target[:, i], quantile)) for i in range(pca_target.shape[1] if col_size is None else col_size)]

            lw=3
            figure=None
            kwargs = {'plot_density': False, 'plot_datapoints': False, 
                      'range': ranges,
                      'fill_contours':False, 'no_fill_contours':False,
                      'bins':30, "label_kwargs": {"fontsize": label_fontsize},
                      "max_n_ticks": 3, "labelpad": 0.2,
                    "contour_kwargs": {'linewidths': lw},
                    # For histogram edges in 1D plots
                    "hist_kwargs": {'linewidth': lw},
                    # For truth lines
                    # "truth_kwargs": {'linewidth': 2.0},
                      }
            
            kwargs['levels'] = [0.05, 0.5, 0.8, 0.9, 0.95]
            
            kwargs["contour_kwargs"]["colors"]=["black"]
            kwargs["hist_kwargs"]["color"]=["black"]
            kwargs["contour_kwargs"]["linestyles"]=["solid"]
            kwargs["hist_kwargs"]["ls"]="solid"
            figure = corner.corner(pca_target[...,:col_size], 
                                   color = 'black', labels=labels, **kwargs)

            kwargs["contour_kwargs"]["colors"]=["red"]
            kwargs["hist_kwargs"]["color"]=["red"]
            kwargs["contour_kwargs"]["linestyles"]=["dashed"]
            kwargs["hist_kwargs"]["ls"]="dashed"
            figure = corner.corner(pca_transport[...,:col_size], color = 'red', 
                                  fig=figure, **kwargs)

            blue_line = mlines.Line2D([], [], color='red', label='Calibrated', lw=8*len(labels)/4,
                                      ls="--")
            red_line = mlines.Line2D([], [], color='black', label='Target',
                                     linestyle='solid',
                                     lw=8*len(labels)/4)

            figure.legend(handles=[blue_line, red_line],
                          bbox_to_anchor=(0., 1.0, 1., .0), 
                        fontsize=fontsize,
                          loc='upper right',
                          frameon=False)

            if config.save_figures:
                misc.save_fig(figure, f"{_save_path}/corner_latn_size_{len(columns)}_target_v_transport_{additional_str}.pdf", use_tight_layout=False, 
                              bbox_inches='tight')
            
            # target vs source
            figure=None

            kwargs["contour_kwargs"]["colors"]=["black"]
            kwargs["hist_kwargs"]["color"]=["black"]
            kwargs["contour_kwargs"]["linestyles"]=["solid"]
            kwargs["hist_kwargs"]["ls"]="solid"
            figure = corner.corner(pca_target[...,:col_size], color = 'black',
                                   fig=figure, labels=labels, **kwargs)
            
            kwargs["contour_kwargs"]["colors"]=["blue"]
            kwargs["hist_kwargs"]["color"]=["blue"]
            kwargs["contour_kwargs"]["linestyles"]=["dotted"]
            kwargs["hist_kwargs"]["ls"]="dotted"
            figure = corner.corner(pca_source[...,:col_size], color = 'blue',
                                   fig=figure, **kwargs)

            blue_line = mlines.Line2D([], [], color='blue', label='Source', lw=8*len(labels)/4, ls=':')
            red_line = mlines.Line2D([], [], color='black', label='Target', ls='-',
                                     lw=8*len(labels)/4)

            figure.legend(handles=[blue_line,red_line],
                          bbox_to_anchor=(0., 1.0, 1., .0), 
                        fontsize=fontsize,
                          loc='upper right',
                          frameon=False)
            
            if config.save_figures:
                misc.save_fig(figure, f"{_save_path}/corner_latn_size_{len(columns)}_target_v_source_{additional_str}.pdf", use_tight_layout=False, bbox_inches='tight')

            if False:
                n_components=target.shape[1]
                if True:
                    # copula
                    from sklearn.preprocessing import QuantileTransformer
                    scaler = QuantileTransformer()

                    pca_target = scaler.fit_transform(target[:, :n_components])
                    pca_transport = scaler.transform(transport[:, :n_components])
                    ranges = [(0,1)]*n_components
                else:
                    # with PCA
                    pca = PCA(n_components=n_components)
                    pca_target = pca.fit_transform(target[:, :n_components])
                    pca_transport = pca.transform(transport[:, :n_components])
                    # pca_transport = pca.transform(source[:, :n_components])
                    ranges = [tuple(np.percentile(pca_target[:, i], [1,99])) for i in range(pca_target.shape[1] if col_size is None else col_size)]

                logging.info(" Corner plot!")

                # corner plots between pythia and herwig
                col_size = None
                labels = [f'Latent {i+1}' for i in range(pca_target.shape[1] if col_size is None else col_size)]
                figure=None
                kwargs = {'plot_density': False, 'plot_datapoints': False, 'range': ranges,
                            'bins':5}

                figure= corner.corner(pca_target[...,:col_size], color = 'red', **kwargs)

                corner.corner(pca_transport[...,:col_size], color = 'blue',
                            fig=figure,labels=labels, labelpad=0.1,
                            **kwargs)
