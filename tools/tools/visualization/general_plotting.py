"general plotting"
import matplotlib.pyplot as plt
import numpy as np
import copy
# from regex import P
import wandb
import matplotlib
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.colors as mcolors

# from tools.visualization.correlation_plots import plot_feature_spread
from .. import misc
from .plot_utils import fig2img
from ..uncertainty import binomial

#### default definitions ####
np.seterr(invalid='ignore')

COLORS = list(mcolors.BASE_COLORS)
# COLORS = [i.replace('tab:', "") for i in list(mcolors.TABLEAU_COLORS)]

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

FIG_SIZE = (8, 6)
FONTSIZE = 20
LABELSIZE = 20
LEGENDSIZE = 20
RATIO_LABELSIZE = 20
RATIO_FONTSIZE = 20
KWARGS_LEGEND={"prop":{'size': LEGENDSIZE}, "frameon": False,
               "title_fontsize":LEGENDSIZE}
font = {'size': FONTSIZE}

matplotlib.rc('font', **font)

lw=3

#### functions ####

def binary_uncertainty(counts, total=None, normalise:bool=True) -> np.ndarray:
    return binomial(counts=counts, total=total, normalise=normalise)
        
def scatter(*args, scatter_styles:list, name:str, **kwargs):
    log = kwargs.get("log", None)
    # loop over columns
    fig, ax = plt.subplots(1, 1, figsize=kwargs.get("figsize", (8, 6)))
    for data,scatter_style in zip(args,scatter_styles):
        ax.scatter(data[:10_000, 0], data[:10_000, 1], **scatter_style)
    plt.legend()
    ax.set_xlabel(kwargs.get("xlabel", "y"))
    ax.set_ylabel(kwargs.get("xlabel", "x"))

    if isinstance(log, dict):
        log[f"{name}"] =  wandb.Image(fig2img(fig))
        plt.close(fig)

    if kwargs.get("save_path", None) is not None:
        misc.save_fig(fig, f"{kwargs['save_path']}{name}.png")

    return log

def generate_figures(nrows, ncols=2, **kwargs):
    heights = [3, 1]*nrows
    gs_kw={"height_ratios": heights}
    fig, ax =  plt.subplots(ncols=ncols, nrows=nrows*2,gridspec_kw=gs_kw,
                            figsize=(8*ncols,6*nrows),
                            **kwargs)
    ax = np.reshape(ax, (2*nrows, ncols))
    return fig, ax

def plot_stairs(counts, bins,sigma=None, total_sigma=None,total_sigma_label=None,
                ax=None, normalise=True, style={}):

    if normalise:
        counts = counts/np.sum(counts)

    if ax is None: fig, ax = plt.subplots(1,1, figsize = (15,7))

    ax.stairs(counts, edges=bins, **style)

    if sigma is not None:
        ax.stairs(counts+sigma, edges=bins, alpha=0.30,
                    color=style.get("color", None),
                    baseline=counts-sigma, fill=True)

    if total_sigma is not None:
        ax.stairs(counts+total_sigma, edges=bins, alpha=0.15,
                    color=style.get("color", None),
                    label=total_sigma_label if total_sigma_label is not None else None,
                    baseline=counts-total_sigma, fill=True)
    return ax

def hist_uncertainty(arr, weights, bins, stacked_bool=False):
    if isinstance(weights[0], np.ndarray):
        weights = [i**2 for i in weights]
        counts, _, _ = plt.hist(arr, weights=weights, bins=bins, stacked=stacked_bool)
    else:
        counts, _, _ = plt.hist(arr, weights=weights**2, bins=bins, stacked=stacked_bool)
    plt.close()
    if stacked_bool:
        return np.sqrt(counts[-1])
    else:
        return np.sqrt(counts)
    
def binned_errorbar(*dist, bins, **kwargs):
    if kwargs.get("ax", True):
        fig, ax = plt.subplots(1,1)
        
    xerr = np.abs((bins[:-1]-bins[1:])/2)
    xs_value = (bins[:-1]+bins[1:]) / 2


    for i in dist:
        eb1 = ax.errorbar(
            xs_value,i,
            xerr=xerr,
            ls='none',
            **kwargs.get("style", {})
            )
    # eb1[-1][0].set_linestyle("solid")
    return ax

def plot_hist(*args, **kwargs) -> plt.Figure:
    """
    Plot histogram.

    Parameters
    ----------
    *args : np.ndarray
        Input arrays representing different 1D distributions.
    **kwargs : dict
        Additional keyword arguments.
        - binrange : tuple, optional
            Range of the x-axis. Default is (-2, 2).
        - style : dict, optional
            Style options for the histogram plot. Default is {"bins": 20, "histtype": "step"}.
        - remove_placeholder : bool, optional
            Flag indicating whether to remove placeholder values (-999) from the input arrays. Default is False.
        - mask_sig : list of np.ndarray, optional
            Masks for the input arrays. Default is None.
        - percentile_lst : list, optional
            List of percentiles for determining the range of the x-axis. Default is [0.05, 99.95].
        - names : list, optional
            Names for the input distributions. Default is ["dist_0", "dist_1", ...].
        - weights : list, optional
            Weights for the input distributions. Default is None.
        - uncertainties : list, optional
            Uncertainties for the input distributions. Default is [None, None, ...].
        - normalise : bool, optional
            Flag indicating whether to normalize the histogram counts. Default is False.
        - plot_bool : bool, optional
            Flag indicating whether to plot the histogram. Default is True.

    Returns
    -------
    dict
        Output bin counts as a dict
    plt.Figure
        Output figure.
    """

    # Function implementation...

    style = kwargs.setdefault("style", {"bins": 20, "histtype":"step"})
    mask_sig=None
    if kwargs.get("remove_placeholder", False):
        if (-999 in args[0]) & (len(args)>1): #remove -999 as a placeholder i args
            if ("mask_sig" in kwargs):
                mask_sig = [j[np.array(np.ravel(i))!=-999] for i, j in zip(args, kwargs["mask_sig"])]
            args = tuple([np.array(i)[np.array(i)!=-999] for i in args])
        

    if (not "range" in style) and (isinstance(style.get("bins", 20), int)):
        percentile_lst = kwargs.get("percentile_lst", [0.05,99.95])

        # just select percentile from first distribution
        if len(args[0])==2: # stacked array
            style["range"] = np.percentile(args[0][-1], percentile_lst)
        else:
            percentiles = [np.percentile(i, percentile_lst) for i in args]
            style["range"] = [np.min(percentiles), np.max(percentiles)]
        
    names = kwargs.get("names", [f"dist_{i}" for i in range(len(args))])

    weights = kwargs.get("weights")
    if weights is None:
        weights = [np.ones(len(i)) for i in args]

    mask_sig = kwargs.get("mask_sig")
    if mask_sig is None:
        mask_sig = [np.ravel(np.ones_like(i)==1) for i in args]
    
    uncertainties = kwargs.get("uncertainties", [None]*len(args))
   
    counts_dict = {}
    for nr, (name, dist, uncertainty) in enumerate(zip(names,args, uncertainties)):
        fig = plt.figure()
        weight = weights[nr]
        if not all(mask_sig[nr]):
            # for stacked histograms
            mask = mask_sig[nr]
            counts, bins, _ = plt.hist([dist[~mask], dist[mask]], stacked=True,
                                        weights=[weight[~mask], weight[mask]],
                                        **style) # stacked doesnt work for np.histogram
            unc = [None, hist_uncertainty([dist[~mask], dist[mask]], weights=[weight[~mask], weight[mask]],
                                          bins=bins, stacked_bool=True)
                   if uncertainty is None else uncertainty]

            if kwargs.get("normalise", False):
                counts = list(counts/np.sum(counts,1)[:,None])
            else:
                counts = list(counts)

            weight = [weight, weight]
        else:
            # for non-stacked histograms
            counts, bins, _ = plt.hist(dist, weights=weight, **style)

            if weight is None:
                unc = [np.sqrt(counts)if uncertainty is None else uncertainty]
                weight = np.ones((len(dist)))
            else:
                unc = [hist_uncertainty(dist, weights=weight, bins=bins,
                                        stacked_bool=False)
                    if uncertainty is None else uncertainty]

            counts = [counts]

            weight = [weight]


        counts_dict[name] = {"counts": counts, "unc": unc, "weight": weight}

        plt.close(fig)

    counts_dict["bins"] = bins
    if kwargs.get("full_plot_bool", False):
         
        # create figure with ratio plot
        axes = kwargs.get('ax')
        if axes is None:
            fig, (ax_1, ax_2) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(9, 8), sharex="col"
                )
        elif not isinstance(axes, list):
            raise ValueError("ax should be a list of axes")
        elif axes is not None:
            ax_1, ax_2 = axes
        # plot histogram
        counts_dict, _ = plot_hist_1d(counts_dict, ax=ax_1, **kwargs)

        styles = kwargs.pop("dist_styles", None)
    
        if styles is not None:
            kwargs["styles"] = styles
        
        # plot ratio
        plot_ratio(counts_dict, truth_key="dist_0", ax=ax_2, **kwargs)

        return counts_dict, (fig, ax_1, ax_2)

    elif kwargs.get("plot_bool", True):
        return plot_hist_1d(counts_dict, **kwargs)
    else:
        return counts_dict, None
    
def plot_hist_integral(distributions, truth_key, var_names, conds_bins,
                       plot_kwargs={}, conds_names=None, save_path=None,
                       **kwargs):
    "still requires some work"

    if isinstance(conds_bins, (list, np.ndarray)):
        conds_bins = np.round(np.percentile(distributions[truth_key]["conds"],
                                                    conds_bins, 0),3)
        ax = plot_hist_integration_over_bins(distributions, bins=conds_bins,
                                             plot_kwargs=plot_kwargs,
                                             var_names=var_names, **kwargs)
    else:
        if conds_names is None:
            raise ValueError("When giving conds_bins as dict, conds_names is required!")

        for key, item in conds_bins.items():
            n_bins = len(item)

            fig, ax_all = generate_figures(n_bins-1, len(var_names))
            for n_row in range(n_bins):
                bins = np.c_[[items[n_row:n_row+2].T for i, items in conds_bins.items()]].T
                # ax = ax_all[2*n_row: 2*n_row+2, 0]
                plot_hist_integration_over_bins(distributions, bins=bins,
                                                    plot_kwargs=plot_kwargs,
                                                    ax=ax_all[2*n_row: 2*n_row+2],
                                                    var_names=var_names,
                                                    conds_col_nr=np.argmax(key==np.array(conds_names)), 
                                                    **kwargs)

            if save_path is not None:
                misc.save_fig(
                    fig,
                    f"{save_path}/{var_names}_{key}_{item}.{kwargs.get('format', 'png')}"
                    )

def plot_hist_integration_over_bins(distributions, bins, plot_kwargs, legend_title,
                                    var_names, ax=None, conds_col_nr=None,
                                    save_path=None,**kwargs):
    legend_title = kwargs.get("legend_title", "")
    for (low, high) in zip(bins[:-1], bins[1:]):
        dists = []
        labels=[]
        bootstraps=[]
        systematics=[]
        weights=[]
        for key, values in distributions.items():
            if conds_col_nr is None:
                mask_conds =  np.all([(low[i]<=values["conds"][:,i]) &
                                    (high[i]>=values["conds"][:,i])
                                    for i in range(len(low))], 0)
            else:
                mask_conds =  ((low[0]<=values["conds"][:,conds_col_nr]) &
                                    (high[0]>=values["conds"][:,conds_col_nr]))
            
            dists.append(values["dist"][mask_conds])
            labels.append(values["labels"][mask_conds]==1)
            bootstraps.append([i[mask_conds] for i in values["bootstraps"]]
                              if "bootstraps" in values else [])
            systematics.append([i[mask_conds] for i in values["systematics"]]
                               if "systematics" in values else [])
            weights.append(values["weights"][mask_conds] if "weights" in values
                           else np.ones(values["labels"][mask_conds].shape))

        for col_nr in range(len(var_names)):
            plot_args = copy.deepcopy(plot_kwargs)
            if isinstance(plot_args, list):
                plot_args=plot_args[col_nr]

            title_str = f"{legend_title} pT: [{', '.join(map(str, low))}: {', '.join(map(str, high))})"

            if ax is None:
                fig, ax = generate_figures(1, 1, sharex="col")
            # histogram
            counts, _ = plot_hist(*[i[:,col_nr] for i in dists],
                                  plot_bool=False,mask_sig=labels,
                                  weights=weights,
                                  **plot_args
                                  )
            
            if len(bootstraps[1])>0: # TODO should not be hard coded
                counts_boot, _ = plot_hist(*[i[:,col_nr] for i in bootstraps[1]],
                                    plot_bool=False,mask_sig=[labels[1]]*len(bootstraps[1]),
                                    **plot_args)
                bootstraps_unc = np.std([counts_boot[i]["counts"][-1]
                                         for i in counts_boot if "dist" in i],0)
                counts["dist_1"]["unc"][-1] = bootstraps_unc

            if len(systematics[1])>0:
                counts_sys, _ = plot_hist(*[i[:,col_nr] for i in systematics[1]],
                                    plot_bool=False,mask_sig=[labels[1]]*len(systematics[1]),
                                    **plot_args)

            counts, _ = plot_hist_1d(counts, ax=ax[0, col_nr],
                                     legend_kwargs={"title":title_str},
                                     **plot_args)

            if kwargs.get("log_y", False):
                ax[0, col_nr].set_yscale("log")

            # ratio plot
            plot_args["styles"] = plot_args.pop("dist_styles", None)
            plot_ratio(counts, truth_key="dist_0", ax=ax[1, col_nr],
                            # ylim=kwargs.get("ylim", [0.95, 1.05]),
                            # styles=copy.deepcopy(dist_styles),
                            zero_line_unc=True,
                            **plot_args)

            ax[1, col_nr].set_xlabel(var_names[col_nr])
            if "ylim" in plot_args:
                ax[1, col_nr].set_ylim(plot_args["ylim"])
            if (save_path is not None) & (not kwargs.get("single_figure", False)):
                misc.save_fig(
                    fig,
                    f"{save_path}/{var_names}_{low}_{high}_{col_nr}.{kwargs.get('format', 'png')}"
                    )
    # return ax

    

def plot_hist_1d(counts_dict:dict, **kwargs):
    """Plot histogram

    Parameters
    ----------
    counts : dict
        Should contain:   bins:[], truth:{counts:[], unc:[], weights: []},
                          different_counts:{counts:[],unc:[], weights: []}

    Returns
    -------
    dict, ax
        return counts and ax
    """
    if "bins" in counts_dict:
        bins = counts_dict["bins"]
    else:
        raise ValueError("counts_dict missing bins")
    
    ax = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', length=4)
    
    # define bins
    xs_value = (bins[:-1]+bins[1:]) / 2
    # fixing the bound of the x axis because of steps
    if kwargs.get("xerr_on_errorbar", False):
        xerr = np.abs((bins[:-1]-bins[1:])/2)
    else:
        # xerr[1:-1] = np.zeros_like(xerr[1:-1])
        xerr = np.zeros_like(bins[:-1])
    
    # get dist_styles or make empty at scale of counts_dict
    dist_styles = kwargs.get("dist_styles", [{}]*len(counts_dict))

    bkg_labels = kwargs.get("background_labels", ["Background"]*len(counts_dict))

    nr_iter = 0
    for name, counts in counts_dict.items():
        if name == "bins":
            continue
        
        # if uncertainties or weight are missing
        counts.setdefault("unc", [np.zeros_like(counts["counts"])[0]])
        counts.setdefault("weight", [np.ones_like(counts["counts"])[0]])
        # this type of loop is need if hist(stacked=True)
        for nr_count, (count, unc, weight) in enumerate(zip(counts["counts"],
                                                            counts["unc"],
                                                            counts["weight"])):

            style = copy.deepcopy(dist_styles[nr_iter])

            if (len(counts["counts"])==2) & (nr_count ==0):
                style["label"] = bkg_labels[nr_iter]
                style["alpha"] = 0.5

            style.pop("drawstyle", None)
            if ('linewidth' or 'lw') not in style:
                style.setdefault("linewidth", lw)

            if not kwargs.get("legend_bool", True):
                style.pop("label", None)

            if not isinstance(weight[0], float):
                sum_weight = sum([np.sum(i) for i in weight])
            else:
                sum_weight = np.sum(weight)

            yerr=(None if unc is None else
                  binary_uncertainty(count, total=sum_weight) if kwargs.get("normalise", True)
                  else unc)

            hist_counts = count/sum_weight if kwargs.get("normalise", True) else count

            if "marker" in style:
                # style["linestyle"]="none"
                eb1 = ax.errorbar(
                    xs_value,
                    hist_counts,
                    xerr=xerr,
                    yerr=yerr,
                    **style
                    )
            else:
                yerr = 0 if yerr is None else yerr
                if not isinstance(weight[0],  (int, float,  np.float32,  np.float64)): # only for stacked=True
                    ax.stairs(hist_counts[-1], bins, baseline=hist_counts[-1]-yerr[-1],fill=True, alpha=0.1, color = style.get("color", None))
                    ax.stairs(hist_counts[-1], bins, baseline=hist_counts[-1]+yerr[-1],fill=True, alpha=0.1, color = style.get("color", None))
                    for nr, (i, label) in enumerate(zip(hist_counts, kwargs["stacked_labels"])): # need to have style be list
                        ax.stairs(i, bins, zorder=1/(nr+1), **style, label=label)
                else:
                    ax.stairs(hist_counts, bins, baseline=hist_counts-yerr,fill=True, alpha=0.1, color = style.get("color", None))
                    ax.stairs(hist_counts, bins, baseline=hist_counts+yerr,fill=True, alpha=0.1, color = style.get("color", None))
                    ax.stairs(hist_counts, bins, **style)#, zorder=1/(nr_count+1))
        nr_iter+=1

    if kwargs.get("legend_bool", True):
        legend_kwargs = KWARGS_LEGEND.copy()
        legend_kwargs.update(kwargs.get("legend_kwargs", {}))
        ax.legend(**legend_kwargs)
    
    
    ax.tick_params(axis="both", which="major", labelsize=LABELSIZE)
    ylabel = "Normalised counts" if kwargs.get("normalise", True) else "#"
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    if kwargs.get("log_yscale", False):
        ax.set_yscale("log")
        
    return counts_dict, ax

def plot_ratio(counts:dict, truth_key:str, **kwargs):
    """Ratio plot

    Parameters
    ----------
    counts : dict
        Counts from plot_hist
    truth_key : str
        Which key in counts are the truth
    kwargs:
        Add ylim if you want arrow otherwise they are at [0,2]
    Returns
    -------
    _type_
        _description_
    """
    ax = kwargs.get("ax", None)
    alpha = kwargs.get("alpha", 0.3)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    bins = counts["bins"]
    overwrite_uncertainty = kwargs.get("overwrite_uncertainty", False)
    normalise_counts = kwargs.get("normalise", True) 

    styles = copy.deepcopy(kwargs.get("styles", [{} for _ in counts.keys()]))
    ylim = kwargs.get("ylim", [0,2])
    xs_value = (bins[:-1]+bins[1:]) / 2
    xerr = (bins[:-1]-bins[1:])/2
    if "counts" in counts[truth_key]:
        counts_truth = counts[truth_key]["counts"][-1]
    else:
        counts_truth = counts[truth_key]
    counts_truth=np.array(counts_truth)
    rel_counts_truth = counts_truth/counts_truth.sum()
    if kwargs.get("zero_line", True):
        # try:
        #     color = styles[np.argmax(np.in1d(list(counts.keys())[:-1], truth_key))]["color"]
        # except:
        #     color="black"
        try:
            # color = styles[np.argmax(np.in1d(list(counts.keys())[:-1], truth_key))][
            #     "color"
            # ]
            style = copy.deepcopy(
                styles[np.argmax(np.in1d(list(counts.keys())[:-1], truth_key))]
                )
        except:
            style = {'color': 'black', 'lw':2, 'ls': 'dashed'}
            
        style['zorder'] = -1
        ax.plot(
            bins,
            np.ones_like(bins),
            # label="Zero line",
            # color=color,
            # linewidth=2,
            # linestyle="dashed",
            **style,
            # zorder=-1,
        )
        ones_line = np.ones_like(xs_value)
        if kwargs.get("zero_line_unc", True):
            if normalise_counts & overwrite_uncertainty:
                unc = binary_uncertainty(counts[truth_key]["counts"][-1])
                unc = unc/rel_counts_truth
            elif (not overwrite_uncertainty) and ('unc' in counts[truth_key]):
                unc = np.divide(counts[truth_key]["unc"][-1],counts_truth)
            else:
                unc = np.divide(np.sqrt(counts_truth), counts_truth)
                
            unc = np.nan_to_num(unc, nan=10, posinf=10)
            
            # ax.fill_between(xs_value, ones_line-np.abs(unc),
            #                 ones_line+np.abs(unc), color=color,
            #                 alpha=alpha/2, zorder=-1, label="Uncertainty")
            style['alpha'] = alpha
            style['label'] = "Uncertainty"
            style.pop('marker', None)
            style.pop('markersize', None)
            ax.stairs(ones_line+np.abs(unc), edges=bins,
                    baseline=ones_line-np.abs(unc), fill=True, **style)
            # ax.stairs(
            #     one_line + np.abs(unc),
            #     bins,
            #     # alpha=kwargs.get('alpha', 0.5),
            #     # color=color,
            #     baseline=one_line - np.abs(unc),
            #     fill=True,
            #     # zorder=-1,
            #     **style,
            # )

        if "total_unc" in counts[truth_key]:
            total_unc = np.array(counts[truth_key]["total_unc"][-1])
            if normalise_counts:
                total_unc = (total_unc/counts_truth.sum()/rel_counts_truth)
            else:
                total_unc = total_unc/counts_truth

            style['alpha'] = alpha/2
            ax.stairs(ones_line+total_unc, bins, baseline=ones_line-total_unc, 
                        fill=True, **style)
            # ax.stairs(ones_line+total_unc, bins, alpha=alpha,
            #         color=color,
            #         baseline=ones_line-total_unc, fill=True)
    nr = 0
    for (name, count), style in zip(counts.items(), styles):
        if (name == "bins") or (name == truth_key):
            continue

        if "counts" in count:
            if not isinstance(count.get("counts",None), list):
                # not isinstance(count.get("unc",None), list)
                raise TypeError("Counts in dict has to be a list")
            count_pr_bins = np.array(count["counts"][-1])
        else:
            count_pr_bins = np.array(count)
            
        if "unc" in count:
            if not isinstance(count.get("unc",None), list):
                # not isinstance(count.get("unc",None), list)
                raise TypeError("Counts in dict has to be a list")
            menStd = np.array(count["unc"][-1])
            if len(menStd.shape)>1:
                menStd = menStd[-1]
        else:
            menStd = np.zeros_like(count_pr_bins)
        with np.errstate(divide='ignore'):
            if normalise_counts:
                y_counts = (count_pr_bins/count_pr_bins.sum())/rel_counts_truth
                yerr_relative = (menStd/count_pr_bins.sum())/rel_counts_truth
            else:
                y_counts = count_pr_bins/counts_truth
                yerr_relative = menStd/counts_truth

        if kwargs.get("reverse_ratio", False):
            y_counts = 1/y_counts

        # up or down error if outside of ylim
        yerr_relative = np.nan_to_num(yerr_relative, nan=10, posinf=10)
        mask_down = ylim[0]>=y_counts
        mask_up = ylim[1]<=y_counts

        
        #additional uncertainties
        if "total_unc" in count:
            total_unc = np.array(count["total_unc"][-1])
            if normalise_counts:
                total_unc = (total_unc/count_pr_bins.sum()/rel_counts_truth)
            else:
                total_unc = total_unc/counts_truth

            ax.stairs(y_counts+total_unc, bins, alpha=alpha,
                    color=style.get("color", None),
                    baseline=y_counts-total_unc, fill=True,
                    lw=lw)
            

        linestyle = style.pop("linestyle", "solid")
        style['lw'] = lw
        # plotting

        # Plot horizontal error bars with dashed lines
        eb1 = ax.errorbar(xs_value, y_counts, yerr=yerr_relative,
                        xerr = np.abs(xerr), linestyle='none', **style
                        )
        # if linestyle != 'none':
        #     for line in ax.get_lines():
        #         line.set_linestyle(linestyle)
        if linestyle != 'none':
            eb1[-1][0].set_linestyle(linestyle)
            eb1[-1][1].set_linestyle(linestyle) 

        #marker up
        if ("label" not in style) & kwargs.get("legend_bool", True):
            style["label"] = name
        style.update({"marker": '^', "s": 35, "alpha": 1})
        for i in ["linestyle", "markersize"]:
            style.pop(i, None)
        ax.scatter(xs_value[mask_up],
                    np.ones(mask_up.sum())*(ylim[1]-ylim[1]/100),
                    **style)
        
        #marker down
        style["marker"] = 'v'
        ax.scatter(xs_value[mask_down],
                    np.ones(mask_down.sum())*(ylim[0]+ylim[0]/100),
                    **style)

        nr+=1

    ax.set_ylabel(kwargs.get("ylabel", "Ratio"), fontsize=RATIO_FONTSIZE)
    ax.set_ylim(ylim)
    ax.tick_params(axis="both", which="major", labelsize=RATIO_LABELSIZE)
    return ax

def create_2dhist(*other_dist, **kwargs):
    names = kwargs.get("names", [f"dist_{i}" for i in range(len(other_dist))])
    bins = kwargs.get("bins", 20)
    x_y_range = kwargs.get("range", None)
    counts_others = {i:{} for i in names}
    weights = kwargs["weights"]
    for nr, (i, name) in enumerate(zip(other_dist, names)):
        if i.shape[1] != 2:
            raise ValueError(f"{name} shape is no 2d: {i.shape}")
        if weights[nr] is None:
            h, xedges, yedges = np.histogram2d(i[:,0], i[:,1],
                                                range=x_y_range,
                                                bins=bins)
            unc = np.sqrt(h)
        else:
            h, xedges, yedges = np.histogram2d(i[:,0], i[:,1], range=x_y_range,
                                            bins=bins, weights=weights[nr])
            unc = np.sqrt(np.histogram2d(i[:,0], i[:,1], range=x_y_range,
                                            bins=bins, weights=weights[nr]**2)[0])
        counts_others[name]["counts"] = h
        counts_others[name]["unc"] = unc

    return counts_others

def plot_roc_curve(truth, pred, weights=None, label="",
                   fig=None, uncertainty=False,
                   **kwargs):
    fpr, tpr, _ = roc_curve(truth, pred, sample_weight=weights, drop_intermediate=False
                            )
    plot_bool = kwargs.get("plot_bool", True)
    
    auc = roc_auc_score(truth, pred, sample_weight=weights)
    if plot_bool:
        if (fig is None) and plot_bool:
            fig = plt.figure(figsize=(8,8))
        auc_str = str( round(auc, 4) )

        label += f"AUC: {auc_str}"
        plt.plot(fpr, tpr, linewidth=2, label=label, 
                 **kwargs.get("plot_kwargs", {'color':"blue"}))
        plt.plot([0, 1], [0, 1], 'k--')
        
        if uncertainty:
            N = len(tpr) if weights is None else np.sum(weights)
            uncertainty = np.sqrt(tpr*(1-tpr)/N)/tpr
            plt.fill_between(fpr, tpr+uncertainty, tpr-uncertainty, alpha=0.5)
        small_scaling = 1e-3
        plt.axis([-small_scaling, 1+small_scaling, -small_scaling, 1+small_scaling])
        plt.xticks(np.arange(0,1, 0.1), rotation=90)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc='best', frameon=False, **kwargs.get('legend_kwargs', {}))
        plt.tight_layout()
    return tpr, fpr, auc, fig

# def envolpe_hist()

if __name__ == "__main__":
    # %matplotlib widget
    if False:
        style_target = {"marker":"o", "color":"black", "label":"Data", "linewidth":0}
        style_source = {"linestyle": "dotted","color":"blue", "label":r"$b$-jets", "drawstyle":"steps-mid"}
        style_trans = {"linestyle": "dashed", "color":"red", "label":"Transported", "drawstyle":"steps-mid"}
        fig, (ax_1, ax_2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(9, 5), sharex="col"
            )


        target=np.random.uniform(-15, 15, 10_000)
        source=np.random.uniform(-15, 15, 10_000)
        trans=np.random.uniform(-15, 15, 10_000)
        bins = [-15, 0.6650, 2.1950, 3.2450, 4.5650, 15]
        mask_sig = [source<0.5, np.ones_like(trans), np.ones_like(target)]
        data, ax_1 = plot_hist(source,trans,target, dist_styles=[style_source, style_trans, style_target],
                            ax=ax_1, mask_sig=mask_sig, style={"bins":bins})
        style_source.pop("drawstyle")
        style_trans.pop("drawstyle")
        ax_2 = plot_ratio(counts=data, ax=ax_2, truth_key="dist_2",
                styles=[style_source, style_trans], ylim=[0.98,1.02])
        fig.tight_layout()
    else:
        # fig5 = plt.figure(constrained_layout=True)
        n_fig=5
        # widths = [3, 1]*n_fig
        heights = [3, 1]*n_fig
        # heights = [1, 3, 2]
        gs_kw={"height_ratios": heights}
        # spec5 = fig5.add_gridspec(ncols=2, nrows=n_fig*2,
        #                         #   width_ratios=widths,
        #                           height_ratios=heights)
        fig, ax =  plt.subplots(ncols=2, nrows=n_fig*2,gridspec_kw=gs_kw,
                                figsize=(8,6*n_fig))