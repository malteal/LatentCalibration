"plotting utils - should be looked at"
import os
import pickle
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..torch import torch_utils as utils
from . import transformations as trans
from . import misc
try:
    import tools.visualization.atlas_utils as atlas_utils
except (ImportError, ModuleNotFoundError):
    print("atlas_utils has import issues - alot of non-essential packages in atlasutils")


from tools.tools.visualization.general_plotting import (plot_hist, plot_hist_1d, 
                                                  plot_hist_integral, plot_hist_integration_over_bins,
                                                  plot_ratio, plot_stairs,
                                                  generate_figures)
# numpy warnings
np.seterr(all="ignore")

default_pt = [20,30,40,60,85,110,140,175,250,400]

# general plotting stuff
FIG_SIZE = (8, 6)
FONTSIZE = 16
LABELSIZE = 16
LEGENDSIZE = 16
KWARGS_LEGEND = {
    "prop": {"size": LEGENDSIZE},
    "frameon": False,
    "title_fontsize": LEGENDSIZE,
}

COLORS = ['black', 'blue', 'red', 'green', 'purple', 'brown',
          'pink', 'gray', 'olive', 'cyan', 'orange']

font = {  #'family' : 'normal',
    # 'weight' : 'bold',
    "size": 16
}

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


matplotlib.rc("font", **font)


def bold(string):
    return f"$\\bf{string}$"

def italic(string):
    return f"$\\it{string}$"

class ScalarFigures:
    def __init__(
        self,
        f_func: callable,
        conds: torch.Tensor,
        others: torch.Tensor,
        xidx: int,
        xlabel: str,
        xmin: int = -10,
        xmax: int = 10,
        npts: int = 100,
    ):
        self.xlabel = xlabel
        self.f_func = f_func
        device = conds.device.type + f":{conds.device.index}"
        xs_value = np.mgrid[xmin : xmax : npts * 1j].reshape((-1, 1))
        if others.is_cuda:
            others = np.tile(others.cpu().detach().numpy(), (npts, 1))
            catconds = torch.cat([conds.cpu().unsqueeze(1)] * npts, axis=0).to(device)
        else:
            others = np.tile(others.detach().numpy(), (npts, 1))
            catconds = torch.cat([conds.unsqueeze(1)] * npts, axis=0)

        self.ten = np.concatenate(
            [others[:, 0:xidx]] + [xs_value] + [others[:, xidx + 1 :]], axis=1
        )

        ten = torch.Tensor(self.ten)
        ten.requires_grad = True
        if conds.is_cuda:
            ten = ten.to(device)

        if conds.shape[0] == 0:
            catconds = ten[:, :0]
        self.catconds = catconds
        self.potential = f_func(self.catconds, ten)

        grad = utils.grad(self.potential, ten)
        if grad.is_cuda:
            self.xnew = grad[:, xidx].cpu().detach().numpy()
        else:
            self.xnew = grad[:, xidx].detach().numpy()

        self.xs_value = xs_value.reshape(npts)
        self.conds_value = str(np.round(self.catconds.mean().cpu().detach().numpy(), 4))

    def plot_potential(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(
            self.ten,
            self.potential.cpu().detach().numpy(),
            linestyle="solid",
            color="blue",
            label=f"g network potential at {self.conds_value}",
        )

        ax.set_xlabel("Source", fontsize=FONTSIZE)
        ax.set_ylabel("Potential", fontsize=FONTSIZE)
        plt.legend(loc="best")

        return fig

    def plot_scalar1D(self):
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(111)

        ax.plot(self.xs_value, self.xnew, linestyle="dashed", color="red")
        ax.plot(self.xs_value, self.xs_value, linestyle="dotted", color="black")

        ax.set_xlabel(self.xlabel, fontsize=FONTSIZE)
        ax.set_ylabel("transported " + self.xlabel, fontsize=FONTSIZE)

        return fig

    def plot_deltascalar1D(self, title=None, plot_bounds=True, col_nr=0) -> plt.Figure:
        """Plotting 1d scalars
        Returns
        -------
        plt.Figure
            output the fiugre
        """

        fig = plt.figure()
        ax_1 = fig.add_subplot(111)

        if title is not None:
            ax_1.set_title(title)
        resolution = np.abs(self.ten[0, col_nr] - self.ten[1, col_nr])
        ax_1.plot(
            self.xs_value[1:],
            (self.xnew[1:] - self.xnew[:-1]) - resolution,
            linestyle="dashed",
            color="red",
            label=r"$\nabla^2$g",
        )  # self.ten[0]-self.ten[1] removes natural gradeint
        lims = [ax_1.get_xlim(), ax_1.get_ylim()]
        if plot_bounds:
            ax_1.hlines(
                -resolution,
                self.xs_value[1],
                self.xs_value[-1],
                linestyle="dotted",
                color="black",
                label="Gradient bound",
            )

        ax_1.set_xlim(*lims[0])
        ax_1.set_ylim(*lims[1])
        ax_1.set_xlabel(self.xlabel, fontsize=FONTSIZE)
        ax_1.set_ylabel(r"$\Delta$ transported", fontsize=FONTSIZE)
        plt.legend(
            title=f"Conditional value {self.conds_value}",
            loc="best",
            title_fontsize=LEGENDSIZE,
        )

        return fig

    def plot_deltascalar1D_between_source(self) -> plt.Figure:  # not in use
        """Plotting 1d scalars
        Returns
        -------
        plt.Figure
            output the fiugre
        """

        fig = plt.figure()
        ax_1 = fig.add_subplot(111)

        ax_1.plot(
            self.xs_value, self.xnew - self.xs_value, linestyle="dashed", color="red"
        )
        ax_1.plot(
            self.xs_value,
            np.zeros_like(self.xs_value),
            linestyle="dotted",
            color="black",
        )

        ax_1.set_xlabel(self.xlabel, fontsize=FONTSIZE)
        ax_1.set_ylabel(r"$\Delta$ " + self.xlabel, fontsize=FONTSIZE)

        return fig


def plot_arrows(  # pylint: disable=too-many-locals
    f_func: callable, conds: torch.Tensor, others: torch.Tensor, **kwargs: dict
) -> plt.Figure:
    """plotting gradient arrows

    Parameters
    ----------
    f_func : callable
        ML model
    conds : torch.Tensor
        Conditional input
    others : torch.Tensor
        other Conditional input
    kwargs : dict
        additional inputs

    Returns
    -------
    plt.Figure
        output figure
    """
    device = conds.device.type + f":{conds.device.index}"
    npts = kwargs.get("npts", 21)
    n2dpts = npts * npts
    xlim = kwargs.get("xlim", [-15, 15])
    ylim = kwargs.get("ylim", [-15, 15])
    xs_value = np.mgrid[xlim[0] : xlim[1] : npts * 1j]
    ys_value = np.mgrid[ylim[0] : ylim[1] : npts * 1j]
    
    x_mesh, y_mesh = np.meshgrid(xs_value, ys_value)

    if others.is_cuda:
        others = np.tile(others.cpu().detach().numpy(), (n2dpts, 1))
        if conds.shape[0]:  # & (not conds.isnan()):
            catconds = torch.cat([conds.cpu().unsqueeze(1)] * n2dpts, axis=0).to(device)
        else:
            catconds = torch.zeros((n2dpts, 0)).to(device)
    else:
        others = np.tile(others.detach().numpy(), (n2dpts, 1))
        if conds.shape[0]:
            catconds = torch.cat([conds.unsqueeze(1)] * n2dpts, axis=0)
        else:
            catconds = torch.zeros((n2dpts, 0))

    if kwargs["yidx"] > kwargs["xidx"]:
        ten = np.concatenate(
            [others[:, 0 : kwargs["xidx"]]]
            + [x_mesh.reshape((-1, 1))]
            + [others[:, kwargs["xidx"] + 1 : kwargs["yidx"]]]
            + [y_mesh.reshape((-1, 1))]
            + [others[:, kwargs["yidx"] + 1 :]],
            axis=1,
        )
    else:
        ten = np.concatenate(
            [others[:, 0 : kwargs["yidx"]]]
            + [y_mesh.reshape((-1, 1))]
            + [others[:, kwargs["yidx"] + 1 : kwargs["xidx"]]]
            + [x_mesh.reshape((-1, 1))]
            + [others[:, kwargs["xidx"] + 1 :]],
            axis=1,
        )

    ten = torch.Tensor(ten)
    ten.requires_grad = True

    if conds.is_cuda:
        ten = ten.to(device)

    transport, zs_function_output = f_func.transport(ten, catconds) 

    zs_function_output = zs_function_output.cpu().detach().numpy()
    transport = transport.cpu().detach().numpy()
    ten = ten.cpu().detach().numpy()

    arrows = transport - ten

    arrowx = arrows[:, kwargs["xidx"]].reshape(x_mesh.shape)
    arrowy = arrows[:, kwargs["yidx"]].reshape(y_mesh.shape)

    fig,ax_1 = plt.subplots(1,1,figsize=(8,8))

    if kwargs.get("plot_cvx_scale"):
        #cvx_potential
        coloring = zs_function_output.reshape((npts, npts))
        # label = "g(x)"
    else:
        # transport amount
        # vec  = (transport[:,[kwargs["xidx"], kwargs["yidx"]]]
        #         -ten[:, [kwargs["xidx"], kwargs["yidx"]]])
        vec  = (transport-ten)
        
        vec_size = np.sqrt(np.sum(vec**2, 1))

        coloring = vec_size.reshape((npts, npts))

        # label = "||arrow$_{\mathrm{2d}}$||"
    mesh = ax_1.pcolormesh(
        x_mesh, y_mesh, coloring, shading="gouraud", cmap='autumn_r',
    )
    cbar = fig.colorbar(mesh, ax=ax_1)
    cbar.set_label('Arrow length')

    ax_1.quiver(x_mesh, y_mesh, arrowx, arrowy, angles="xy")

    ax_1.set_xlabel(kwargs["xlabel"], fontsize=1.5*FONTSIZE)
    ax_1.set_ylabel(kwargs["ylabel"], fontsize=1.5*FONTSIZE)
    ax_1.set_xlim([xlim[0], xlim[1]])
    ax_1.set_ylim([ylim[0], ylim[1]])
    if "title" in kwargs:
        ax_1.set_title(kwargs["title"], loc='left',
                       fontdict={"fontsize": 18})
    fig.tight_layout()
    return fig


def plot_callback(  # pylint: disable=too-many-locals,too-many-statements
    *,
    g_func: callable,
    sources: torch.Tensor,
    targets: torch.Tensor,
    nonconvex_inputs: list,
    convex_inputs: list,
    writer: callable = None,
    epoch: int = None,
    outfolder: str = None,
    prefix: str = "",
    datatype: str = "cern",
    style_target={
        "marker": "o",
        "color": "black",
        "label": "Data",
        "linewidth": 0,
        "markersize": 4,
    },
    style_source={
        "linestyle": "dotted",
        "color": "blue",
        "label": r"$b$-jets",
        "drawstyle": "steps-mid",
    },
    style_trans={
        "linestyle": "dashed",
        "color": "red",
        "label": "Transported",
        "drawstyle": "steps-mid",
    },
    **kwargs,
) -> None:
    """Generate the validation plot and metrics

    Parameters
    ----------
    g_func : callable
        ML model
    sources : torch.Tensor
        source distribution
    targets : torch.Tensor
        target distribution
    nonconvex_inputs : list
        number of non convex input
    convex_inputs : list
        number of convex input
    writer : callable, optional
        Tensorboard writer, by default None
    epoch : int, optional
        epoch number for naming, by default None
    outfolder : str, optional
        output path of the images, by default None
    prefix : str, optional
        Prefix for naming images, by default ""
    datatype : str, optional
        toy or cern data, by default "cern"
    """
    device = g_func.device

    nnonconvex = len(nonconvex_inputs)
    # nconvex = len(convex_inputs)

    condssim = sources["conds"].cpu().detach().numpy()
    logitpssim = sources["transport"].cpu().detach().numpy()

    condsdata = targets["conds"].cpu().detach().numpy()
    logitpsdata = targets["transport"].cpu().detach().numpy()

    logitpscal = sources["eval_transport"].cpu().detach().numpy()
    if "uncertainty" in sources:
        uncertainty = sources["uncertainty"].cpu().detach().numpy()
    else:
        uncertainty = None

    avgconds = torch.tensor(condssim.mean(0), device=device)
    avgfeats = torch.tensor(logitpssim.mean(0), device=device)
    ylim_ratio = [0.9, 1.1]

    # sig template
    sig_sim = (sources["sig_mask"] == 1).cpu().detach().numpy()

    if (datatype == "cern") & (nnonconvex == 1):
        condssim = np.exp(condssim)
        condsdata = np.exp(condsdata)
        legend_title = f"{atlas_utils.atlas_str}\n"
        legend_title += (
            rf"p$_T$: [{str(round(condsdata.min(),2))},"
            rf" {str(round(condsdata.max(),2))}] GeV"
        )
    else:
        legend_title = None

    if logitpsdata.shape[1] == 1:
        var_name = convex_inputs[0]

        name = f"{prefix}_1d_histogram_{epoch}_"

        fig, (ax_1, ax_2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(9, 5), sharex="col"
        )
        dist_styles = [style_target.copy(), style_source.copy(), style_trans.copy()]
        count_data, ax_1 = plot_hist(
            np.ravel(logitpsdata),
            np.ravel(logitpssim),
            np.ravel(logitpscal),
            dist_styles=dist_styles,
            mask_sig=[
                np.ones(len(logitpsdata)) == 1,
                sig_sim,
                np.ones(len(logitpsdata)) == 1,
            ],
            legend_kwargs={"title": legend_title},
            ax=ax_1,
            uncertainties=[None, None, uncertainty],
            percentile_lst=[0, 100],
        )
        dist_styles[1].pop("drawstyle", None)
        dist_styles[2].pop("drawstyle", None)
        ax_2 = plot_ratio(
            counts=count_data,
            ax=ax_2,
            truth_key="dist_0",
            styles=dist_styles,
            ylim=ylim_ratio,
        )
        ax_2.tick_params(axis="both", which="major", labelsize=LABELSIZE)
        ax_1.set_ylabel("Normalised entries", fontsize=FONTSIZE)
        ax_2.set_xlabel(rf"$dist_0$", fontsize=FONTSIZE)

        fig.tight_layout()
        misc.save_fig(fig, name, f"{outfolder}/1d_hist/", writer, epoch)
        fig.clf()

        scalar_fig = ScalarFigures(
            g_func,
            avgconds,
            avgfeats,
            0,
            xlabel="Source",
            xmin=logitpssim.min() - logitpssim.min() / 10,
            xmax=logitpssim.max() + logitpssim.max() / 10,
        )
        fig = scalar_fig.plot_deltascalar1D()
        name = prefix + f"{var_name}_transport_{epoch}"
        if outfolder is not None:
            misc.save_fig(
                fig, name, outfolder + "/gradient_of_transport/", writer, epoch
            )
        fig.clf()

        fig = scalar_fig.plot_scalar1D()
        name = prefix + f"source_vs_transport_{epoch}"
        misc.save_fig(fig, name, outfolder + "/source_vs_transport/", writer, epoch)
        fig.clf()

        fig = scalar_fig.plot_potential()
        name = prefix + f"g_potential_{epoch}"
        misc.save_fig(fig, name, outfolder + "/potential/", writer, epoch)
        fig.clf()
        if "prob" in datatype.lower():
            pssim = trans.probsfromlogits(logitpssim)
            psdata = trans.probsfromlogits(logitpsdata)
            pscal = trans.probsfromlogits(logitpscal)

            name = f"{prefix}_proba_{epoch}_"

            fig, (ax_1, ax_2) = plt.subplots(
                2,
                1,
                gridspec_kw={"height_ratios": [3, 1]},
                figsize=(9, 5),
                sharex="col",
            )

            dist_styles = [style_target.copy(), style_source.copy(), style_trans.copy()]

            count_data, ax_1 = plot_hist(
                psdata,
                pssim,
                pscal,
                dist_styles=dist_styles,
                mask_sig=[
                    np.ones_like(np.ravel(psdata)),
                    np.ones_like(np.ravel(pssim)),
                    np.ones_like(np.ravel(pscal)),
                ],
                legend_kwargs={"title": legend_title},
                ax=ax_1,
                style={"bins": np.linspace(0, 1, 15)},
            )
            dist_styles[1].pop("drawstyle", None)
            dist_styles[2].pop("drawstyle", None)
            ax_2 = plot_ratio(
                counts=count_data,
                ax=ax_2,
                truth_key="dist_0",
                styles=dist_styles,
                ylim=[0.5, 1.5],
            )
            ax_2.tick_params(axis="both", which="major", labelsize=LABELSIZE)
            ax_1.set_ylabel("Normalised entries", fontsize=FONTSIZE)
            ax_2.set_xlabel("probability", fontsize=FONTSIZE)
            # ax_1.set_yscale("log")
            fig.tight_layout()
            misc.save_fig(fig, name, outfolder + "/hist_probs/", writer, epoch)
            fig.clf()
        plt.close("all")

    elif logitpsdata.shape[1] == 2:
        xmin, xmax = (
            np.min([logitpssim[:, 0], logitpsdata[:, 0]]),
            np.max([logitpssim[:, 0], logitpsdata[:, 0]]),
        )
        ymin, ymax = (
            np.min([logitpssim[:, 1], logitpsdata[:, 1]]),
            np.max([logitpssim[:, 1], logitpsdata[:, 1]]),
        )
        hist2d_style = {"bins": 50, "range": [[xmin, xmax], [ymin, ymax]]}
        fig, ax_1 = plot_2dhist(
            data=logitpssim, hist2d_style=hist2d_style, style3d={"color": "blue"}
        )
        fig, ax_1 = plot_2dhist(
            data=logitpscal,
            hist2d_style=hist2d_style,
            ax_figure=ax_1,
            fig=fig,
            style3d={"linestyles": "dotted", "color": "black"},
        )
        fig, ax_1 = plot_2dhist(
            data=logitpsdata,
            hist2d_style=hist2d_style,
            style3d={"color": "red"},
            ax_figure=ax_1,
            fig=fig,
        )
        # sys.exit()
        name = f"{prefix}_2d_histogram_{epoch}_"
        with open(outfolder + f"/{name}.fig.pickle", "wb") as pickle_fig:
            pickle.dump(fig, pickle_fig)
        plt.close("all")  # stupid to run close all but could not use fig.close

    else:
        if ("prob" in datatype.lower()) and (logitpsdata.shape[1] == 3):
            pssim = trans.probsfromlogits(logitpssim)
            pscal = trans.probsfromlogits(logitpscal)
            psdata = trans.probsfromlogits(logitpsdata)
        # higher order correlation plots
        if kwargs.get("correlation_bool", True):
            corr_path = f"{outfolder}/correlation_plots/"
            if not os.path.exists(corr_path):
                os.mkdir(corr_path)
            kwargs = {"y_lim": [0, 1], "x_lim": [0, 1], "n_sample": 10_000}
            print("Plotting correlations")
            for name, sample in zip(
                [style_trans["label"], style_source["label"]], [pscal, pssim]
            ):
                corr_name = f"{prefix}_correlation_plot_{name}.png"
                plot_feature_spread(
                    psdata,
                    sample,
                    save_dir=corr_path + corr_name,
                    feature_nms=[r"$p_b$", r"$p_c$", r"$p_u$"],
                    labels=[style_target["label"], name],
                    **kwargs,
                )

        # plot the conditional distributions
        for ifeat, featname in enumerate(nonconvex_inputs):
            name = prefix + f"hist_{featname}"
            fig, ax_1 = plt.subplots(1, 1, figsize=FIG_SIZE)
            _, ax_1 = plot_hist(
                condssim[:, ifeat],
                condsdata[:, ifeat],
                dist_styles=[style_source, style_target],
                ax=ax_1,
                mask_sig=[sig_sim, np.ones_like(condsdata[:, ifeat])],
                title=legend_title,
            )
            ax_1.tick_params(axis="both", which="major", labelsize=LABELSIZE)
            ax_1.set_ylabel("#", fontsize=FONTSIZE)
            ax_1.set_xlabel(rf"${featname}$", fontsize=FONTSIZE)
            misc.save_fig(fig, name, outfolder + "/conditional_dist/", writer, epoch)
            fig.clf()
            plt.close("all")

        for ifeat, featname in enumerate(convex_inputs):
            histname = prefix + f"hist_logit_{featname}"

            fig, (ax_1, ax_2) = plt.subplots(
                2,
                1,
                gridspec_kw={"height_ratios": [3, 1]},
                figsize=(9, 5),
                sharex="col",
            )
            dist_styles = [style_target.copy(), style_source.copy(), style_trans.copy()]
            count_data, ax_1 = plot_hist(
                logitpsdata[:, ifeat],
                logitpssim[:, ifeat],
                logitpscal[:, ifeat],
                dist_styles=dist_styles,
                mask_sig=[
                    np.ones_like(logitpsdata[:, ifeat]),
                    sig_sim,
                    np.ones_like(logitpsdata[:, ifeat]),
                ],
                legend_kwargs={"title": legend_title},
                ax=ax_1,
                uncertainties=[None, None, uncertainty],
            )
            dist_styles[1].pop("drawstyle", None)
            dist_styles[2].pop("drawstyle", None)
            ax_2 = plot_ratio(
                counts=count_data,
                ax=ax_2,
                truth_key="dist_0",
                styles=dist_styles,
                ylim=ylim_ratio,
            )
            ax_2.tick_params(axis="both", which="major", labelsize=LABELSIZE)
            ax_1.set_ylabel("Normalised entries", fontsize=FONTSIZE)
            ax_2.set_xlabel(rf"${featname}$", fontsize=FONTSIZE)

            fig.tight_layout()
            misc.save_fig(fig, histname, outfolder + "/hist_logit/", writer, epoch)
            fig.clf()

            if ("prob" in datatype.lower()) and (logitpsdata.shape[1] == 3):
                proba_name = (
                    r"$p_b$"
                    if "p_b" in featname
                    else (
                        r"$p_c$"
                        if "p_c" in featname
                        else r"$p_u$" if "p_u" in featname else ""
                    )
                )
                histname = histname.replace("logit", proba_name)

                fig, (ax_1, ax_2) = plt.subplots(
                    2,
                    1,
                    gridspec_kw={"height_ratios": [3, 1]},
                    figsize=(9, 5),
                    sharex="col",
                )
                dist_styles = [
                    style_target.copy(),
                    style_source.copy(),
                    style_trans.copy(),
                ]
                count_data, ax_1 = plot_hist(
                    psdata[:, ifeat],
                    pssim[:, ifeat],
                    pscal[:, ifeat],
                    dist_styles=dist_styles,
                    mask_sig=[
                        np.ones_like(psdata[:, ifeat]),
                        sig_sim,
                        np.ones_like(psdata[:, ifeat]),
                    ],
                    legend_kwargs={"title": legend_title},
                    ax=ax_1,
                    style={"bins": np.linspace(0, 1, 15)},
                )
                dist_styles[1].pop("drawstyle", None)
                dist_styles[2].pop("drawstyle", None)
                ax_2 = plot_ratio(
                    counts=count_data,
                    ax=ax_2,
                    truth_key="dist_0",
                    styles=dist_styles,
                    ylim=ylim_ratio,
                )
                ax_2.tick_params(axis="both", which="major", labelsize=LABELSIZE)
                ax_1.set_ylabel("Normalised entries", fontsize=FONTSIZE)
                ax_2.set_xlabel(proba_name, fontsize=FONTSIZE)
                ax_1.set_yscale("log")
                fig.tight_layout()
                misc.save_fig(fig, histname, outfolder + "/hist_probs/", writer, epoch)
                fig.clf()

        if ("cern" in datatype.lower()) and (logitpsdata.shape[1] == 3):
            for featname, low, up in zip(
                ["pc/pb", "pu/pb", "pu/pc"], [0, 0, 1], [1, 2, 2]
            ):
                fig, (ax_1, ax_2) = plt.subplots(
                    2,
                    1,
                    gridspec_kw={"height_ratios": [3, 1]},
                    figsize=(9, 5),
                    sharex="col",
                )
                dist_styles = [
                    style_target.copy(),
                    style_source.copy(),
                    style_trans.copy(),
                ]

                count_data, ax_1 = plot_hist(
                    psdata[:, up] / psdata[:, low],
                    pssim[:, up] / pssim[:, low],
                    pscal[:, up] / pscal[:, low],
                    dist_styles=dist_styles,
                    mask_sig=[
                        np.ones_like(psdata[:, up]),
                        sig_sim,
                        np.ones_like(pssim[:, up]),
                    ],
                    legend_kwargs={"title": legend_title},
                    ax=ax_1,
                    style={"bins": 30, "histtype": "step", "range": [1, 25]},
                    normalise=False,
                )
                dist_styles[1].pop("drawstyle", None)
                dist_styles[2].pop("drawstyle", None)
                ax_2 = plot_ratio(
                    counts=count_data,
                    ax=ax_2,
                    truth_key="dist_0",
                    styles=dist_styles,
                    ylim=ylim_ratio,
                )
                ax_2.tick_params(axis="both", which="major", labelsize=LABELSIZE)
                ax_1.set_ylabel("Normalised entries", fontsize=FONTSIZE)
                ax_2.set_xlabel(featname, fontsize=FONTSIZE)
                ax_1.set_yscale("log")

                fig.tight_layout()
                misc.save_fig(
                    fig,
                    histname + featname.replace("/", "_"),
                    outfolder + "/logit_ratio/",
                    writer,
                    epoch,
                )
                fig.clf()

            for jfeat in range(ifeat + 1, len(convex_inputs)):
                jfeatname = convex_inputs[jfeat]

                fig = plot_arrows(
                    g_func,
                    avgconds.to(device),
                    avgfeats.to(device),
                    xidx=ifeat,
                    yidx=jfeat,
                    xlabel=featname,
                    ylabel=jfeatname,
                )

                name = prefix + f"stream_{featname}_{jfeatname}.pdf"
                misc.save_fig(fig, name, outfolder + "/gradient_arrows/", writer, epoch)

                fig.clf()
        plt.close("all")  # stupid to run close all but could not use fig.close

        if ("cern" in datatype.lower()) and (logitpsdata.shape[1] == 3):
            dl1rsim = trans.dl1r(pssim)
            dl1rcal = trans.dl1r(pscal)
            dl1rdata = trans.dl1r(psdata)
            try:
                name = prefix + "eff_dl1r"
                fig, mae_trans, mae_source = plot_eff(
                    dl1rdata,
                    dl1rsim,
                    dl1rcal,
                    xlabel="DL1r",
                    legend_title=legend_title,
                    style_source=style_source,
                    style_trans=style_trans,
                    style_target=style_target,
                )

                misc.save_fig(fig, name, outfolder + "/eff_dl1r/", writer, epoch)
                fig.clf()
            except ValueError:
                print("need to fix size difference")

            name = f"{epoch}"
            # fig = plot_hist(dl1rdata, dl1rsim, dl1rcal, sig_source=sig_sim,xlabel="DL1r",
            #                     binrange=[-15, 0.6650, 2.1950, 3.2450, 4.5650, 15],
            #                     legend_title=legend_title)
            names = ["dl1r_working_points", "hist_dl1r"]
            for nr, bins in enumerate(
                [
                    [-15, 0.6650, 2.1950, 3.2450, 4.5650, 15],
                    # np.linspace(-5,15,31)]
                    np.linspace(-4, 12.5, 16),
                ]
            ):
                fig, (ax_1, ax_2) = plt.subplots(
                    2,
                    1,
                    gridspec_kw={"height_ratios": [3, 1]},
                    figsize=(9, 5),
                    sharex="col",
                )
                dist_styles = [
                    style_target.copy(),
                    style_source.copy(),
                    style_trans.copy(),
                ]
                count_data, ax_1 = plot_hist(
                    dl1rdata,
                    dl1rsim,
                    dl1rcal,
                    dist_styles=dist_styles,
                    mask_sig=[np.ones_like(dl1rdata), sig_sim, np.ones_like(dl1rdata)],
                    legend_kwargs={"title": legend_title},
                    ax=ax_1,
                    style={"bins": bins},
                )
                dist_styles[1].pop("drawstyle", None)
                dist_styles[2].pop("drawstyle", None)
                ax_2 = plot_ratio(
                    counts=count_data,
                    ax=ax_2,
                    truth_key="dist_0",
                    styles=dist_styles,
                    ylim=ylim_ratio,
                )
                ax_2.tick_params(axis="both", which="major", labelsize=LABELSIZE)
                ax_1.set_ylabel("Normalised entries", fontsize=FONTSIZE)
                ax_2.set_xlabel("DL1r", fontsize=FONTSIZE)

                fig.tight_layout()
                misc.save_fig(
                    fig,
                    f"{prefix}{names[nr]}_{epoch}",
                    f"{outfolder}/{names[nr]}/",
                    writer,
                    epoch,
                )
                fig.clf()

            # fig = plot_hist(dl1rdata, dl1rsim, dl1rcal,xlabel="DL1r",
            #                 sig_source=sig_sim,
            #                 legend_title=legend_title)#, binrange=(-10, 20))

            # misc.save_fig(fig, histname, outfolder + "/hist_dl1r/", writer, epoch)
            # fig.clf()
            # plt.close("all")

    # return mae_trans, mae_source, mae_trans_list, mae_source_list


def eqprob(xs_value: int) -> np.ndarray:
    """get array that sum to 1

    Parameters
    ----------
    xs_value : int
        size of the probability vector

    Returns
    -------
    np.ndarray
        vector that sum to 1
    """
    return np.ones_like(xs_value) / xs_value.shape[0]


def add_lower_plot(
    ax_figure: plt.Figure,
    target: np.ndarray,
    trans: np.ndarray,
    source: np.ndarray,
    x_axis_values: list,
    **kwargs: dict,
) -> plt.Figure:
    """Add residual to figure ax

    Parameters
    ----------
    ax_figure : plt.Figure
        figure to plot the residual on
    target : torch.Tensor
        target distribution
    trans : torch.Tensor
        transported source distribution
    source : torch.Tensor
        source distributin
    x_axis_values : torch.Tensor
        _description_
    kwargs : dict
        additional inputs

    Returns
    -------
    plt.Figure
        _description_
    """
    style_t = style_target.copy()
    style_t.pop("marker")
    style_t.update({"linestyle": "solid"})
    for nr, (value, style) in enumerate(
        zip([target, trans, source], [style_t, style_trans, style_source])
    ):
        ax_figure.plot(
            x_axis_values[nr],
            np.abs(target - value) if kwargs["residual"] else value,
            linewidth=2,
            **style,
        )
    if kwargs["log"]:
        ax_figure.set_yscale("log")
    return ax_figure


def add_ratio(
    ax_figure: plt.Figure,
    target: np.ndarray,
    trans: np.ndarray,
    source: np.ndarray,
    x_axis_values: list,
    xerr: np.ndarray,
    y_err: bool = True,
    line: bool = False,
    **kwargs: dict,
) -> plt.Figure:
    """Add residual to figure ax

    Parameters
    ----------
    ax_figure : plt.Figure
        figure to plot the residual on
    target : torch.Tensor
        target distribution
    trans : torch.Tensor
        transported source distribution
    source : torch.Tensor
        source distributin
    x_axis_values : torch.Tensor
        _description_
    kwargs : dict
        additional inputs

    Returns
    -------
    plt.Figure
        _description_"""
    if y_err:
        y_error = np.sqrt(
            (1 / target * np.sqrt(trans)) ** 2
            + (trans / target**2 * np.sqrt(target)) ** 2
        )
        y_error_original = np.sqrt(
            (1 / target * np.sqrt(source)) ** 2
            + (source / target**2 * np.sqrt(target)) ** 2
        )

    x_values = x_axis_values if len(x_axis_values) else [target, target, target]

    ax_figure.errorbar(
        x_values[0],
        trans / target,
        yerr=y_error if y_err else None,
        xerr=np.abs(xerr),
        label=style_trans["label"],
        color=style_trans["color"],
        marker="o" if not line else None,
        linestyle="none",
    )
    ax_figure.errorbar(
        x_values[1],
        source / target,
        yerr=y_error_original if y_err else None,
        xerr=np.abs(xerr),
        label=style_source["label"],
        color=style_source["color"],
        marker="o" if not line else None,
        linestyle="none",
    )

    if kwargs["log"]:
        ax_figure.set_yscale("log")

    ax_figure.set_ylabel("MC/data", fontsize=FONTSIZE)
    return ax_figure


# def eff_error(proba, sample_size):
#     return np.sqrt( proba * (1-proba) / sample_size ) / proba

# def eff_error(proba, sample_size):
#     value = sample_size*proba
#     return np.sqrt( value * (1-value/sample_size) ) / sample_size


def plot_eff(
    target: np.ndarray,
    source: np.ndarray,
    trans: np.ndarray,
    xlabel: str = "",
    legend_title: str = "",
    style_source={},
    style_trans={},
    style_target={},
) -> plt.Figure:
    """plot the efficency

    Parameters
    ----------
    target : np.ndarray
        target distribution
    source : np.ndarray
        source distribution
    trans : np.ndarray
        transport source distribution
    xlabel : str, optional
        x axis label, by default ""
    title : str, optional
        title on figure, by default ""

    Returns
    -------
    plt.Figure
        output Figure
    """

    plotrange = np.percentile(target, 0.1), np.percentile(target, 99.9)

    target = np.flip(np.sort(target))
    source = np.flip(np.sort(source))
    trans = np.flip(np.sort(trans))

    fig, (ax_1, ax_2, ax_3) = plt.subplots(
        3, 1, gridspec_kw={"height_ratios": [3, 1, 1]}, figsize=FIG_SIZE, sharex=True
    )

    style_t = style_target.copy()
    style_s = style_source.copy()
    style_tt = style_trans.copy()
    style_t.update({"linestyle": "solid"})

    ax_1.plot(
        target,
        np.cumsum(eqprob(target)),
        # linewidth=2,
        **style_t,
    )
    ax_1.plot(
        source,
        np.cumsum(eqprob(source)),
        # linewidth=2,
        **style_s,
    )
    ax_1.plot(
        trans,
        np.cumsum(eqprob(trans)),
        # linewidth=2,
        **style_tt,
    )

    style = {"bins": [np.percentile(target, i) for i in range(0, 100)]}
    counts_trans, bins = np.histogram(trans, **style)
    counts_source, _ = np.histogram(source, **style)
    counts_target, _ = np.histogram(target, **style)

    intergral_source = (np.cumsum(counts_source[::-1])[::-1] / counts_source.sum()) / (
        np.cumsum(counts_target[::-1])[::-1] / counts_target.sum()
    )
    intergral_trans = (np.cumsum(counts_trans[::-1])[::-1] / counts_trans.sum()) / (
        np.cumsum(counts_target[::-1])[::-1] / counts_target.sum()
    )
    # xs_value = (bins[:-1]+bins[1:]) / 2
    # xerr = (bins[:-1]-bins[1:])/2

    if any(np.isnan(intergral_trans)) | any(np.isnan(intergral_source)):
        raise ValueError("Problem with NaN")

    style_source_diff = {"linestyle": "dotted", "color": "blue"}
    style_trans_diff = {"linestyle": "dashed", "color": "red"}

    # Efficiency matching
    from scipy.interpolate import UnivariateSpline, interpolate

    target_spline = interpolate.interp1d(target, np.cumsum(eqprob(target)))
    source_spline = interpolate.interp1d(source, np.cumsum(eqprob(source)))
    trans_spline = interpolate.interp1d(trans, np.cumsum(eqprob(trans)))
    x = np.linspace(*plotrange, 101)
    ax_2.plot(x, np.ones_like(x), "k--")
    ax_2.plot(x, trans_spline(x) / target_spline(x), **style_trans_diff)
    ax_2.plot(x, source_spline(x) / target_spline(x), **style_source_diff)
    ax_2.set_ylabel("Ratios", fontsize=FONTSIZE)
    ax_2.set_ylim([0.9, 1.1])

    ## quantile matching
    ax_3.plot(target, np.zeros_like(target), "k--")
    ax_3.plot(target, target - trans, **style_trans_diff)
    ax_3.plot(target, target - source, **style_source_diff)
    ax_3.set_ylabel("Quantile match", fontsize=FONTSIZE)
    ax_3.set_ylim([-0.5, 0.5])

    mae_trans, mae_source = np.mean(np.abs(trans - target)), np.mean(
        np.abs(source - target)
    )

    ax_3.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax_3.set_xlim(*plotrange)

    ax_1.set_xlim(*plotrange)
    ax_1.set_ylim(0, 1)
    ax_1.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax_1.set_ylabel("Efficiency", fontsize=FONTSIZE)
    ax_1.legend(loc="best", title=legend_title, prop={"size": FONTSIZE}, frameon=False)

    fig.tight_layout()
    return fig, mae_trans, mae_source


def plot_2dhist(
    data: np.ndarray,
    style3d: dict = None,
    hist2d_style: dict = None,
    ax_figure=None,
    fig=None,
) -> plt.Figure:
    """
    This function will plot 2d histograms to visualize 2d histogram transport.

    Parameters
    ----------
    data : np.ndarray
        2d numpy array of the histogram
    style3d : dict, optional
        Style of the 3d plot of the histogram. The default is None.
    hist2d_style : dict, optional
        Style of the 2dhistogram. The default is None.
    ax_figure : plt.figure, optional
        if you need to plot the figure on a plt.Figure
        The default is None.
    fig : plt.figure, optional
        if you need to plot the figure on a plt.Figure
        The default is None.

    Returns
    -------
    ax_figure :
        Axes3DSub
    fig : plt.figure
        a figure

    """
    if ax_figure is None:
        fig = plt.figure()
        ax_figure = fig.add_subplot(projection="3d")
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], **hist2d_style)

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])

    ax_figure.plot_wireframe(xpos, ypos, hist, rstride=2, cstride=2, **style3d)

    return fig, ax_figure


def true_transport(g_func: callable, source, conds, target):
    # plot the transport vs prediction
    # if source.is_cuda:
    #     device = f"{source.device.type}:{source.device.index}"
    # else:
    #     device = "cpu"
    # xval = torch.Tensor(np.mgrid[-1:0:100j]).unsqueeze(1).to(device)
    # xval.requires_grad = True
    target_sort, _ = target.sort(axis=0)
    source_sort, source_argsort = source.sort(axis=0)

    yvalnom = utils.trans(g_func, source_sort, conds[source_argsort][:, :, 0])

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)

    ax.plot(
        utils.detach(source_sort),
        utils.detach(yvalnom),
        color="red",
        lw=2,
        label="derived transport",
    )
    ax.plot(
        utils.detach(source_sort),
        utils.detach(target_sort),
        "--",
        color="red",
        lw=2,
        label="true optimal transport",
    )

    # ax.set_xlim(-1, 0)
    # ax.set_ylim(-0.5, 0.5)
    ax.tick_params(axis="both", which="major", labelsize=LABELSIZE)

    ax.set_xlabel("$x$", fontsize=FONTSIZE)
    ax.set_ylabel("$x' = T(x)$", fontsize=FONTSIZE)

    fig.legend(loc=(0.4, 0.2), prop={"size": LEGENDSIZE}, frameon=False)
    fig.tight_layout()
    return fig


def plot_training_setup(
    source_values: iter,
    target_values: iter,
    outdir: str,
    eval_data: dict = None,
    generator=None,
    **kwargs,
):
    os.makedirs(f"{outdir}", exist_ok=True)
    percentile_lst = kwargs.get('percentile_lst', [0.01, 99.99])

    # get iterator data for training
    if not isinstance(source_values, torch.Tensor):
        source_values = source_values.data[:1_000_000]
    if not isinstance(target_values, torch.Tensor):
        target_values = target_values.data[:1_000_000]

    # get transport if generator defined
    mask_sig_source = source_values[:, -1] == 1
    mask_sig_target = target_values[:, -1] == 1
    if generator is not None:
        cvx_dim = generator.cvx_dim
        noncvx_dim = generator.noncvx_dim
        transport = generator.chunk_transport(
            source_values[:, noncvx_dim : noncvx_dim + cvx_dim],
            source_values[:, :noncvx_dim],
            sig_mask=mask_sig_source,
            n_chunks=len(source_values)//4096,
        )

        transport = torch.concat(
            [
                source_values[:, :noncvx_dim].cpu(),
                transport,
                source_values[:, -1:].cpu(),
            ],
            1,
        )

        transport = transport.cpu().detach().numpy()

    mask_sig_source = mask_sig_source.cpu().detach().numpy()
    mask_sig_target = mask_sig_target.cpu().detach().numpy()
    source_values = source_values.cpu().detach().numpy()
    target_values = target_values.cpu().detach().numpy()
    
    plot_kwargs = kwargs.get("plot_kwargs", {})
    # training iterator data
    style = {
        "bins": kwargs.get("n_bins", 40),
        # "histtype": "step",
        # "density": False,
        # "stacked": True,
    }
    plot_var = kwargs.get("plot_var", ["pT", "pb", "pc", "pu"])
    
    if target_values.shape[-1]-1!=len(plot_var):
        raise ValueError("plot_var size is not same as columns in target. variable names missing!")
    
    dist_labels = kwargs.get("dist_labels", ["Source", "Target", "Transport"])
    dist_styles = [
        {"label":i, "color": color}
        for i, color in zip(dist_labels,["blue", "black", "red", "green", "orange"])
        ]

    train_ranges = []

    # plotning large dimensions can be difficult
    large_ot_dims = len(plot_var)>32
    
    if large_ot_dims:
        # if large dims only plot the ratio in a grid formation
        n_cols = int(np.sqrt(len(plot_var)))
        fig, ax_ratio = plt.subplots(
            n_cols+1, n_cols, squeeze=False, figsize=(8*n_cols, 3*n_cols),
        )
        ax_ratio = np.ravel(ax_ratio)
    else:
        # if not large dims plot single row of histograms and ratios
        fig, (ax_hist, ax_ratio) = plt.subplots(
            2, len(plot_var), gridspec_kw={"height_ratios": [3, 1]},
            figsize=(8*len(plot_var), 6), sharex="col"
        )


    for nr, i in enumerate(plot_var):
        x_range = np.percentile(target_values[:, nr], percentile_lst)
        train_ranges.append(x_range)
        style["range"] = x_range
        mask_sig=[mask_sig_source, mask_sig_target]
        dist_args=[source_values[:,nr], target_values[:,nr]]
        if generator is not None:
            mask_sig.append(mask_sig_source)
            dist_args.append(transport[:,nr])
        counts, _ = plot_hist(*dist_args, mask_sig=mask_sig, dist_styles=dist_styles,
                              ax=ax_hist[nr] if not large_ot_dims else None,
                              style=style, plot_bool=not large_ot_dims,
                              **plot_kwargs)
        plot_ratio(counts=counts, truth_key="dist_1", #TODO should be changed to first or last key
                              ax=ax_ratio[nr], style=style,
                              styles=dist_styles,
                              ylim=[0.75, 1.25])
        if large_ot_dims:
            ax_ratio[nr].set_xlabel(i)
        else:
            ax_hist[nr].set_yscale("log")
            ax_ratio[nr].set_xlabel(i)
    
    # cannot do tight_layout with very large figures
    plt.tight_layout()

    os.makedirs(f"{outdir}/training_sample/", exist_ok=True)
    plt.savefig(f"{outdir}/training_sample/training_{kwargs.get('epoch', '')}.pdf")
    plt.close(fig)

    if "ftag" in kwargs.get("datatype", "").casefold():  # plot dl1r not sure its working
        mask_sig=[mask_sig_source, mask_sig_target] # TODO should be made standalone in ftag-repo
        dl1r_source = trans.dl1r(trans.probsfromlogits(source_values[:, 1:4]))
        dl1r_target = trans.dl1r(trans.probsfromlogits(target_values[:, 1:4]))
        dist_args=[dl1r_source, dl1r_target]
        style = {
            "bins": kwargs.get("n_bins", 40),
            "range": np.percentile(dl1r_target, percentile_lst),
        }
        if generator is not None:
            dl1r_transport = trans.dl1r(trans.probsfromlogits(transport[:, 1:4]))
            mask_sig.append(mask_sig_source)
            dist_args.append(dl1r_transport)

        x_range = np.percentile(dl1r_target, percentile_lst)
        style["range"] = x_range

        fig, (ax_hist, ax_ratio) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]},
                                                figsize=(8, 6), sharex="col"
                                                )

        counts, _ = plot_hist(*dist_args, mask_sig=mask_sig, dist_styles=dist_styles,
                              ax=ax_hist, style=style,
                              **plot_kwargs)
        plot_ratio(counts=counts, truth_key="dist_1", #TODO should be changed to first or last key
                              ax=ax_ratio, style=style,
                              styles = dist_styles,ylim=[0.75, 1.25])

        ax_ratio.set_xlabel("DL1r")
        ax_hist.set_yscale("log")
        plt.tight_layout()
        plt.savefig(f"{outdir}/training_sample/training_dl1r_{kwargs.get('epoch', '')}.png")
        plt.close(fig)
    
    # plot the eval sample
    # TODO not compatible with large dimensions yet
    if isinstance(eval_data, dict) and not large_ot_dims:
        for name, values in eval_data.items():
            nr = 0
            for keys_to_plot in ["conds", "transport"]:
                eval_path = f"{outdir}/eval_sample/{keys_to_plot}"
                os.makedirs(eval_path, exist_ok=True)
                shape = values[list(values.keys())[0]][keys_to_plot].shape[1]
                if shape==0:
                    continue
                fig, (ax_hist, ax_ratio) = plt.subplots(
                    2, shape, gridspec_kw={"height_ratios": [3, 1]},
                    figsize=(8*shape, 6), sharex="col", squeeze=False
                )

                colors={i:j for i,j in  zip(values.keys(), COLORS)}

                for col in range(shape):
                    style = {
                        "bins": kwargs.get("n_bins", 40),
                        "range": train_ranges[nr]
                    }
                    mask_sig=[]
                    dist_args=[]
                    dist_styles=[]
                    
                    for sub_keys in values.keys():
                        dist_args.append(values[sub_keys][keys_to_plot][:, col].detach().numpy())
                        mask_sig.append(values[sub_keys]["sig_mask"].detach().numpy())
                        dist_styles.append({"label": sub_keys, "color": colors[sub_keys]})

                        if len(dist_args[-1])==0:
                            continue
                        
                        if "truth" in sub_keys:
                            style["range"] =  np.percentile(dist_args[-1], percentile_lst)

                        if ("eval_transport" in values[sub_keys]) & (keys_to_plot!="conds"):
                            dist_args.append(values[sub_keys]["eval_transport"][:, col].detach().numpy())
                            mask_sig.append(values[sub_keys]["sig_mask"].detach().numpy())
                            dist_styles.append({"label": f"eval_transport_{sub_keys}", "color": colors[sub_keys],
                                                "alpha":0.5})

                    if len(dist_args[-1])==0:
                        continue
                    counts, _ = plot_hist(*dist_args, mask_sig=mask_sig, dist_styles=dist_styles,
                                            ax=ax_hist[col], style=style,
                                            **plot_kwargs)
                    plot_ratio(counts=counts, truth_key="dist_0", #TODO should be changed to first or last key
                                            ax=ax_ratio[col], style=style,
                                            styles=dist_styles,
                                            ylim=[0.75, 1.25])

                    ax_ratio[col].set_xlabel(plot_var[col])
                    ax_hist[col].set_yscale("log")
                    nr += 1
                plt.tight_layout()

                plt.savefig(f"{eval_path}/eval_{name}_{kwargs.get('epoch', '')}.png")
                plt.close(fig)
                
def get_sys_unc_on_eff(eff_values: np.ndarray, eff_sys_values:dict,
                       average_bool=True) -> None:
    # get name of all systematics  
    unique_sys_labels = np.unique([i.replace('_down', '').replace('_up', '') for i in eff_sys_values.keys()])
       
    # for loop over all systematics
    all_eff_sys = {}
    for label in unique_sys_labels:

        if 'bins' in label or 'nominal' in label:
            continue

        # get up/down labels for that systematics
        sys_labels = [i for i in eff_sys_values.keys() if label in i]
        eff_lst = []
        for sys_label in sys_labels:
            # subtract the systematics bkg contribution
            eff_val = (((eff_values-eff_sys_values[sys_label][:, -1]))/
                                (1-eff_sys_values[sys_label][:, -2]))
            if average_bool:
                eff_lst.append(eff_val)
            else:
                all_eff_sys[sys_label] = eff_val

        # take the center of up/down
        if average_bool:
            all_eff_sys[label] = np.mean(eff_lst, 0)

    return all_eff_sys    
    
    

def efficiency_plots(*distributions, conditions, conds_bins_lst:list,
                     eff_cut=7, weights=None,
                     mask_from_eff_cut = None, **kwargs):
    """create eff plot - similar to https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2020-009/fig_03a.pdf

    Missing correct uncertainties

    the first value of labels used as truth

    you can also just have calculated eff_dict
    
    todo REMOVE THE PLOTTING PART AND MAKE IT A SEPARATE FUNCTION
    """
    
    if isinstance(eff_cut, dict) and (mask_from_eff_cut is None):
        raise ValueError("mask_from_eff_cut has to be defined beforehand if eff_cut is a dict")
        

    if len(conditions) != len(distributions):
        raise ValueError(
            "conditions length and distributions length has to be the same"
        )
    conditions = [np.ravel(i) for i in conditions]
    # if conds_bins_lst is None:
    #     conds_bins_lst = np.percentile(truth_conds[0], np.arange(0, 110, 10))
    #     conds_bins_lst[-1] = 400
    labels = kwargs.get("labels", ["Transport", "Truth", "Source"])
    sig_labels = kwargs.get("sig_labels")
    calculate_sf = kwargs.get("calculate_sf", False)
    plot_individual_points = kwargs.get("plot_individual_points")

    if calculate_sf:
        conds_bins_lst = [default_pt,default_pt, default_pt]+[default_pt]*len(distributions[3:])
    elif (len(distributions)==3): # if too many bins we assume we sample uniform with flow. For data (first element) we have less stat so less bins
        conds_bins_lst = [conds_bins_lst, default_pt, conds_bins_lst]+[conds_bins_lst]*len(distributions[3:])
    else: # TODO redo stupid implementation
        conds_bins_lst = [conds_bins_lst,conds_bins_lst,conds_bins_lst]+[conds_bins_lst]*len(distributions[3:])

    if kwargs.get("eff_dict") is None:
        eff_dict_stat_unc = {i: [] for i in labels}
        eff_dict = {i: [] for i in labels}
        eff_dict_sum_w = {i: [] for i in labels}
        for nr, (dist, conds, label, conds_bins) in enumerate(zip(distributions, conditions, labels, conds_bins_lst)):
            for low, high in zip(conds_bins[:-1], conds_bins[1:]):
                mask_pt = (conds >= low) & (conds < high)
                dist_pt_bin = dist[mask_pt]
                weight_pt_bin = weights[nr][mask_pt] if weights is not None else np.ones(np.sum(mask_pt))

                if isinstance(eff_cut, list):
                    mask_cut_truth = np.ravel((dist_pt_bin >= eff_cut[0]) & (
                        dist_pt_bin < eff_cut[1]
                    ))

                if isinstance(eff_cut, dict):
                    mask_cut_truth = mask_from_eff_cut[nr][mask_pt]
                else:
                    mask_cut_truth = np.ravel(dist_pt_bin >= eff_cut)
                
                normed_weights = weight_pt_bin/weight_pt_bin.sum()
                
                # ratio that pass the working point
                pass_ratio = normed_weights[mask_cut_truth].sum()
                
                unc = np.sqrt(pass_ratio*(1-pass_ratio)
                              *(normed_weights**2).sum())

                eff_dict_sum_w[label].append(normed_weights)
                # eff only signal
                if (sig_labels is not None):
                    eff_bkg_pass = normed_weights[(np.ravel(mask_cut_truth) 
                                                   & ~np.ravel(sig_labels[nr][mask_pt]))
                                                  ].sum()
                    eff_bkg = normed_weights[~np.ravel(sig_labels[nr][mask_pt])].sum()

                    eff_dict[label].append([pass_ratio, eff_bkg, eff_bkg_pass])

                    # eff_i*(1-eff_i)/N
                    eff_dict_stat_unc[label].append(
                        [unc, 
                            np.sqrt(eff_bkg*(1-eff_bkg)*(normed_weights**2).sum()),

                            np.sqrt(eff_bkg_pass*(1-eff_bkg_pass)*(normed_weights**2).sum()),
                         ]
                        )
                else:
                    eff_dict[label].append([pass_ratio])
                    eff_dict_stat_unc[label].append(unc)
        eff_dict = {key: np.array(val) for key, val in eff_dict.items()}
        eff_dict_stat_unc = {key: np.array(val) for key, val in eff_dict_stat_unc.items()}
            
        total_unc_lst = kwargs.get("total_unc")
        if kwargs.get('return_raw', False):
            return eff_dict

        if calculate_sf and kwargs.get('bkg_subtraction', True):
            keys = list(eff_dict.keys())
        
            # loading the systematics bkg contribution
            eff_sys=None
            
            if (kwargs.get('systematics_uncertainty_on_data', True) 
            and (kwargs.get('legend_title') is not None)):
                # get the systematics uncertainty on the data
                # this is calculated in bkg_eff_from_systematics.py
                selection_name = misc.replace_symbols(kwargs.get('legend_title').casefold())
                selection_name = 'dl1rb' if 'd_b' in kwargs.get('legend_title').casefold() else selection_name
                
                sample_name = 'flow' if 'prime' in kwargs['labels'][0] else 'mc'
                
                if isinstance(eff_cut, float):
                    selection_name = f"{eff_cut}*{selection_name}"
                possible_path = glob(f"/home/users/a/algren/work/run_ot/outfiles/{sample_name}/bkg_eff_for_*{selection_name}*") # TODO should be dynamic

                if len(possible_path)==1:
                    eff_sys = misc.load_h5(possible_path[0])

                total_unc_lst[0] = None # remove the systematics variation on the transport

                if len(possible_path)>1:
                    raise ValueError("Multiple files found -  i should really have this as an input to the function")
            
            # eff_dict_stat_unc[keys[nr]]

            for nr in range(len(keys)):
                eff_dict_stat_unc[keys[nr]] = eff_dict_stat_unc[keys[nr]][:, 0]
                
                if "data" in keys[nr].casefold():
                    # subtract nominal bkg 
                    # second columns is the bkg contribution 
                    vals = (((eff_dict[keys[nr]][:,0]-eff_dict[keys[nr+1]][:, -1]))/
                            (1-eff_dict[keys[nr+1]][:, -2]))
                
                    # unc_vals = np.sqrt(eff_dict[keys[nr]][:,0]*(1-eff_dict[keys[nr]][:,0])*weights_sum)
                    unc_vals = eff_dict_stat_unc[keys[nr]]**2
                
                    # eff_dict_stat_unc[keys[nr]] = np.sqrt(unc_vals)

                    # get systematics uncertainty
                    if eff_sys is not None:
                        eff_sys_unc = get_sys_unc_on_eff(eff_dict[keys[nr]][:,0], eff_sys,
                                                         average_bool=not plot_individual_points)
                        if plot_individual_points:
                            for i,j in eff_sys_unc.items():
                                eff_dict[f'Data - {i} bkg'] = j
                        else:
                            full_unc = np.sqrt(np.sum([(vals-j)**2 
                                                    for _, j in eff_sys_unc.items()], 0)
                                            +unc_vals)
                            if isinstance(total_unc_lst, list):
                                total_unc_lst[nr] = full_unc
                            else:
                                eff_dict_stat_unc[keys[nr]] = full_unc

                    eff_dict[keys[nr]] = vals
                     
                else:
                    # if plot_individual_points:
                    #     eff_sys_unc = get_sys_unc_on_eff(eff_dict[keys[nr]][:,0], eff_sys)
                    # # save bkg eff
                    eff_dict[keys[nr]] = (((eff_dict[keys[nr]][:,0]-eff_dict[keys[nr]][:, -1]))/
                            (1-eff_dict[keys[nr]][:, -2]))
        else:
            # only select prob of passing
            eff_dict.update({i:j[:, 0] for i,j in eff_dict.items() if 'bin' not in i})
            eff_dict_stat_unc.update({i:j[:, 0] for i,j in eff_dict_stat_unc.items() if 'bin' not in i})

        eff_dict["bins"] = conds_bins_lst

    ############ PLOTTING ############
    if kwargs.get("plot_bool", False):
        # if not plot_individual_points:
        return eff_dict, plot_efficiencies(
            eff_dict, eff_dict_stat_unc=eff_dict_stat_unc, 
            total_unc_lst=total_unc_lst, **kwargs)
        # else:
        #     keys = list(eff_dict.keys())
        #     # labels = [i for i in labels if 'down' in i]
        #     for i in labels[2:]:
        #         _kwargs = copy.deepcopy(kwargs)
        #         idx_labels_to_run = [nr for nr,j in enumerate(eff_dict.keys()) if i in j]
        #         labels_to_run = [keys[nr] for nr in idx_labels_to_run]
        #         eff_dict_to_run = {i: eff_dict[i] for i in labels_to_run}
        #         eff_dict_stat_unc_to_run = {i: eff_dict_stat_unc.get(i,None) for i in labels_to_run}
        #         eff_dict_to_run["bins"] = eff_dict["bins"]
        #         _kwargs['labels'] = labels_to_run
        #         _kwargs['truth_key'] = labels_to_run[0]
        #         _kwargs['legend_title']+=f' {i}'
        #         style = _kwargs['styles'][0]
        #         style['label']  = labels_to_run[0]
        #         _kwargs['styles'] = [copy.deepcopy(style)]
        #         style['label']  = labels_to_run[-1]
        #         style['color']  = 'black'
        #         _kwargs['styles'].append(style)
        #         fig = plot_efficiencies(eff_dict_to_run, eff_dict_stat_unc = eff_dict_stat_unc_to_run, total_unc_lst=[None]*len(eff_dict_to_run), **_kwargs)
        #         # misc.save_fig(fig, f"outfiles/eff_{i}.png")
        # return eff_dict, fig
    return eff_dict

def plot_efficiencies(eff_dict, eff_dict_stat_unc=None, total_unc_lst=None, **kwargs):
    labels = kwargs.get('labels')
    bins = eff_dict["bins"][0]
    if labels is None:
        labels = eff_dict.keys()
    
    if eff_dict_stat_unc is None:
        eff_dict_stat_unc = {}
    
    eff_dict = {i:j for i,j in eff_dict.items() if "_bkg" not in i}

    styles = kwargs.get("styles")
    if styles is None:
        styles = [{"color": COLORS[nr]} for nr in range(len(eff_dict)-1)]
    else:
        # Check if each style in styles contains a color, if not, add it
        for i, style in enumerate(styles):
            if "color" not in style:
                styles[i]["color"] = COLORS[i % len(COLORS)]  # Cycle through colors if more styles than colors

    if len(eff_dict)-1 > len(styles): # -1 for bins
        raise ValueError("Colors is not long enough to match eff_dict length")

    fig_kwargs = kwargs.get('fig_kwargs')

    if fig_kwargs is not None:
        fig = fig_kwargs.get('fig')
        ax_1 = fig_kwargs.get('ax')
        ax_2 = fig_kwargs.get('ax_ratio')  
    elif eff_dict['bins'][0] == eff_dict['bins'][1]:
        fig, (ax_1, ax_2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 8), sharex="col"
        )
    else:
        fig, ax_1 = plt.subplots(1, 1, figsize=(8, 8))
        ax_2=None
    nr = 0
    name_lst=[]
    for name, values in eff_dict.items():
        if ("bins" in name) | ("_bkg" in name):
            continue
        if isinstance(total_unc_lst, list):
            total_unc = kwargs["total_unc"][nr]
        else:
            total_unc = (total_unc_lst if "transport" in name.lower()
                else None)
        sigma = (kwargs.get("transport_unc", None) if "transport" in name.lower()
                    else eff_dict_stat_unc.get(name) )
        
        if ('lw' or 'linewidth') not in styles[nr]:
            styles[nr]['lw'] = 2.5
        
        if "marker" in styles[nr]:
            bins = np.array(eff_dict["bins"][nr])
            xs_value = (bins[:-1] + bins[1:]) / 2
            xerr = (bins[:-1] - bins[1:]) / 2
            # plotting
            ax_1.errorbar(xs_value, np.ravel(values), yerr=sigma,
                            xerr = np.abs(xerr), **styles[nr]
                            )

            #additional uncertainties
            if total_unc is not None:
                print(nr)
                ax_1.stairs(values+total_unc, bins, alpha=0.2,
                        color=styles[nr].get("color", None),
                        baseline=values-total_unc, fill=True,
                        label=kwargs.get("total_unc_label", None)
                        )
                
        else:
            ax_1 = plot_stairs(
                np.ravel(values),
                eff_dict['bins'][nr],
                ax=ax_1,
                normalise=False,
                style=styles[nr],
                sigma=sigma,
                total_sigma=total_unc,
                total_sigma_label=kwargs.get("total_unc_label", None),
            )
                
        name_lst.append(styles[nr].get("label"))
        if total_unc is not None:
            name_lst.append(kwargs.get("total_unc_label", None))
        nr += 1
    handles, lgd_labels = ax_1.get_legend_handles_labels()
    order = [np.argmax(np.in1d(name_lst, i)) for i in lgd_labels]
    lgd_labels = np.array(lgd_labels)[order]
    handles = np.array(handles)[order]

    legend_title = f"{atlas_utils.atlas_str}\n"
    legend_title += kwargs.get("legend_title", f"DL1r cut at {kwargs.get('eff_cut', 'Unknown')}")
    ax_1.legend(handles,lgd_labels, title=legend_title, loc=kwargs.get('legend_loc', "lower center"), frameon=False)

    transport_unc = kwargs.get("transport_unc", None)
    total_unc = kwargs.get("total_unc", None)
    
    ######### pack for plot_ratio #########
    eff_dict_for_ratio = {name: {"counts": [eff_dict[name]]} for name in labels}

    # fill in uncertainties
    for nr, (name, values) in enumerate(eff_dict_for_ratio.items()):
        
        # stat uncertainty
        if (transport_unc is not None) & ("transport" in name.lower()):
            eff_dict_for_ratio[name]["unc"] = [transport_unc]
        elif eff_dict_stat_unc.get(name) is not None:
            eff_dict_for_ratio[name]["unc"] = [eff_dict_stat_unc[name]]

        # stat+system uncertainty called total_unc
        if (isinstance(kwargs.get("total_unc"), list)
            and (kwargs["total_unc"][nr] is not None)):
            eff_dict_for_ratio[name]["total_unc"] = [kwargs["total_unc"][nr]]
        
    eff_dict_for_ratio["bins"] = np.array(eff_dict["bins"][0])

    if ax_2 is not None:
        plot_ratio(
            eff_dict_for_ratio,
            truth_key=kwargs.get("truth_key", labels[0]),
            ax=ax_2,
            ylim=kwargs.get("ratio_ylim", [0.9, 1.1]),
            styles=styles,
            normalise=False,
            # reverse_ratio=True if calculate_sf else False,
            zero_line_unc=True,
            alpha=0.2,
            overwrite_uncertainty=False
        )
        ax_2.set_xlabel(kwargs.get("xlabel", r"$p_\mathrm{T}$ [GeV]"))
        ax_2.set_ylabel(kwargs.get("ratio_ylabel", "Ratio"))
        ax_2.set_xlim([bins[0], bins[-1]])
        ax_2.set_xscale("log")
    else:
        ax_1.set_xlim([bins[0], bins[-1]])
        ax_1.set_xscale("log")
        ax_1.set_xlabel(kwargs.get("xlabel", r"$p_\mathrm{T}$ [GeV]"))

    if kwargs.get("ylim") is not None:
        ax_1.set_ylim(kwargs.get("ylim"))

    ax_1.set_ylabel(kwargs.get("ylabel", "Efficiency"))

    return fig

### from tools package

# def generate_figures(nrows, ncols=2, **kwargs):
#     heights = [3, 1] * nrows
#     gs_kw = {"height_ratios": heights}
#     fig, ax = plt.subplots(
#         ncols=ncols,
#         nrows=nrows * 2,
#         gridspec_kw=gs_kw,
#         figsize=(8 * ncols, 6 * nrows),
#         **kwargs,
#     )
#     ax = np.reshape(ax, (2 * nrows, ncols))
#     return fig, ax


# def plot_stairs(
#     counts,
#     bins,
#     sigma=None,
#     total_sigma=None,
#     total_sigma_label=None,
#     ax=None,
#     normalise=True,
#     style={},
# ):
#     if normalise:
#         counts = counts / np.sum(counts)
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(15, 7))
#     ax.stairs(counts, edges=bins, **style)
#     if sigma is not None:
#         ax.stairs(
#             counts + sigma,
#             edges=bins,
#             alpha=0.30,
#             color=style.get("color", None),
#             # label=style.get("label", None),
#             baseline=counts - sigma,
#             fill=True,
#         )
#     if total_sigma is not None:
#         ax.stairs(
#             counts + total_sigma,
#             edges=bins,
#             alpha=0.15,
#             color=style.get("color", None),
#             label=total_sigma_label if total_sigma_label is not None else None,
#             baseline=counts - total_sigma,
#             fill=True,
#         )
#     return ax


# def hist_uncertainty(arr, weights, bins, stacked_bool=False):
#     if isinstance(weights[0], np.ndarray):
#         weights = [i**2 for i in weights]
#         counts, _, _ = plt.hist(arr, weights=weights, bins=bins, stacked=stacked_bool)
#     else:
#         counts, _, _ = plt.hist(
#             arr, weights=weights**2, bins=bins, stacked=stacked_bool
#         )
#     plt.close()
#     if stacked_bool:
#         return np.sqrt(counts[-1])
#     else:
#         return np.sqrt(counts)


# def plot_hist(*args, **kwargs) -> plt.Figure:  # pylint: disable=too-many-locals
#     """plot histogram

#     Parameters
#     ----------
#     target : np.ndarray
#         target distribution
#     source : np.ndarray
#         source distribution
#     trans : np.ndarray
#         transported source distribution
#     binrange : tuple, optional
#         range of the x axis, by default (-2, 2)

#     Returns
#     -------
#     plt.Figure
#         output figure
#     """

#     style = kwargs.setdefault("style", {"bins": 20, "histtype": "step"})
#     mask_sig = None
#     if kwargs.get("remove_placeholder", False):
#         if (-999 in args[0]) & (len(args) > 1):  # remove -999 as a placeholder i args
#             if "mask_sig" in kwargs:
#                 mask_sig = [
#                     j[np.array(np.ravel(i)) != -999]
#                     for i, j in zip(args, kwargs["mask_sig"])
#                 ]
#             args = tuple([np.array(i)[np.array(i) != -999] for i in args])

#     if (not "range" in style) and (isinstance(style.get("bins", 20), int)):
#         percentile_lst = kwargs.get("percentile_lst", [0.05, 99.95])
#         # just select percentile from first distribution
#         if len(args[0]) == 2:  # stakced array
#             style["range"] = np.percentile(args[0][-1], percentile_lst)
#         else:
#             style["range"] = np.percentile(args[0], percentile_lst)

#     names = kwargs.get("names", [f"dist_{i}" for i in range(len(args))])
#     weights = kwargs.get("weights", [np.ones(len(i)) for i in args])

#     if mask_sig is None:
#         mask_sig = kwargs.get(
#             "mask_sig", [np.ravel(np.ones_like(i) == 1) for i in args]
#         )

#     uncertainties = kwargs.get("uncertainties", [None] * len(args))

#     counts_dict = {}
#     for nr, (name, dist, uncertainty) in enumerate(zip(names, args, uncertainties)):
#         fig = plt.figure()
#         weight = weights[nr]
#         if not all(mask_sig[nr]):
#             mask = mask_sig[nr]
#             counts, bins, _ = plt.hist(
#                 [dist[~mask], dist[mask]],
#                 stacked=True,
#                 weights=[weight[~mask], weight[mask]],
#                 **style,
#             )  # stacked doesnt work for np.histogram
#             unc = [
#                 None,
#                 (
#                     hist_uncertainty(
#                         [dist[~mask], dist[mask]],
#                         weights=[weight[~mask], weight[mask]],
#                         bins=bins,
#                         stacked_bool=True,
#                     )
#                     if uncertainty is None
#                     else uncertainty
#                 ),
#             ]
#             if kwargs.get("norm", False):
#                 counts = list(counts / np.sum(counts, 1)[:, None])
#             else:
#                 counts = list(counts)
#             weight = [weight, weight]
#         else:
#             counts, bins, _ = plt.hist(dist, weights=weight, **style)
#             if weight is None:
#                 unc = [np.sqrt(counts) if uncertainty is None else uncertainty]
#                 weight = np.ones((len(dist)))
#             else:
#                 unc = [
#                     (
#                         hist_uncertainty(
#                             dist, weights=weight, bins=bins, stacked_bool=False
#                         )
#                         if uncertainty is None
#                         else uncertainty
#                     )
#                 ]
#             # if not style.get("stacked", False):
#             counts = [counts]

#             weight = [weight]

#         counts_dict[name] = {"counts": counts, "unc": unc, "weight": weight}

#         plt.close(fig)

#     counts_dict["bins"] = bins
#     if kwargs.get("plot_bool", True):
#         return plot_hist_1d(counts_dict, **kwargs)
#     else:
#         return counts_dict, None


# def plot_hist_integral(
#     distributions,
#     truth_key,
#     var_names,
#     conds_bins,
#     plot_kwargs={},
#     conds_names=None,
#     save_path=None,
#     **kwargs,
# ):
#     "still requires some work"

#     if isinstance(conds_bins, (list, np.ndarray)):
#         conds_bins = np.round(
#             np.percentile(distributions[truth_key]["conds"], conds_bins, 0), 3
#         )
#         ax = plot_hist_integration_over_bins(
#             distributions,
#             bins=conds_bins,
#             plot_kwargs=plot_kwargs,
#             var_names=var_names,
#             **kwargs,
#         )
#     else:
#         if conds_names is None:
#             raise ValueError("When giving conds_bins as dict, conds_names is required!")

#         for key, item in conds_bins.items():
#             n_bins = len(item)

#             fig, ax_all = generate_figures(n_bins - 1, len(var_names))
#             for n_row in range(n_bins):
#                 bins = np.c_[
#                     [items[n_row : n_row + 2].T for i, items in conds_bins.items()]
#                 ].T
#                 # ax = ax_all[2*n_row: 2*n_row+2, 0]
#                 plot_hist_integration_over_bins(
#                     distributions,
#                     bins=bins,
#                     plot_kwargs=plot_kwargs,
#                     ax=ax_all[2 * n_row : 2 * n_row + 2],
#                     var_names=var_names,
#                     conds_col_nr=np.argmax(key == np.array(conds_names)),
#                     **kwargs,
#                 )

#             if save_path is not None:
#                 misc.save_fig(
#                     fig,
#                     (
#                         f"{save_path}/{var_names}_{key}_{item}.{kwargs.get('format', 'png')}"
#                     ),
#                 )


# def plot_hist_integration_over_bins(
#     distributions,
#     bins,
#     plot_kwargs,
#     legend_title,
#     var_names,
#     ax=None,
#     conds_col_nr=None,
#     save_path=None,
#     **kwargs,
# ):
#     legend_title = kwargs.get("legend_title", "")
#     for low, high in zip(bins[:-1], bins[1:]):
#         dists = []
#         labels = []
#         bootstraps = []
#         systematics = []
#         for key, values in distributions.items():
#             if conds_col_nr is None:
#                 mask_conds = np.all(
#                     [
#                         (low[i] <= values["conds"][:, i])
#                         & (high[i] >= values["conds"][:, i])
#                         for i in range(len(low))
#                     ],
#                     0,
#                 )
#             else:
#                 mask_conds = (low[0] <= values["conds"][:, conds_col_nr]) & (
#                     high[0] >= values["conds"][:, conds_col_nr]
#                 )

#             dists.append(values["dist"][mask_conds])
#             labels.append(values["labels"][mask_conds] == 1)
#             bootstraps.append(
#                 [i[mask_conds] for i in values["bootstraps"]]
#                 if "bootstraps" in values
#                 else []
#             )
#             systematics.append(
#                 [i[mask_conds] for i in values["systematics"]]
#                 if "systematics" in values
#                 else []
#             )

#         for col_nr in range(len(var_names)):
#             plot_args = copy.deepcopy(plot_kwargs)
#             title_str = (
#                 f"{legend_title} pT: [{', '.join(map(str, low))}:"
#                 f" {', '.join(map(str, high))})"
#             )

#             if ax is None:
#                 fig, ax = generate_figures(1, 1, sharex="col")
#             # histogram
#             counts, _ = plot_hist(
#                 *[i[:, col_nr] for i in dists],
#                 plot_bool=False,
#                 mask_sig=labels,
#                 **plot_args,
#             )

#             if len(bootstraps[1]) > 0:  # TODO should not be hard coded
#                 counts_boot, _ = plot_hist(
#                     *[i[:, col_nr] for i in bootstraps[1]],
#                     plot_bool=False,
#                     mask_sig=[labels[1]] * len(bootstraps[1]),
#                     **plot_args,
#                 )
#                 bootstraps_unc = np.std(
#                     [counts_boot[i]["counts"][-1] for i in counts_boot if "dist" in i],
#                     0,
#                 )
#                 counts["dist_1"]["unc"][-1] = bootstraps_unc
#             if len(systematics[1]) > 0:
#                 counts_sys, _ = plot_hist(
#                     *[i[:, col_nr] for i in systematics[1]],
#                     plot_bool=False,
#                     mask_sig=[labels[1]] * len(systematics[1]),
#                     **plot_args,
#                 )

#             counts, _ = plot_hist_1d(
#                 counts,
#                 ax=ax[0, col_nr],
#                 legend_kwargs={"title": title_str},
#                 **plot_args,
#             )
#             if kwargs.get("log_y", False):
#                 ax[0, col_nr].set_yscale("log")

#             # ratio plot
#             plot_ratio(
#                 counts,
#                 truth_key="dist_0",
#                 ax=ax[1, col_nr],
#                 # ylim=kwargs.get("ylim", [0.95, 1.05]),
#                 # styles=copy.deepcopy(dist_styles),
#                 zero_line_unc=True,
#                 **plot_args,
#             )

#             ax[1, col_nr].set_xlabel(var_names[col_nr])
#             if "ylim" in plot_args:
#                 ax[1, col_nr].set_ylim(plot_args["ylim"])
#             if (save_path is not None) & (not kwargs.get("single_figure", False)):
#                 misc.save_fig(
#                     fig,
#                     (
#                         f"{save_path}/{var_names}_{low}_{high}_{col_nr}.{kwargs.get('format', 'png')}"
#                     ),
#                 )
#     # return ax


# def plot_hist_1d(counts_dict: dict, **kwargs):
#     """Plot histogram

#     Parameters
#     ----------
#     counts : dict
#         Should contain:   bins:[], truth:{counts:[], unc:[], weights: []},
#                           different_counts:{counts:[],unc:[], weights: []}

#     Returns
#     -------
#     dict, ax
#         return counts and ax
#     """
#     if "bins" in counts_dict:
#         bins = counts_dict["bins"]
#     else:
#         raise ValueError("counts_dict missing bins")

#     ax = kwargs.get("ax", None)
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))

#     ax.xaxis.set_minor_locator(AutoMinorLocator())
#     ax.tick_params(which="minor", length=4)

#     # define bins
#     xs_value = (bins[:-1] + bins[1:]) / 2
#     # fixing the bound of the x axis because of steps
#     if kwargs.get("xerr_on_errorbar", False):
#         xerr = np.abs((bins[:-1] - bins[1:]) / 2)
#     else:
#         # xerr[1:-1] = np.zeros_like(xerr[1:-1])
#         xerr = np.zeros_like(bins[:-1])

#     for nr, (name, counts) in enumerate(counts_dict.items()):
#         if name == "bins":
#             continue

#         # if uncertainties or weight are missing
#         counts.setdefault("unc", [np.zeros_like(counts["counts"])[0]])
#         counts.setdefault("weight", [np.ones_like(counts["counts"])[0]])
#         # this type of loop is need if hist(stacked=True)
#         for nr_count, (count, unc, weight) in enumerate(
#             zip(counts["counts"], counts["unc"], counts["weight"])
#         ):
#             style = copy.deepcopy(kwargs.get("dist_styles", [{}] * (nr + 1))[nr])
#             if (len(counts["counts"]) == 2) & (nr_count == 0):
#                 style["label"] = "Background"
#                 style["alpha"] = 0.5
#             # else:  TODO weird to force alpha=1 right?
#             #     style["alpha"] = 1

#             style.pop("drawstyle", None)
#             if not isinstance(weight[0], float):
#                 sum_weight = sum([np.sum(i) for i in weight])
#             else:
#                 sum_weight = np.sum(weight)

#             yerr = (
#                 None
#                 if unc is None
#                 else unc / sum_weight if kwargs.get("normalise", True) else unc
#             )
#             hist_counts = count / sum_weight if kwargs.get("normalise", True) else count

#             if "marker" in style:
#                 # style["linestyle"]="none"
#                 eb1 = ax.errorbar(
#                     xs_value,
#                     hist_counts,
#                     xerr=xerr,
#                     yerr=(
#                         None
#                         if unc is None
#                         else unc / sum_weight if kwargs.get("normalise", True) else unc
#                     ),
#                     **style,
#                 )
#                 # eb1[-1][0].set_linestyle(style.get("linestyle", "solid"))
#             else:
#                 yerr = 0 if yerr is None else yerr
#                 # ax.stairs(count/np.sum(weight) if kwargs.get("normalise", True) else count, bins, **style)

#                 # eb1 = ax.errorbar(
#                 #     xs_value,
#                 #     count/np.sum(weight) if kwargs.get("normalise", True) else count,
#                 #     marker=".",
#                 #     linewidth=0,
#                 #     yerr=None if unc is None else unc/np.sum(weight),
#                 #     )
#                 # print(eb1[-1])
#                 # eb1[-1][0].set_linestyle(style.get("linestyle", "solid"))
#                 if not isinstance(
#                     weight[0], (int, float, np.float32, np.float64)
#                 ):  # only for stacked=True
#                     ax.stairs(
#                         hist_counts[-1],
#                         bins,
#                         baseline=hist_counts[-1] - yerr[-1],
#                         fill=True,
#                         alpha=0.1,
#                         color=style.get("color", None),
#                     )
#                     ax.stairs(
#                         hist_counts[-1],
#                         bins,
#                         baseline=hist_counts[-1] + yerr[-1],
#                         fill=True,
#                         alpha=0.1,
#                         color=style.get("color", None),
#                     )
#                     for nr, (i, label) in enumerate(
#                         zip(hist_counts, kwargs["stacked_labels"])
#                     ):  # need to have style be list
#                         ax.stairs(i, bins, zorder=1 / (nr + 1), **style, label=label)
#                 else:
#                     ax.stairs(
#                         hist_counts,
#                         bins,
#                         baseline=hist_counts - yerr,
#                         fill=True,
#                         alpha=0.1,
#                         color=style.get("color", None),
#                     )
#                     ax.stairs(
#                         hist_counts,
#                         bins,
#                         baseline=hist_counts + yerr,
#                         fill=True,
#                         alpha=0.1,
#                         color=style.get("color", None),
#                     )
#                     ax.stairs(hist_counts, bins, **style)  # , zorder=1/(nr_count+1))
#     legend_kwargs = KWARGS_LEGEND.copy()
#     legend_kwargs.update(kwargs.get("legend_kwargs", {}))
#     ax.legend(**legend_kwargs)

#     ax.tick_params(axis="both", which="major", labelsize=LABELSIZE)
#     ylabel = "Normalised counts" if kwargs.get("normalise", True) else "#"
#     ax.set_ylabel(ylabel, fontsize=FONTSIZE)
#     if kwargs.get("log_yscale", False):
#         ax.set_yscale("log")

#     return counts_dict, ax


# def plot_ratio(counts: dict, truth_key: str, **kwargs):
#     """Ratio plot

#     Parameters
#     ----------
#     counts : dict
#         Counts from plot_hist
#     truth_key : str
#         Which key in counts are the truth
#     kwargs:
#         Add ylim if you want arrow otherwise they are at [0,2]
#     Returns
#     -------
#     _type_
#         _description_
#     """
#     ax = kwargs.get("ax", None)
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(11, 8))
#     bins = counts["bins"]
    

#     styles = copy.deepcopy(kwargs.get("styles", [{} for _ in counts.keys()]))
#     ylim = kwargs.get("ylim", [0, 2])
#     xs_value = (bins[:-1] + bins[1:]) / 2
#     xerr = (bins[:-1] - bins[1:]) / 2
#     if "counts" in counts[truth_key]:
#         counts_truth = counts[truth_key]["counts"][-1]
#     else:
#         counts_truth = counts[truth_key]
#     counts_truth = np.array(counts_truth)
#     if kwargs.get("zero_line", True):
#         try:
#             # color = styles[np.argmax(np.in1d(list(counts.keys())[:-1], truth_key))][
#             #     "color"
#             # ]
#             style = styles[np.argmax(np.in1d(list(counts.keys())[:-1], truth_key))]
#         except:
#             style = {'color': 'black', 'lw':2, 'ls': 'dashed'}
            
#         style['zorder'] = -1
#         ax.plot(
#             bins,
#             np.ones_like(bins),
#             # label="Zero line",
#             # color=color,
#             # linewidth=2,
#             # linestyle="dashed",
#             **style,
#             # zorder=-1,
#         )
#         if kwargs.get("zero_line_unc", False):
#             if np.sum(counts[truth_key]["unc"][-1]) != 0:
#                 unc = np.divide(
#                     counts[truth_key]["unc"][-1], counts_truth
#                 )
#             else:
#                 unc = np.sqrt(counts_truth) / counts_truth
                
#             one_line = np.ones_like(xs_value)

#             # ax.fill_between(
#             #     xs_value,
#             #     one_line - np.abs(unc),
#             #     one_line + np.abs(unc),
#             #     color=color,
#             #     alpha=kwargs.get("alpha", 0.3),
#             # )
#             ax.stairs(
#                 one_line + np.abs(unc),
#                 bins,
#                 alpha=kwargs.get('alpha', 0.5),
#                 # color=color,
#                 baseline=one_line - np.abs(unc),
#                 fill=True,
#                 # zorder=-1,
#                 **style,
#             )

#         if "total_unc" in counts[truth_key]:
#             total_unc = np.array(counts[truth_key]["total_unc"][-1])

#             if kwargs.get("normalise", True):
#                 total_unc = (total_unc / count_pr_bins.sum()) / (
#                     counts_truth / counts_truth.sum()
#                 )
#             else:
#                 total_unc = total_unc / counts_truth

#             ax.stairs(
#                 one_line + total_unc,
#                 bins,
#                 alpha=kwargs.get('alpha', 0.3),
#                 # color=color,
#                 baseline=one_line - total_unc,
#                 fill=True,
#                 **style,
#             )
#     nr = 0
#     for (name, count), style in zip(counts.items(), styles):
#         if (name == "bins") or (name == truth_key):
#             continue

#         if "counts" in count:
#             if not isinstance(count.get("counts", None), list):
#                 # not isinstance(count.get("unc",None), list)
#                 raise TypeError("Counts in dict has to be a list")
#             count_pr_bins = np.array(count["counts"][-1])
#         else:
#             count_pr_bins = np.array(count)

#         if "unc" in count:
#             if not isinstance(count.get("unc", None), list):
#                 # not isinstance(count.get("unc",None), list)
#                 raise TypeError("Counts in dict has to be a list")
#             menStd = np.array(count["unc"][-1])
#             if len(menStd.shape) > 1:
#                 menStd = menStd[-1]
#         else:
#             menStd = np.zeros_like(count_pr_bins)
#         with np.errstate(divide="ignore"):
#             if kwargs.get("normalise", True):
#                 y_counts = (count_pr_bins / count_pr_bins.sum()) / (
#                     counts_truth / counts_truth.sum()
#                 )
#                 yerr_relative = (menStd / count_pr_bins.sum()) / (
#                     counts_truth / counts_truth.sum()
#                 )
#             else:
#                 y_counts = count_pr_bins / counts_truth
#                 yerr_relative = menStd / counts_truth

#         if kwargs.get("reverse_ratio", False):
#             y_counts = 1 / y_counts

#         # up or down error if outside of ylim
#         yerr_relative = np.nan_to_num(yerr_relative, nan=10, posinf=10)
#         mask_down = ylim[0] >= y_counts
#         mask_up = ylim[1] <= y_counts

#         linestyle = style.pop("linestyle", "solid")

#         # additional uncertainties
#         if "total_unc" in count:
#             total_unc = np.array(count["total_unc"][-1])
#             if kwargs.get("normalise", True):
#                 total_unc = (total_unc / count_pr_bins.sum()) / (
#                     counts_truth / counts_truth.sum()
#                 )
#             else:
#                 total_unc = total_unc / counts_truth
#             ax.stairs(
#                 y_counts + total_unc,
#                 bins,
#                 alpha=kwargs.get("alpha", 0.3),
#                 color=style.get("color", None),
#                 baseline=y_counts - total_unc,
#                 fill=True,
#             )

#         # plotting
#         eb1= ax.errorbar(xs_value, y_counts, yerr=yerr_relative, xerr=np.abs(xerr), linestyle='none', **style)
#         if linestyle != 'none':
#             eb1[-1][0].set_linestyle(linestyle)
#             eb1[-1][1].set_linestyle(linestyle) 
        
#         # marker up
#         if ("label" not in style) & kwargs.get("legend_bool", True):
#             style["label"] = name
#         style.update({"marker": "^", "s": 35, "alpha": 1})
#         for i in ["markersize", "linestyle", 'error_kw']:
#             style.pop(i, None)
#         ax.scatter(
#             xs_value[mask_up],
#             np.ones(mask_up.sum()) * (ylim[1] - ylim[1] / 100),
#             **style,
#         )

#         # marker down
#         style["marker"] = "v"
#         ax.scatter(
#             xs_value[mask_down],
#             np.ones(mask_down.sum()) * (ylim[0] + ylim[0] / 100),
#             **style,
#         )

#         nr += 1

#     ax.set_ylabel(kwargs.get("ylabel", "Ratio"), fontsize=FONTSIZE)
#     ax.set_ylim(ylim)
#     ax.tick_params(axis="both", which="major", labelsize=LABELSIZE)
#     return ax


if __name__ == "__main__":
    # %matplotlib widget
    if False:
        style_target = {
            "marker": "o",
            "color": "black",
            "label": "Data",
            "linewidth": 0,
        }
        style_source = {
            "linestyle": "dotted",
            "color": "blue",
            "label": r"$b$-jets",
            "drawstyle": "steps-mid",
        }
        style_trans = {
            "linestyle": "dashed",
            "color": "red",
            "label": "Transported",
            "drawstyle": "steps-mid",
        }
        fig, (ax_1, ax_2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(9, 5), sharex="col"
        )

        target = np.random.uniform(-15, 15, 10_000)
        source = np.random.uniform(-15, 15, 10_000)
        trans = np.random.uniform(-15, 15, 10_000)
        bins = [-15, 0.6650, 2.1950, 3.2450, 4.5650, 15]
        mask_sig = [source < 0.5, np.ones_like(trans), np.ones_like(target)]
        data, ax_1 = plot_hist(
            source,
            trans,
            target,
            dist_styles=[style_source, style_trans, style_target],
            ax=ax_1,
            mask_sig=mask_sig,
            style={"bins": bins},
        )
        style_source.pop("drawstyle")
        style_trans.pop("drawstyle")
        ax_2 = plot_ratio(
            counts=data,
            ax=ax_2,
            truth_key="dist_2",
            styles=[style_source, style_trans],
            ylim=[0.98, 1.02],
        )
        fig.tight_layout()
    else:
        # fig5 = plt.figure(constrained_layout=True)
        n_fig = 5
        # widths = [3, 1]*n_fig
        heights = [3, 1] * n_fig
        # heights = [1, 3, 2]
        gs_kw = {"height_ratios": heights}
        # spec5 = fig5.add_gridspec(ncols=2, nrows=n_fig*2,
        #                         #   width_ratios=widths,
        #                           height_ratios=heights)
        fig, ax = plt.subplots(
            ncols=2, nrows=n_fig * 2, gridspec_kw=gs_kw, figsize=(8, 6 * n_fig)
        )
