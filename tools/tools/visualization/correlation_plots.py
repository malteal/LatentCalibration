"correlation plots"

import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm

def get_weights(data):
    return np.ones_like(data) / len(data)

def tensor2numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x

def get_bins(data, nbins=20, sd=None):
    max_ent = data.max().item()
    min_ent = data.min().item()
    if sd is not None:
        max_ent = max(max_ent, sd.max().item())
        min_ent = min(min_ent, sd.min().item())
    return np.linspace(min_ent, max_ent, num=nbins)


def add_hist(ax, data, bin, color, label):
    _, bins, _ = ax.hist(
        data,
        bins=bin,
        density=False,
        histtype="step",
        color=color,
        label=label,
        weights=get_weights(data),
    )
    return bins


def add_error_hist(
    ax, data, bins, color, error_bars=False, normalised=True, label="", norm=None
):
    y, binEdges = np.histogram(data, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = binEdges[1] - binEdges[0]
    norm_passed = norm is None
    n_fact = np.sum(y) if norm_passed else norm
    menStd = np.sqrt(y)
    if normalised or norm_passed:
        y = y / n_fact
        menStd = menStd / n_fact
    if error_bars:
        ax.errorbar(bincenters, y, yerr=menStd, color=color, fmt=".", label=label)
    else:
        ax.bar(
            bincenters,
            menStd,
            width=width,
            edgecolor=color,
            lw=0,
            fc=(0, 0, 0, 0),
            bottom=y,
            hatch="\\\\\\\\\\",
            label=label,
        )
        ax.bar(
            bincenters,
            -menStd,
            width=width,
            edgecolor=color,
            lw=0,
            fc=(0, 0, 0, 0),
            bottom=y,
            hatch="\\\\\\\\\\",
            label=label,
        )


def add_contour(axes, i, j, data, sampled, x_bounds=None):
    if x_bounds is None:
        x_bounds = np.quantile(sampled[:, j], [0.0, 1.0])
        y_bounds = np.quantile(sampled[:, i], [0.0, 1.0])

    sns.kdeplot(
        x=tensor2numpy(sampled[:, j]),
        y=tensor2numpy(sampled[:, i]),
        ax=axes[i, j],
        alpha=0.4,
        levels=3,
        color="red",
        fill=True,
    )
    sns.kdeplot(
        x=tensor2numpy(data[:, j]),
        y=tensor2numpy(data[:, i]),
        ax=axes[i, j],
        alpha=0.4,
        levels=3,
        color="blue",
        fill=True,
    )

    axes[i, j].set_xlim(x_bounds[0], x_bounds[1])
    axes[i, j].set_ylim(y_bounds[0], y_bounds[1])

def plot_feature_spread(
    target_data: np.ndarray,
    sampled: np.ndarray,
    feature_nms: Union[List[str], None] = None,
    labels: Union[str, None] = None,
    save_dir: Union[str, None] = None,
    plot_mode: str = "paper",
    **kwargs
):
    """
    This function plots the spread of features in a dataset. It takes the target data,
    the sampled data, and optional feature names, a tag for the plot, a directory to save the plot to,
    and a plot mode which can be either "diagnose" or "paper". It will plot histograms of the feature values
    for both the target data and the sampled data in a matrix format, with the diagonal showing the spread
    of each individual feature, and the off-diagonal elements showing the feature pairs.

    Parameters:
        target_data (np.ndarray): The target data for which the features should be plotted.
        sampled (np.ndarray): The sampled data from which the features should be plotted.
        feature_nms (List[str], optional): The names of the features. Defaults to None.
        labels (str, optional): A labels to be added to the plot file name. Defaults to None.
        save_dir (Path, optional): The directory where the plot should be saved. Defaults to None.
        plot_mode (str, optional): The plot mode, which can be either "diagnose" or "paper". Defaults to "diagnose".

    Returns:
        None
    """
    nbins = 30
    n_features = sampled.shape[1]
    n_sample = kwargs.get("n_sample", 1_000)

    if feature_nms is None:
        feature_nms = [f"feature_{i}" for i in range(n_features)]

    if labels is None:
        labels = ["Data", "Generated"]
        # tag = "substructure"

    assert n_features == len(
        feature_nms
    ), "Number of feature names must match number of features"
    fig, axes = plt.subplots(
        n_features,
        n_features,
        figsize=(2 * n_features + 3, 2 * n_features + 1),
        gridspec_kw={"wspace": 0.06, "hspace": 0.06},
    )
    for i in tqdm(range(n_features)):
        if i != 0:
            axes[i, 0].set_ylabel(feature_nms[i])
        else:
            axes[0, 0].set_ylabel(
                "Normalised Entries", horizontalalignment="right", y=1.0
            )
        for j in range(n_features):
            if j != 0:
                pass
                axes[i, j].set_yticklabels([])
            axes[-1, j].set_xlabel(feature_nms[j])
            if i != n_features - 1:
                axes[i, j].tick_params(
                    axis="x", which="both", direction="in", labelbottom=False
                )
                # axes[i, j].set_yticks([0, 0.5, 1.])
                if i == j == 0:
                    axes[i, j].tick_params(axis="y", colors="w")
                elif j > 0:
                    axes[i, j].tick_params(
                        axis="y", which="both", direction="in", labelbottom=False
                    )
            if i == j:
                og = target_data[:, i]
                x_bounds = np.percentile(og, kwargs.get("percentiles", [0.0, 99.9]))
                bins = get_bins(
                    og[(og > x_bounds[0]) & (og < x_bounds[1])], nbins=nbins
                )
                bins = add_hist(axes[i, j], sampled[:, i], bins, "red", "Generated")
                add_hist(axes[i, j], target_data[:, i], bins, "blue", "Real")
                add_error_hist(axes[i, j], sampled[:, i], bins=bins, color="red")
                add_error_hist(axes[i, j], target_data[:, i], bins=bins, color="blue")
                axes[i, j].set_yticklabels([])
            if i > j:
                add_contour(axes, i, j, target_data[:n_sample], sampled[:n_sample])
            elif i < j:
                axes[i, j].set_visible(False)

            if "x_lim" in kwargs:
                axes[i, j].set_xlim(kwargs["x_lim"])
            if "y_lim" in kwargs:
                axes[i, j].set_ylim(kwargs["y_lim"])

                

    # labels = ["Real", "Generated"]
    other_handles = [Line2D([], [], color=colors) for colors in ["blue", "red"]]
    fig.legend(
        handles=other_handles,
        labels=labels,
        loc="upper right",
        fontsize=25,
        bbox_to_anchor=(0.9, 0.9),
    )
    fig.align_xlabels(axes)
    fig.align_ylabels(axes)

    # suffix = "pdf" if plot_mode == "paper" else "png"
    if isinstance(save_dir, str):
        fig.savefig(save_dir, bbox_inches="tight")
    
if __name__ == "__main__":
    # %matplotlib widget
    col=3
    target = np.random.normal(0,1,(10_000,col))
    sampled = np.random.normal(-2,0.5,(10_000,col))
    plot_feature_spread(
                    target_data=target,
                    sampled= sampled,
                    n_sample=100
                    )