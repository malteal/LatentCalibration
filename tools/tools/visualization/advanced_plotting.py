"Advanced plotting functions"
from typing import Tuple, Union


import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import logging 
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

def create_cmap(vmin: float, vmax: float, size_of_cmap:int, cmap: str='viridis',
                color_outsize=[1, 0, 0, 1], size_outside:int=16,
                max_size: float=2) -> Tuple:
    """
    this should be cmap=cmap, norm=norm in the pcolormesh function or sth else
    """
    # Step 1: Use the 'viridis' colormap for the range [0, 1]
    viridis = plt.cm.get_cmap(cmap, size_of_cmap)  # Get size_of_cmap colors from 'viridis'
    newcolors = viridis(np.linspace(vmin, vmax, size_of_cmap))  # Create a color array for these colors

    # Step 2: Append a distinct color (e.g., red) for values above 1
    red = np.array(color_outsize)  # RGBA for red
    newcolors = np.vstack((newcolors, [red]*size_outside))  # Append red to the end of the color array

    # Create a new colormap with the combined colors
    cmap = mcolors.ListedColormap(newcolors)

    # Define boundaries and normalization
    bounds = np.linspace(vmin, vmax, size_of_cmap).tolist() + np.linspace(vmax, max_size, size_outside).tolist()  # Ensures a smooth gradient up to 1, then jumps to 2 for red
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm
def violinplot(x: Union[np.ndarray, pd.DataFrame], features_per_plot: int = 32, plots_per_row: int = 1) -> Tuple[plt.Figure, plt.Axes]:
    """
    x: np.ndarray - Should be a [n_samples, n_features] array
    features_per_plot: int - Number of features per plot
    plots_per_row: int - Number of plots per row
    """
    
    # get n features
    n_features = x.shape[-1]

    n_plots = (n_features // features_per_plot)
    n_plots += 1 if n_plots == 0 else 0
    
    # get number of features in each plot
    n_features_in_each = [n_features // n_plots] * n_plots
    
    # add the remaining features to the last plot
    n_features_in_each[-1] += n_features - np.sum(n_features_in_each)

    # insert the first feature
    n_features_in_each.insert(0, 0)
    
    # get the index of the feature ranges
    f_index = np.cumsum(n_features_in_each)

    # define x as a pandas dataframe
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)

    # Calculate the number of rows needed
    n_rows = (n_plots + plots_per_row - 1) // plots_per_row

    fig, ax = plt.subplots(n_rows, plots_per_row, figsize=(8 * plots_per_row, 6 * n_rows))
    ax = ax.flatten() #if n_rows > 1 else [ax]

    for i,_ax in enumerate(ax):
        data = x.iloc[:, f_index[i]:f_index[i+1]]
        data_melted = data.melt()
        
        # Convert the 'variable' column to numeric
        data_melted['variable'] = pd.to_numeric(data_melted['variable'], errors='coerce')
        
        # Ensure the 'variable' column is numeric
        if not np.issubdtype(data_melted['variable'].dtype, np.number):
            raise ValueError("The 'variable' column could not be converted to numeric.")
        
        # create violin plot
        sn.violinplot(x="variable", y="value",
                          data=data_melted, hue="variable",
                          gap=.1, split=True, inner="quart",
                          ax=_ax, bw_adjust=.5, cut=0)

        
        # Remove the legend
        ax[i].legend_.remove()

    # Remove any unused subplots
    for j in range(n_plots, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()
    
    return fig, ax


def plot_pcolormesh(function: callable, x_vals: np.ndarray,
                    xlim: list, ylim: list,
                    conditions: np.ndarray=None, **kwargs: dict):
    """plotting pcolormesh of function

    Parameters
    ----------
    f_func : callable
        ML model
    x_vals : torch.Tensor
        other Conditional input
    conds : torch.Tensor
        Conditional input
    kwargs : dict
        additional inputs

    Returns
    -------
    plt.Figure
        output figure
    """
    npts = kwargs.get("npts", 21)

    n2dpts = npts * npts
    xs_value = np.mgrid[xlim[0] : xlim[1] : npts * 1j]
    ys_value = np.mgrid[ylim[0] : ylim[1] : npts * 1j]
    
    x_mesh, y_mesh = np.meshgrid(xs_value, ys_value)

    x_vals = np.tile(x_vals, (n2dpts, 1))
    if conditions is not None:
        raise NotImplementedError("conditions not implemented")
        # conditions = torch.cat([conds.unsqueeze(1)] * n2dpts, axis=0)

    if kwargs["yidx"] > kwargs["xidx"]:
        ten = np.concatenate(
            [x_vals[:, 0 : kwargs["xidx"]]]
            + [x_mesh.reshape((-1, 1))]
            + [x_vals[:, kwargs["xidx"] + 1 : kwargs["yidx"]]]
            + [y_mesh.reshape((-1, 1))]
            + [x_vals[:, kwargs["yidx"] + 1 :]],
            axis=1,
        )
    else:
        ten = np.concatenate(
            [x_vals[:, 0 : kwargs["yidx"]]]
            + [y_mesh.reshape((-1, 1))]
            + [x_vals[:, kwargs["yidx"] + 1 : kwargs["xidx"]]]
            + [x_mesh.reshape((-1, 1))]
            + [x_vals[:, kwargs["xidx"] + 1 :]],
            axis=1,
        )
    if kwargs.get('axis_trans') is not None:
        new_axis = kwargs['axis_trans'](ten)

        x_mesh = new_axis[:, kwargs["xidx"]].reshape((npts, npts))
        y_mesh = new_axis[:, kwargs["yidx"]].reshape((npts, npts))

    # get function output
    output = function(ten, conditions).reshape((npts, npts))
    print(f'max output: {output.max()}')

    fig,ax_1 = plt.subplots(1,1,figsize=(8,8))
    
    mesh = ax_1.pcolormesh(
        x_mesh, y_mesh, output, shading="gouraud",
        **kwargs.get('pcolormesh_kwargs', {})
    )
    cbar = fig.colorbar(mesh, ax=ax_1, **kwargs.get('colorbar_kwargs', {}))
    cbar.set_label(kwargs.get('color_label'))

    ax_1.set_xlabel(kwargs.get("xlabel", ''))
    ax_1.set_ylabel(kwargs.get("ylabel", ''))
    ax_1.set_xlim([x_mesh.min(), x_mesh.max()])
    ax_1.set_ylim([y_mesh.min(), y_mesh.max()])

    if "title" in kwargs:
        ax_1.set_title(kwargs["title"], loc='left',
                       fontdict={"fontsize": 18})
    fig.tight_layout()

    return fig, cbar

if __name__ == "__main__":
    # pass
    # testing functions
    x = np.random.rand(100, 512)
    fig, ax = violinplot(x, features_per_plot=64, plots_per_row=2)
    plt.show()
    
    
    # ### dummy example
    # cmap, norm = aplot.create_cmap(0, 1.00001, 256)
    # fig, cbar = aplot.plot_pcolormesh(run_downscaler, x_vals=avgfeats,
    #                             xidx=idx[0],yidx=idx[1],
    #                             xlabel=labels[0], ylabel=labels[-1], npts=101,
    #                         title=atlasutils.get_atlas_internal_str()+"\n p"+ r"$_\mathrm{T}=$"+f"{pt_str} GeV, "+f"{labels[-1]}={missing_dim} \n"+r"$\sqrt{\mathrm{s}}=13$ TeV, 139 $\mathrm{fb}^{-1}$",
    #                         xlim=[-14, 20],
    #                         ylim=[-17, 12],
    #                         color_label='$p_{bkg}^\prime/p_{data}$',
    #                         # axis_trans=to_probs,
    #                         pcolormesh_kwargs={'cmap':cmap, 'norm':norm},
    #                         )