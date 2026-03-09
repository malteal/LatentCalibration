
import pyrootutils

root = pyrootutils.setup_root(search_from=".", pythonpath=True)

import matplotlib
import h5py
from pathlib import Path

from tools.tools import misc
import tools.tools.visualization.general_plotting as plot

font = {'size': 30}

matplotlib.rc('font', **font)

jets_column_names = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    'jet_energy',
    "jet_nparticles",
]
jets_column_labels = [
    "$p_{T}$ [GeV]",
    "$\eta$",
    "$\phi$",
    "E [GeV]",
    "N$_{\mathrm{constituent}}$",
]
csts_columns_names = [
    'pt',
    'deta',
    'dphi',
    'd0val',
    'd0err',
    'dzval',
    'dzerr',
]
csts_columns_labels = [
    'p_{T}',
    '\Delta\eta',
    '\Delta\phi',
    '$d_{0}$',
    '$\sigma_{d_{0}}$',
    '$d_{z}$',
    '$\sigma_{d_{z}}$',
]
glob_model_path = Path("/home/users/a/algren/scratch/latn_calib/")

if __name__ == '__main__':
    maxevents = 100_000
    save_figs = True

    # jetclass
    jetclass = h5py.File('/srv/fast/share/rodem/JetClassH5_full_csts/train_100M_combined_0.h5')
    jetclass_scalar = jetclass['jets'][:maxevents]
    label = jetclass['labels'][:4*maxevents] == 0
    mask = jetclass['mask'][:4*maxevents]
    jetclass = jetclass['csts'][:4*maxevents, :, 3:][label][mask[label]]
    jetclass = jetclass[jetclass[:, 0]!=0]

    # # load save latn
    BigBelloNominal = h5py.File(glob_model_path / 'nominal_low_and_high_combined.h5')
    BigBelloNominal_scalar = BigBelloNominal['jets'][:maxevents]
    mask = BigBelloNominal['mask'][:maxevents]
    BigBelloNominal = BigBelloNominal['csts'][:maxevents, :, 3:][mask]
    BigBelloNominal = BigBelloNominal[BigBelloNominal[:, 0]!=0]

    # load save latn
    PSmurr2_smeared = h5py.File(glob_model_path / 'PSmurr2_smeared_low_and_high_combined.h5')
    PSmurr2_smeared_scalar = PSmurr2_smeared['jets'][:maxevents]
    mask = PSmurr2_smeared['mask'][:maxevents]
    PSmurr2_smeared = PSmurr2_smeared['csts'][:maxevents, :, 3:][mask]
    PSmurr2_smeared = PSmurr2_smeared[PSmurr2_smeared[:, 0]!=0]

    for col_nr, label in enumerate(csts_columns_names[3:]):
        counts, (fig, ax, ax_ratio) = plot.plot_hist(
            # jetclass[:, col_nr],
            BigBelloNominal[:, col_nr],
            PSmurr2_smeared[:, col_nr], 
            dist_styles=[
                # {'label': 'JetClass', 'color': 'green'},
                {'label': 'Source', 'color': 'blue'},
                {'label': 'Target', 'color': 'black'},
                ],
            percentile_lst=[13,87],
            full_plot_bool=True,
            ylim=[0.7, 1.3],
            legend_kwargs={'loc': 'lower center'},
            )
        ax_ratio.set_xlabel(csts_columns_labels[3:][col_nr],
                            fontsize=30)

        if save_figs:
            misc.save_fig(fig, f'/home/users/a/algren/work/latn_calib/notebooks/figs/csts_input_{label}.pdf')

    for col_nr, label in enumerate(jets_column_names):
        counts, (fig, ax, ax_ratio) = plot.plot_hist(
            # jetclass[:, col_nr],
            BigBelloNominal_scalar[:, col_nr],
            PSmurr2_smeared_scalar[:, col_nr], 
            dist_styles=[
                # {'label': 'JetClass', 'color': 'green'},
                {'label': 'Source', 'color': 'blue'},
                {'label': 'Target', 'color': 'black'},
                ],
            percentile_lst=[1,99],
            full_plot_bool=True,
            ylim=[0.7, 1.3],
            # legend_kwargs={'loc': 'lower center'},
            )
        ax_ratio.set_xlabel(jets_column_labels[col_nr], fontsize=30)

        if save_figs:
            misc.save_fig(fig, f'/home/users/a/algren/work/latn_calib/notebooks/figs/jets_{label}.pdf')

