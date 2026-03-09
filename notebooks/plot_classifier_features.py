


import pyrootutils

root = pyrootutils.setup_root(search_from=".", pythonpath=True)

import torch as T
import sys
import hydra
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

from tools.tools import misc
import tools.tools.visualization.general_plotting as plot

dataloader = {
    '_target_': 'src.datamodules.hdf_streamer.StreamModule',
    'name': 'jetclass',
    'train_set': {
        'path': '/srv/fast/share/rodem/JetClassH5_full_csts/train_100M_*',
        'n_jets': 4_000_000,
        '_target_': 'src.datamodules.hdf_streamer.JetHDFStream',
        '_partial_': True,
        'n_classes': 10,
    },
    'val_set': {
        'path': '/srv/fast/share/rodem/JetClassH5_full_csts/val_5M_combined.h5',
        'n_jets': 100_000,
        '_target_': 'src.datamodules.hdf_streamer.JetHDFStream',
        '_partial_': True,
        'n_classes': 10,
    },
    'test_set': {
        'path': '/srv/fast/share/rodem/JetClassH5_full_csts/val_5M_combined.h5',
        'n_jets': 100_000,
        '_target_': 'src.datamodules.hdf_streamer.JetHDFStream',
        '_partial_': True,
        'n_classes': 10,
    },
    'batch_size': 1000,
    'num_workers': 4,  # Speed testing show that 3 workers is enough even with augs
    'transforms': {
        '_target_': 'src.datamodules.preprocessing.batch_preprocess',
        '_partial_': True,
        'fn':
            [{'_target_': 'joblib.load',
            'filename': '/home/users/a/algren/work/latn_calib/resources/cst_quant.joblib'
            }],
        'scalar_fn': {
            '_target_': 'joblib.load',
            'filename': '/home/users/a/algren/work/latn_calib/resources/jet_quant.joblib',
        },
    'inpt_name': "x"
    },
}

jets_column_names = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    'jet_energy',
    "jet_nparticles",
]
jets_column_labels = [
    "$p_{\mathrm{T}}$ [GeV]",
    "$\eta$",
    "$\phi$",
    "E [GeV]",
    "N$_{\mathrm{particles}}$",
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
    '$p_{\mathrm{T}}$ [GeV]',
    '$\Delta\eta$',
    '$\Delta\phi$',
    '$d_{0}$ [mm]',
    '$\sigma_{d_{0}}$ [mm]',
    '$d_{z}$ [mm]',
    '$\sigma_{d_{z}}$ [mm]',
]
if __name__ == '__main__':
    config = hydra.utils.instantiate(dataloader)
    save_figs = True
    size = 100_000 # if set to None, the quantile transformers will be saved
    
    train_loader = config.val_dataloader()
    
    # quick test - not in use
    for i in train_loader:
        break
    
    dataset = train_loader.dataset
    scalar_fn = dataset.transforms[0].keywords['scalar_fn']
    csts_fn = dataset.transforms[0].keywords['fn'][0]
    
    jets = dataset.file['jets'][: size]
    mask = dataset.file['mask'][: size]
    csts_id = dataset.file['csts_id'][: size]
    csts = dataset.file['csts'][: size]
    
    neut_mask = (csts_id == 0) | (csts_id == 2)
    # csts[:, -4:][neut_mask] = T.nan

    if size is None:
        # calculate scaler for the point cloud input features
        
        # Make a quantile transformer
        csts_fn = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=500,
            subsample=len(csts) + 1,
        )
        csts_fn.fit(csts)

        # Make a quantile transformer for the jets
        scalar_fn = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=500,
            subsample=len(jets) + 1,
        )
        scalar_fn.fit(jets)
        if False:
            dump(csts_fn, "cst_quant.joblib")
            dump(scalar_fn, "jet_quant.joblib")
            sys.exit()


    for col_nr, label in enumerate(jets_column_names):
        fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True)
        counts, ax[0] = plot.plot_hist(jets[:, col_nr], ax=ax[0],
                                   dist_styles=[{'color': 'r', 'label': 'Raw'}],
                                   legend_bool=True,
                                   )
        counts, ax[1] = plot.plot_hist(scalar_fn.transform(jets)[:, col_nr], ax=ax[1],
                                   dist_styles=[
                                       {'color': 'r', 'label': 'QuantileTransform', 'ls': 'dashed'}
                                       ],
                                   legend_bool=True,
                                   )
        ax[0].set_xlabel(jets_column_labels[col_nr])
        ax[1].set_xlabel(jets_column_labels[col_nr])
        ax[0].legend().remove()
        ax[1].legend().remove()
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
        plt.tight_layout()
        if save_figs:
            misc.save_fig(fig, f'/home/users/a/algren/work/latn_calib/notebooks/figs/{label}.pdf',
                          tight_layout=False)

    for col_nr, label in enumerate(csts_columns_names):
        if col_nr <= 2:
            csts_mask = mask
            percentile_lst=[1,99]
        else:
            csts_mask = mask & ~neut_mask
            if 'err' in label:
                percentile_lst=[1,99]
            else:
                percentile_lst=[5,95]
        fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True)
        counts, _ = plot.plot_hist(csts[csts_mask][:, col_nr], ax=ax[0],
                                   dist_styles=[{'color': 'r', 'label': 'Raw'}],
                                   percentile_lst=percentile_lst
                                   )
        counts, _ = plot.plot_hist(csts_fn.transform(csts[csts_mask])[:, col_nr], ax=ax[1],
                                   dist_styles=[{'color': 'r', 'label': 'QuantileTransform', 'ls': 'dashed'}],
                                   percentile_lst=percentile_lst
                                   )
        ax[0].set_xlabel(csts_columns_labels[col_nr])
        ax[1].set_xlabel(csts_columns_labels[col_nr])
        ax[0].legend().remove()
        ax[1].legend().remove()
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
        plt.tight_layout()
        if save_figs:
            misc.save_fig(fig, f'/home/users/a/algren/work/latn_calib/notebooks/figs/{label}.pdf',
                          tight_layout=False)
