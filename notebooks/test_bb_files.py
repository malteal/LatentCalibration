
import pyrootutils

root = pyrootutils.setup_root(search_from=".", pythonpath=True)

import torch as T
import sys
import numpy as np
import hydra
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

from tools.tools import misc
import tools.tools.visualization.general_plotting as plot
import h5py
from notebooks.plot_classifier_features import jets_column_names


label_names = [
    "label_QCD",
    "label_Tbl",
    "label_Tbqq",
    "label_Wqq ",
    "label_Zqq ",
    "label_Hbb ",
    "label_Hcc ",
    "label_Hgg ",
    "label_H4q ",
    "label_Hqql"]

if __name__ == "__main__":
    # rsync -r outputQCD_renamedPSmurr2_smeared_* algren@login1.baobab.hpc.unige.ch:/home/users/a/algren/scratch/latn_calib/BB_nominal_ttbar
    data_path = "/home/users/a/algren/scratch/latn_calib/BB_nominal_ttbar_combined.h5"
    hdf = h5py.File(data_path, "r")
    
    data_path = "/home/users/a/algren/scratch/latn_calib/PSmurr2_smeared_combined.h5"
    QCD_hdf = h5py.File(data_path, "r")
    
    labels = hdf['labels'][:]
    # qcd_labels = QCD_hdf['labels'][:500_000]
    # labels = np.hstack([labels, qcd_labels])
    
    plt.figure()
    plt.hist(labels, bins=10, alpha=0.5, label='Labels Histogram',range=[0,9])
    plt.legend(frameon=False)
    # plt.yscale('log')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')

    sys.exit()
    for nr, col in enumerate(jets_column_names):
        plt.figure(figsize=(10, 6))
        style = {'histtype': 'step', 'alpha': 0.5, 'bins': 30}
        for i in range(6):
            if any(labels==i):
                _,bins,_ = plt.hist(
                    hdf["jets"][:, nr][labels==i],
                    stacked=True, label=label_names[i], **style
                )
                style['bins'] = bins
        plt.legend(frameon=False)
        plt.yscale('log')
        plt.xlabel(col)
    