
"pipeline"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import os
import matplotlib.pyplot as plt
import logging
import numpy as np
import torch as T
import corner
import hydra
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import tools.tools.visualization.general_plotting as plot
import tools.tools.transformations as trans
from tools.tools import misc
from src.pc_classifier import load_lightning_module
from src.datamodule import get_latent_of_datasets,load_model_and_config

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
  path = Path('/srv/beegfs/scratch/groups/rodem/latn_calib/latn_calib_clf'
          '/latn_calib_clf_jetclass_2025_01_31_07_52_34_378267')
  glob_data_path = '/home/users/a/algren/scratch/latn_calib/'
  # load data
  n_batches=100
  layers_removed=-2
  
  data_paths = {
    # 'JetClass': None,
    
    'BigBelloNominal': f'{glob_data_path}/nominal_combined.h5',
    # 'PSmurr2_smeared': f'{glob_data_path}/PSmurr2_smeared_combined.h5',
    'PSmurr1': f'{glob_data_path}/tests/12_12_2024_murr2/QCD_renamedmurr2_0.h5',
    
    # tests
    # 'BigBelloNominal': f'{glob_data_path}/12_12_2024_nominal/QCD_0.h5',
    # 'PSmurr1': f'{glob_data_path}/13_12_2024_PSmurr1/outputQCD_renamedPSmurr1_1k_0.h5',
    # 'PSmurr2': f'{glob_data_path}/13_12_2024_PSmurr2/outputQCD_renamedPSmurr2_1k_0.h5',
    # 'murr2': f'{glob_data_path}/12_12_2024_variation/QCD_renamedmurr2_0.h5',
    # 'PSmurr2_smeared': f'{glob_data_path}/16_12_2024_PSmurr2_smeared/outputQCD_renamedPSmurr2_100k_smeared_0.h5',
  }
  
  # # load save latn
  BigBelloNominal = misc.load_h5(path / 'predictions/BigBelloNominal/train_combined.h5')['latn_10']
  # load save latn
  PSmurr2_smeared = misc.load_h5(path / 'predictions/PSmurr2_smeared/train_combined.h5')['latn_10']
  
  # load data
  data = get_latent_of_datasets(path, data_paths, n_batches=n_batches, layers_removed=layers_removed)
  
  # load model
  model, _ = load_model_and_config(model_path=path, layers_removed=None)
  # sys.exit()
  
  # PSmurr2_smeared['latn'] = trans.log_squash(PSmurr2_smeared['latn'])
  # BigBelloNominal['latn'] = trans.log_squash(BigBelloNominal['latn'])

  # for i,j in data.items():
  #   data[i]['latn'] = trans.log_squash(j['latn'])


  # Loop through each dimension and plot it in a separate subplot
  ctxt_labels = ['pt', 'eta', 'phi', 'mass', 'ntracks']
  
  dist_styles = [{'label': name, 'color': plot.COLORS[nr]} for nr, name in enumerate(data)]
  for i in range(len(ctxt_labels)):

    _, ax = plot.plot_hist(
      *[j['jets'][:, i] for _, j in data.items()], 
                   dist_styles=dist_styles,
                   full_plot_bool=True,
                   ylim=[0.9, 1.1], style={'bins':15})

    ax[-1].set_xlabel(ctxt_labels[i])
    ax[-1].set_ylabel("Normalised counts", fontsize=10)
  
  # plot marginal dist of latn space
  if True:
    number_of_figres_pr_plot = 16
    # Determine the number of dimensions
    if False:
      num_dimensions = data[list(data_paths.keys())[0]]['latn'].shape[1]
    else:
      num_dimensions = BigBelloNominal.shape[1]
      
    n_plots = num_dimensions//number_of_figres_pr_plot
    
    save_path = f'{path}/plots/latn_{num_dimensions}'

    os.makedirs(save_path, exist_ok=True)
    
    num_subplots = ((4, number_of_figres_pr_plot // 4) 
                    if number_of_figres_pr_plot % 4 == 0 
                    else (2, number_of_figres_pr_plot // 2))

    for plot_nr in tqdm(range(n_plots)):

      # Create a figure and a grid of subplots
      scale=2
      fig, axes = plt.subplots(*num_subplots, figsize=(scale*20, scale*10))
      # Loop through each dimension and plot it in a separate subplot
      for i in range(number_of_figres_pr_plot):
        ax = axes[i // num_subplots[-1], i % num_subplots[-1]]  # Determine the position in the grid
        
        col_nr = i+plot_nr*number_of_figres_pr_plot
        
        plot.plot_hist(
          # *[j['latn'][:, i] for _, j in data.items()],
          BigBelloNominal[:, col_nr],
          PSmurr2_smeared[:, col_nr],
          percentile_lst=[0, 100],
                      dist_styles=dist_styles, 
                      ax=ax)
        ax.set_xlabel(f'Latent {col_nr+1}')
        ax.set_ylabel("Normalised counts", fontsize=10)
        ax.set_yscale('log')
      misc.save_fig(fig, f'{save_path}/marginal_{plot_nr}.pdf')
  
  # plot background rejection
  if layers_removed is None:
    plt.figure()
    col = 0
    bkg_rejs = np.linspace(0.2, 1, 101)[:,None]
    for name in data:
      mask_bkg = data[name]['labels']==0
      ratio = (1-sigmoid(data[name]['latn'][mask_bkg, col])>=bkg_rejs).sum(1)/mask_bkg.sum()
      
      plt.plot(bkg_rejs, ratio, label=name)
    plt.legend(frameon=False)
    plt.xlabel('Classifier output')
    plt.ylabel('Background rejection')
  
  if False:
    # train random forest classifier
    bkg = data['PSmurr2_smeared']['latn']
    bkg = np.hstack((bkg, np.zeros((len(bkg),1))))

    sig = data['BigBelloNominal']['latn']
    sig = np.hstack((sig, np.ones((len(sig),1))))

    data_clf = np.vstack((bkg, sig))
    train, test = train_test_split(data_clf, test_size=0.33, random_state=42)

    forest = RandomForestClassifier(n_estimators=200, max_depth=10).fit(train[:, :-1], train[:, -1:])
    plot.plot_roc_curve(test[:,-1:], forest.predict_proba(test[:,:-1])[:, 1])
    plot.plot_roc_curve(train[:,-1:], forest.predict_proba(train[:,:-1])[:, 1])

  if False:
    #PCA on input

    n_components=16
    pca = PCA(n_components=16)
    # target = data['PSmurr2_smeared']['latn'][:, :n_components]
    # source = data['BigBelloNominal']['latn'][:, :n_components]
    target = PSmurr2_smeared
    source = BigBelloNominal

    # source = (source-target.mean(0))/target.std(0)
    # target = (target-target.mean(0))/target.std(0)
    
    pca_target = pca.fit_transform(target)
    pca_transport = pca.transform(source)

    # for i in range(n_components):
    #     plot.plot_hist(pca_target[:,i], pca_transport[:,i],
    #                 full_plot_bool=True, ylim=[0.9, 1.1],
    #                 style={'bins': 50, }, dist_styles=dist_styles)

    logging.info(" Corner plot!")
    kwargs = {'plot_density': False, 'plot_datapoints': False,
              'bins':30}
    for dist in [[source, target], [pca_transport, pca_target]]:
      # corner plots between pythia and herwig
      col_size = None
      ranges = [tuple(np.percentile(dist[0][:, i], [0.1,99.9])) for i in range(dist[0].shape[1] if col_size is None else col_size)]
      labels = [f'Latent {i+1}' for i in range(dist[0].shape[1] if col_size is None else col_size)]
      figure=None
      kwargs['range'] = ranges

      figure= corner.corner(dist[0][...,:col_size], color = 'red', **kwargs)

      corner.corner(dist[1][...,:col_size], color = 'blue',
                  fig=figure,labels=labels, labelpad=0.1,
                  **kwargs)
  if False:
    # DEPRECATED
    # plot dependence of classifier output on pt and eta 
    # Create a figure and a grid of subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # 4x4 grid of subplots

    # Loop through each dimension and plot it in a separate subplot
    dist_styles = [{'color': 'blue', 'label': 'Pythia'},
                    {'color': 'red', 'label': 'Herwig'}]
    plot.plot_hist(outputs_pythia[0][:, 0], outputs_herwig[0][:, 0], dist_styles=dist_styles,
                    ax=ax)
    ax.set_title(f'Output probabilities')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    dist_styles = [{'label': 'Pythia'}]

    # plot pythia probability in different pt bins
    pt_bins = [500, 525, 550]
    
    # Create a figure and a grid of subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # 4x4 grid of subplots
    for nr, pt_bin in enumerate(pt_bins[:-1]):
      mask = (jet_pythia['pt'].values >= pt_bins[nr]) & (jet_pythia['pt'].values < pt_bins[nr+1])
      dist_styles[0]['label'] = f'Pythia pT: {pt_bins[nr]}-{pt_bins[nr+1]}'

      # Loop through each dimension and plot it in a separate subplot
      plot.plot_hist(outputs_pythia[0][:, 0][mask], dist_styles=dist_styles,
                      ax=ax)

      ax.set_title(f'Output probabilities')

      # Adjust layout to prevent overlap
      plt.tight_layout()

    # plot pythia probability in different pt bins
    eta_bins = [0, 0.7, 1.8]
    # Create a figure and a grid of subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # 4x4 grid of subplots
    for nr, pt_bin in enumerate(eta_bins[:-1]):
      mask = (np.abs(jet_pythia['eta'].values) >= eta_bins[nr]) & (np.abs(jet_pythia['eta'].values) < eta_bins[nr+1])
      dist_styles[0]['label'] = f'Pythia eta: {eta_bins[nr]}-{eta_bins[nr+1]}'

      # Loop through each dimension and plot it in a separate subplot
      plot.plot_hist(outputs_pythia[0][:, 0][mask], dist_styles=dist_styles,
                      ax=ax)

      ax.set_title(f'Output probabilities')

      # Adjust layout to prevent overlap
      plt.tight_layout()

  if False:
    # corner plots between pythia and herwig
    col_size = None
    ranges = [tuple(np.percentile(latn_pythia[:, i], [1,99])) for i in range(latn_pythia.shape[1] if col_size is None else col_size)]
    labels = [f'Latent {i+1}' for i in range(latn_pythia.shape[1] if col_size is None else col_size)]
    figure=None
    if len(latn_herwig)!=0 and False:
      # flipping to the other corner
      figure= corner.corner(latn_herwig[...,:col_size][:, ::-1], 
                            color = 'red',reverse = True, range=ranges[::-1])
    else:
      figure= corner.corner(latn_herwig[...,:col_size], 
                            color = 'red', range=ranges,
                            plot_datapoints = False,
                            plot_density=False)

    corner.corner(latn_pythia[...,:col_size], color = 'blue',
                  range=ranges,fig=figure,labels=labels, labelpad=0.1,
                  plot_datapoints = False, plot_density=False)
    
    # corner plots between pythia signal and pythia background
    col_size = None
    ranges = [tuple(np.percentile(latn_pythia[:, i], [0,100])) for i in range(latn_pythia.shape[1] if col_size is None else col_size)]
    labels = [f'Latent {i+1}' for i in range(latn_pythia.shape[1] if col_size is None else col_size)]
    
    mask_sig = y_pythia==1

    figure=None
    if len(latn_herwig)!=0 and False:
      # flipping to the other corner
      figure= corner.corner(latn_herwig[...,:col_size][:, ::-1][~mask_sig], 
                            color = 'red', reverse = True, range=ranges[::-1])
    else:
      figure= corner.corner(latn_herwig[...,:col_size][~mask_sig], 
                            color = 'red', range=ranges)

    corner.corner(latn_pythia[...,:col_size][mask_sig], color = 'blue',
                  range=ranges,fig=figure,labels=labels, labelpad=0.1)
    
    
