'evaluate the classifier performance'
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
from pathlib import Path
import hydra
import numpy as np
import torch as T
from tqdm import tqdm
import corner
import matplotlib.lines as mlines

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from run.train_flow import load_flow_from_path
from src.datamodule import flow_norm
from tools.tools import misc
from tools.tools.visualization import general_plotting as plot


if __name__ == "__main__":
    glob_path = Path("/srv/beegfs/scratch/groups/rodem/latn_calib/latn_calib_clf")
    
    model_path = glob_path / "flow_test_jetclass_2025_02_23_09_59_06_974389"
    
    config = misc.load_yaml(f"{model_path}/.hydra/config.yaml")
    key = config.data.val_set.key
    
    # load data
    val_data_path = Path(config.data.val_set.path)
    # replace str in Path
    out_domain_path = Path(config.data.val_set.path.replace('PSmurr2_smeared', 'BigBelloNominal'))
    val_data = misc.load_h5(val_data_path)[key]
    out_domain_val = misc.load_h5(out_domain_path)[key]

    maxevents = 100_000
    gauss_val_data = flow_norm(val_data[:maxevents], model_path)
    gauss_out_domain = flow_norm(out_domain_val[:maxevents], model_path)

    # gauss_val_data_back = flow_norm(model_path, gauss_val_data, to_base=False)
    # gauss_out_domain_back = flow_norm(model_path, gauss_out_domain, to_base=False)
    
    fig, ax = plt.subplots(2, gauss_val_data.shape[1], 
                           figsize=(gauss_val_data.shape[1]*8,8),
                           sharey="row")
    dist_styles=[{'color':'b', 'label':'PSmurr2_smeared'},
                {'color':'r', 'label':'BigBelloNominal'}]

    for i in range(gauss_val_data.shape[1]):
        plot.plot_hist(
            val_data[:,i],
            out_domain_val[:,i],
            ax=ax[0,i],
            dist_styles=dist_styles,
            style={'bins': 100},
            percentile_lst=[0, 100])

        plot.plot_hist(
            gauss_val_data[:,i],
            gauss_out_domain[:,i],
            ax=ax[1,i],
            dist_styles=dist_styles,
            style={'bins': 100},
            percentile_lst=[0, 100])
        
        ax[1,i].set_yscale('log')
        ax[0,i].set_yscale('log')
        
    
    if False: # plot corner plot after flow transformation
        # corner plots between pythia and herwig
        col_size = None
        quantile = [0.1, 0.99]

        ranges = [tuple(np.quantile(gauss_val_data[:, i], quantile)) for i in range(gauss_val_data.shape[1] if col_size is None else col_size)]
        labels = [f'Latent {i+1}' for i in range(gauss_val_data.shape[1] if col_size is None else col_size)]

        kwargs = {'plot_density': False, 'plot_datapoints': False, 'range': ranges, 'fill_contours':False,
                    'no_fill_contours':False,
                    'bins':30}
        
        # kwargs['levels'] = [0.05, 0.5, 0.8, 0.9, 0.95]
        
        figure= corner.corner(gauss_val_data[...,:col_size], 
                            color = 'red', **kwargs)

        figure = corner.corner(gauss_out_domain[...,:col_size], 
                            color = 'blue',
                            fig=figure,labels=labels, labelpad=0.1, **kwargs)

        blue_line = mlines.Line2D([], [], color='blue', label='Transport', lw=8*len(labels)/4)
        red_line = mlines.Line2D([], [], color='red', label='Target',
                                    lw=8*len(labels)/4)

        figure.legend(handles=[blue_line,red_line],
                        bbox_to_anchor=(0., 1.0, 1., .0), 
                    fontsize=50*len(labels)/4,
                        loc='upper right',
                        frameon=False)

    elif True: # train classifier
        from run.train_disc import init_classifier
        from tools.tools.hydra_utils import hydra_init
        from sklearn.model_selection import train_test_split



        config = hydra_init(str(root/"configs/eval_ot.yaml"), verbose=True)

        (classifier, dataset, dataset_valid, transport,
        transport_valid, target, target_valid, save_path) = init_classifier(
            config, **config.discriminator)

        classifier.save_path = classifier.save_path.replace('classifier', 'classifier_flow_features')

        val_data = np.concatenate([val_data, np.ones((len(val_data),1))],1)

        out_domain_val = np.concatenate([out_domain_val, np.zeros((len(out_domain_val),1))],1)
        
        datasets = np.concatenate([val_data, out_domain_val], 0)
        # Assuming X is your feature matrix and y is your target vector
        dataset, dataset_valid = train_test_split(datasets, test_size=0.2, random_state=42)

        classifier.create_loaders(dataset, valid=dataset_valid)

        classifier.set_optimizer(True, lr=1e-4)

        classifier.run_training(n_epochs=config.discriminator.n_epochs,
                                standard_lr_scheduler=True)

        classifier.plot_log()

        classifier.plot_auc(f'{classifier.save_path}/plots_/auc.png', plot_roc=True)
