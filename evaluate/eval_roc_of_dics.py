"Train final classifier between transport and target"
from copy import deepcopy
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
import numpy as np
from glob import glob
import torch as T
from tqdm import tqdm
import tools.tools.visualization.general_plotting as plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
import matplotlib.pylab as pylab

from run.train_disc import init_classifier, get_run_clf_layers

from tools.tools.hydra_utils import hydra_init
from tools.tools import misc
import matplotlib

def get_roc(classifier, dataset, plot_kwargs={}):
    train_loader = T.utils.data.DataLoader(
        dataset, **{"pin_memory":True, "batch_size":512, 'shuffle': False})
    
    outputs = []
    predicts = []
    inputs = []
    labels=[]
    
    # run inference with the classifier
    with T.no_grad():
        for batch in tqdm(train_loader):
            y = batch[:,classifier.input_dim:].to(classifier.device)
            x = batch[:,:classifier.input_dim].float().to(classifier.device)

            output = classifier(x)
            outputs.append(np.ravel(output[0].detach().cpu().numpy()))
            predicts.append(np.ravel(output[1].detach().cpu().numpy()))
            inputs.append(x.cpu().numpy())
            labels.append(y.cpu().numpy())

    predicts = np.hstack(predicts)
    # proba = trans.sigmoid(predicts)
    inputs = np.concatenate(inputs, 0)
    labels=np.ravel(np.concatenate(labels, 0))

    return predicts, labels, plot.plot_roc_curve(labels, predicts, **plot_kwargs)

logging.basicConfig(level=logging.warning)

if __name__ == "__main__":
    plot.FIG_SIZE = (8, 6)
    plot.FONTSIZE = 20
    plot.LABELSIZE = 20
    plot.LEGENDSIZE = 20
    plot.RATIO_LABELSIZE = 20
    plot.RATIO_FONTSIZE = 20
    plot.KWARGS_LEGEND={"prop":{'size': plot.LEGENDSIZE}, "frameon": False,
                        "title_fontsize":plot.LEGENDSIZE}
    plot.font = {'size': plot.FONTSIZE}

    matplotlib.rc('font', **plot.font)
    params = {'legend.fontsize': plot.LEGENDSIZE,
            'axes.labelsize': plot.LABELSIZE,
            'axes.titlesize':plot.LABELSIZE,
            'xtick.labelsize':plot.LABELSIZE,
            'ytick.labelsize':plot.LABELSIZE}
    pylab.rcParams.update(params)

    config = hydra_init(str(root/"configs/eval_ot.yaml"), verbose=True)
    model_path = config.paths.JetClass.nominal
    
    if config.discriminator.single_layer:
        save_path_to_pic = f"{model_path}/plots/source_all/roc_single_layer_and_auc.pdf"
    elif config.discriminator.use_calib:
        save_path_to_pic = f"{model_path}/plots/source_all/roc_no_OT_and_auc.pdf"
    else:
        save_path_to_pic = f"{model_path}/plots/source_all/roc_and_auc.pdf"
    
    run_clf_layers_options = get_run_clf_layers(model_path)
    run_clf_layers_options = list(run_clf_layers_options.values())
    
    if config.discriminator.use_calib:
        no_OT_calibs = [True]
        use_calib = [True, False]
    else:
        use_calib = [False]
        no_OT_calibs = [False, True]
    fig, ax = plt.subplots(1,1, figsize=(8,8))

    combinations = list(
        itertools.product(run_clf_layers_options, no_OT_calibs,use_calib)
        )[:None]

    nr=0
    # config.discriminator.valid_size=1000
    # config.discriminator.train_size=1000
    for run_clf_layers, no_OT_calib, use_calib in combinations[:]:
        
        # remember to set the correct config values, might be nested
        config.run_clf_layers = run_clf_layers

        cfg_disc = deepcopy(config.discriminator)

        cfg_disc['no_OT_calib'] = no_OT_calib
        cfg_disc['use_calib'] = use_calib
    
        (classifier, dataset, dataset_valid, transport,
         transport_valid, target, target_valid, save_path) = init_classifier(
             config, **cfg_disc)
        
        logging.info(" Load trained classifier")

        try:
            trained_clf = misc.sort_by_creation_time(glob(f'{save_path}/clf_models/*'))[-1]
        except IndexError:
            logging.info(f" No trained classifier found in {save_path}. Pply not trained!")
            continue

        logging.info(f" Loading {trained_clf}")
        classifier.load(trained_clf)
        
        label = f"$z_{{{dataset_valid.shape[-1]-1}}}$, "

        if isinstance(run_clf_layers, list):
            label = "Penult. " + label + '\n'
        
        plot_kwargs= {
            'label': label,
            # 'label': f"{dataset_valid.shape[-1]-1}d: ROC with ",
            'fig': ax, 
            'plot_kwargs': {'color': plot.COLORS[nr]},
            # 'legend_kwargs': {'ncols':2},
            }
        
        no_OT_bool = no_OT_calib and not use_calib
        
        if no_OT_bool:
            plot_kwargs['plot_kwargs']['ls']='--'

        predicts, labels, _ = get_roc(classifier, dataset_valid, plot_kwargs=plot_kwargs)
        
        counts_dict, (fig_marginals, ax_marginals, ax_ratio) = plot.plot_hist(
            predicts[labels==0], predicts[labels==1], full_plot_bool=True,
            style={"range": np.percentile(predicts, [0,100]), "bins": 50},
            dist_styles=[
                ({'color': "blue", 'label': 'Source', "ls": "dotted", "lw":2} if no_OT_bool else
                {'color': "red", 'label': 'Calibrated', "ls":"dashed", "lw":2}),
                {'color': "black", 'label': 'Target'},
            ],
            )
        ax_marginals.set_ylim(0, ax_marginals.get_ylim()[1]*1.2)
        ax_ratio.set_xlabel('Discriminate scores')
        ax_marginals.legend(title=label.replace(",", ""), frameon=False,
                            **{"prop":{'size': 20},"title_fontsize":20})
        
        if config.save_figures:
            title= "with OT" if not no_OT_bool else "without OT"
            marginal_save_path = f"{model_path}/plots/source_all/marginals_{title}_{nr}.pdf"
            misc.save_fig(fig_marginals, marginal_save_path)

        nr += 1 if no_OT_bool else 0

    # Create custom title for dimension indication
    plt.tight_layout()
    legend_handles = [
        Line2D([0], [0], color='black', label='With OT', linestyle='-'),
        Line2D([0], [0], color='black', label='No OT', linestyle='--'),
    ]
    # Add the custom legend
    fig.legend(handles=legend_handles, loc='upper center', ncols=2, bbox_to_anchor=(0.5, 1.025), frameon=False)
    
    if config.save_figures:
        logging.info(f" Save path: {save_path_to_pic}")
        misc.save_fig(fig, save_path_to_pic)


