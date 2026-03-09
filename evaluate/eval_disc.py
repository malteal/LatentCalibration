"Train final classifier between transport and target"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
import numpy as np
from glob import glob
import corner
import torch as T
from tqdm import tqdm
from sklearn.decomposition import PCA
import tools.tools.visualization.general_plotting as plot


from run.train_disc import init_classifier

import tools.tools.transformations as trans
from tools.tools.hydra_utils import hydra_init
from tools.tools import misc

log = logging.getLogger(__name__)

if __name__ == "__main__":

    config = hydra_init(str(root/"configs/eval_ot.yaml"), verbose=True)
    no_OT_calib=config.discriminator.no_OT_calib

    (classifier, dataset, dataset_valid, transport,
     transport_valid, target, target_valid, save_path) = init_classifier(
         config, **config.discriminator
         )
    
    logging.info(" Load trained classifier")
    trained_clf = misc.sort_by_creation_time(glob(f'{save_path}/clf_models/*'))[-1]

    logging.info(f" Loading {trained_clf}")
    classifier.load(trained_clf)
    
    train_loader = T.utils.data.DataLoader(
        dataset_valid, **{"pin_memory":True, "batch_size":512, 'shuffle': False})
    
    outputs = []
    predicts = []
    inputs = []
    labels=[]
    
    # run inference with the classifier
    with T.no_grad():
        for batch in tqdm(train_loader):
            y = batch[:,classifier.input_dim:].to(classifier.device)
            x = batch[:,:classifier.input_dim].float().to(classifier.device)
            if False:
                mask = y==1
                
                x = x[mask.flatten()]
                y = y[mask]
                if x.sum()==0:
                    continue

            output = classifier(x)
            outputs.append(np.ravel(output[0].detach().cpu().numpy()))
            predicts.append(np.ravel(output[1].detach().cpu().numpy()))
            inputs.append(x.cpu().numpy())
            labels.append(y.cpu().numpy())

    outputs = np.hstack(outputs)
    predicts = np.hstack(predicts)
    proba = trans.sigmoid(predicts)
    weights = proba/(1-proba+1e-10)
    inputs = np.concatenate(inputs, 0)
    labels=np.ravel(np.concatenate(labels, 0))
    
    plot.plot_hist(weights, percentile_lst=[0,99])
    plot.plot_hist(proba[labels==0], proba[labels==1], percentile_lst=[0,100],
                   dist_styles=[{'label': 'bkg', 'color': 'red'},
                                {'label': 'sig', 'color': 'blue'}],
                   legend_kwargs={'title': 'source vs target'})
    
    kldiv = np.log(proba[labels==1]/(1-proba[labels==1]))
    plot.plot_hist(kldiv[kldiv<1e10], percentile_lst=[0,100],
                   dist_styles=[{'label': 'sig', 'color': 'blue'}],
                   legend_kwargs={'title': 'OT(source) vs target'})
    
    # weights = np.clip(weights, 0, 10)
    print(np.log(np.clip(weights, None, 1e10)).mean())
    # mask = outputs<np.max(outputs)

    target_mask = dataset_valid[:, -1] == 1
    
    dist_styles = [
        {'label': 'Target', 'color': 'red'},
        ({'label': 'Source', 'color': 'blue'} if no_OT_calib 
        else {'label': 'Calibrated', 'color': 'blue'}),
        {'label': 'Classifier', 'color': 'green'}]

    for i in range(4):
        counts, fig = plot.plot_hist(dataset_valid[target_mask,i], inputs[labels==0][:,i], inputs[labels==0][:,i],
                       full_plot_bool=True, ylim=[0.9, 1.1],
                       weights = [np.ones(target_mask.sum())/target_mask.sum(), np.ones(sum(labels==0))/sum(labels==0), weights[labels==0]/weights[labels==0].sum()],
                       style={'bins': 50, }, dist_styles=dist_styles,
                       normalise=False
                       )
        fig[-1].set_xlabel(f'latn{i}')
        


    if False:
        n_components=16
        if True:
            pca = PCA(n_components=n_components)
            pca_target = pca.fit_transform(target_valid[:, :n_components])
            pca_transport = pca.transform(transport_valid[:, :n_components])
            # for i in range(n_components):
            #     plot.plot_hist(pca_target[:,i], pca_transport[:,i],
            #                 full_plot_bool=True, ylim=[0.9, 1.1],
            #                 style={'bins': 50, }, dist_styles=dist_styles)
        else:
            pca_target = target_valid[:, :n_components]
            pca_transport = transport_valid[:, :n_components]

        pca_transport = pca_transport[proba[labels==0]<0.01]
        pca_target = pca_target[proba[labels==1]>0.99]

        logging.info(" Corner plot!")

        # corner plots between pythia and herwig
        col_size = None
        ranges = [tuple(np.percentile(pca_target[:, i], [0.1,99.9])) for i in range(pca_target.shape[1] if col_size is None else col_size)]
        axis_labels = [f'Latent {i+1}' for i in range(pca_target.shape[1] if col_size is None else col_size)]
        figure=None
        kwargs = {'plot_density': False, 'plot_datapoints': False, 'range': ranges,
                    'bins':20}
        
        weights = [np.ones(len(pca_target))/len(pca_target),
                np.ones(len(pca_transport))/len(pca_transport)]
        
        figure= corner.corner(pca_target[...,:col_size], color = 'red', weights=weights[0], **kwargs)

        corner.corner(pca_transport[...,:col_size], color = 'blue',
                    fig=figure,labels=axis_labels, labelpad=0.1,
                    weights=weights[1], **kwargs)

        logging.info(" Job finished!")
