"Train final classifier between transport and target"
import pyrootutils
from sklearn.discriminant_analysis import StandardScaler

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
import joblib
import os
import numpy as np
import torch as T
from glob import glob

from run.predict_ot import PredictionHandler
from src.datamodule import get_classifier_output

from tools.tools.discriminator import DenseNet
from tools.tools.hydra_utils import hydra_init
from tools.tools import misc


def get_run_clf_layers(ot_path):
    # should be updated with the one in smk
    if '128_only' in ot_path:
        run_clf_layers = {'null': None,
                        #   '8-6': [8, 6],'8-4': [8, 4], 
                          '8-2': [8, 2], '8': 8}
    elif '128' in ot_path:
        run_clf_layers = {'null': None, '8-2': [8, 2], '8': 8}
    else:
        run_clf_layers = {'null': None, '2': 2}

    return run_clf_layers

def init_classifier(config, no_OT_calib:bool=False, single_layer:bool=False,
                    use_calib:bool=False, **kwargs):

    selection = config['type_to_eval']
    gen_name = config["evaluation_sample"]

    # get paths to model
    path_to_eval = config['paths']
    nominal_path = path_to_eval[selection][config['name_to_eval']]
    
    logging.info("initializing save path!")
    save_path = f'{nominal_path}/classifier/'

    ######### get nominal prediction #########
    logging.info(" Setting up dataset!")
    transport_handler = PredictionHandler(nominal_path, gen_name,
                                          config["bkg_scaler_str"],
                                          force_prediction = False,
                                          allow_prediction = False)

    data = transport_handler.load_prediction()
    run_clf_layers = config.get("run_clf_layers")

    if config.get("run_clf_layers"):
        # get final output distribution
        data = get_classifier_output(data, nominal_path, 
                                     config.get("run_clf_layers"))

    # define columns in the dataset
    columns = data['eval_transport'].keys()

    clf_layer_str = 'null' if run_clf_layers is None else str(run_clf_layers) if isinstance(run_clf_layers, int) else '-'.join([str(i) for i in run_clf_layers])

    # get classifier layer name
    # discriminator on the output distribution
    save_path = save_path.replace('classifier', f'classifier_{clf_layer_str}')
    # save_path = save_path.replace('classifier', f'classifier_{len(columns)}_{clf_layer_str}')
    
    # change classifier name
    if no_OT_calib:
        save_path = save_path.replace('classifier', 'classifier_no_OT')
    
    # get latn distribution
    if no_OT_calib and not use_calib:
        logging.info(" Get source sample")
        transport = data['source'][columns].values
        transport_valid = data['source_valid'][columns].values
    else:
        logging.info(" Get calibration sample")
        transport = data['eval_transport'][columns].values
        transport_valid = data['eval_transport_valid'][columns].values

    target = data['Data'][columns].values
    target_valid = data['Data_valid'][columns].values

    logging.info(" Setting up classifier!")
    dense_kwargs={'N': 512, 'device': 'cuda',
                  'save_path': save_path, 'clip_bool': True,
                  'activation_str': 'silu',
                  'sigmoid': False, 'n_layers': 4, 'drp': 0,
                  'early_stoppping': 100, 'batchnorm': False # it will use layernorm then
                  }

    classifier = DenseNet(transport.shape[-1], **dense_kwargs)

    if single_layer:
        logging.info(" Defining single linear layer!")

        # redefine network to single
        classifier.network = T.nn.Linear(transport.shape[-1], 1).to('cuda')
        
        # redefine save path    
        save_path = save_path.replace('classifier', 'classifier_single_layer')
        classifier.save_path = save_path

    transport = np.concatenate([transport,np.zeros((len(transport),1))],1)[:kwargs.get('train_size')]
    transport_valid = np.concatenate([transport_valid,np.zeros((len(transport_valid),1))],1)[:kwargs.get('valid_size')]

    target = np.concatenate([target,np.ones((len(target),1))],1)[:kwargs.get('train_size')]
    target_valid = np.concatenate([target_valid,np.ones((len(target_valid),1))],1)[:kwargs.get('valid_size')]
    
    # define dataset by merging
    dataset = np.concatenate([transport, target], 0)
    dataset_valid = np.concatenate([transport_valid, target_valid], 0)

    # make dir
    os.makedirs(save_path, exist_ok=True)
    
    # define scaler 
    if os.path.exists(f'{save_path}/scaler.pkl'):
        # load scaler
        scaler = joblib.load(f'{save_path}/scaler.pkl')
        
        dataset[:, :-1] = scaler.transform(dataset[:, :-1])
        dataset_valid[:, :-1] = scaler.transform(dataset_valid[:, :-1])
    else:
        scaler = StandardScaler()
        dataset[:, :-1] = scaler.fit_transform(dataset[:, :-1])
        dataset_valid[:, :-1] = scaler.transform(dataset_valid[:, :-1])

        # save scaler
        joblib.dump(scaler, f'{save_path}/scaler.pkl')
        
    trained_clf = misc.sort_by_creation_time(glob(f'{save_path}/clf_models/*'))

    if len(trained_clf) != 0:
        logging.info(f" Loading {trained_clf[-1]}")
        classifier.load(trained_clf[-1])
        
        # reset loss data log
        classifier.loss_data = {i: []
            for i,j in classifier.loss_data.items()
        }

    return classifier, dataset, dataset_valid, transport, transport_valid, target, target_valid, save_path


log = logging.getLogger(__name__)

if __name__ == "__main__":

    config = hydra_init(str(root/"configs/eval_ot.yaml"), verbose=True)

    (classifier, dataset, dataset_valid, transport,
     transport_valid, target, target_valid, save_path) = init_classifier(
         config, **config.discriminator)

    classifier.create_loaders(dataset, valid=dataset_valid)
    logging.info(f" Save path: {save_path}")

    logging.info(" Setup optimizer")
    classifier.set_optimizer(True, lr=1e-4)

    logging.info(" Train classifier")
    classifier.run_training(n_epochs=config.discriminator.n_epochs,
                            standard_lr_scheduler=True)
    
    logging.info(" Plot performance")
    classifier.plot_log()
    
    classifier.plot_auc(f'{classifier.save_path}/plots_/auc.png', plot_roc=True)

    logging.info(" Job finished!")
