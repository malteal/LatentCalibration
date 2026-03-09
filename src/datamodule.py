"training flows from template generator"

import logging
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from functools import partial
from omegaconf import ListConfig
from tqdm import tqdm
from typing import Union
import pandas as pd
from pathlib import Path
from glob import glob
import numpy as np
import os
from copy import deepcopy
import torch as T
import joblib
import hydra
try:
    import onnxruntime as ort
except:
    ort = None
from zipfile import BadZipFile

# ot-framework imports
from src.pc_classifier import load_lightning_module
from src.utils import jet_class_labels
try:
    from run.train_flow import load_flow_from_path
except ImportError:
    pass

# external imports
from tools.tools.physics.detector_coords import jet_variables
from tools.tools import misc


def get_classifier_output(data:dict, ot_path:str, layers_removed:int=None, keys:list=None):
    # get all data keys if keys not given
    if keys is None:
        keys = data.keys()
    
    # get config of OT
    transport_config = misc.load_yaml(f'{ot_path}/.hydra/config.yaml')
    
    # get path of classifier
    classifier_path = transport_config['data']['source_sample']['data'][0]['path']
    classifier_path = classifier_path.split('predictions')[0]
    
    # load classifier
    clf_model, _ = load_model_and_config(classifier_path)
    
    # remove layers
    if isinstance(layers_removed, (list, ListConfig)):
        clf_model.classifier.network = clf_model.classifier.network[-layers_removed[0]:-layers_removed[1]]
    else:
        clf_model.classifier.network = clf_model.classifier.network[-layers_removed:]

    #define dict for output
    clf_output={}
    
    # define columns
    columns = data['eval_transport'].columns
    
    # run over keys: data, target, transport etc.
    for key in keys:
        # define input
        target = T.tensor(data[key][data['eval_transport'].columns].values,
                    device='cuda').float()
        
        # eval classifier
        clf_out = clf_model.classifier(target).detach().cpu().numpy()
        
        # save output as dataframe
        if clf_out.shape[-1] == 10:
            # specific for JetClass
            columns = jet_class_labels()

        if clf_out.shape[-1] == len(columns):
            clf_output[key] = pd.DataFrame(clf_out, columns=columns)
        else:
            columns=[f'latn{i}' for i in range(clf_out.shape[-1])]
            clf_output[key] = pd.DataFrame(clf_out, columns=columns)

    return clf_output

def load_onnx_in_inference(path_to_onnx: str, device: str = 'cpu'): # -> ort.InferenceSession:
    """
    Load an ONNX model for inference with specified session options.

    Parameters:
    path_to_onnx (str): Path to the ONNX model file.

    Returns:
    ort.InferenceSession: The ONNX Runtime inference session.
    """
    so = ort.SessionOptions()
    providers=['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    
    # Set these to 1 for reproducibility
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1

    ort_sess = ort.InferenceSession(path_to_onnx, providers=providers, sess_options=so)
    
    return ort_sess


def get_data(paths: Union[list, str], num_data: int = None, pad: bool = True, ncol: int = 3,
             center_jet: bool = True, normalize_pt: bool = True):
    """
    Load and process data from .npz files.

    Parameters:
    paths (list): List of file paths or a single path pattern to .npz files.
    num_data (int): Number of data points to return. If -1, return all data. Default is -1.
    pad (bool): Whether to pad the arrays to the same length along axis 1. Default is True.
    ncol (int): Number of columns to extract from the arrays. Default is 3.
    center_jet (bool): Whether to center the jet by subtracting the average y and phi. Default is True.
    normalize_pt (bool): Whether to normalize the pt values. Default is True.

    Returns:
    tuple: A tuple containing:
        - X (numpy.ndarray): The processed feature array.
        - y (numpy.ndarray): The concatenated label array.
    """
    if isinstance(paths, str):
        if '.npz' not in paths:
            paths = glob(paths if '*' in paths else f'{paths}/*.npz')
        else:
            paths = [paths]

    Xs = []
    ys = []
    size=0
    for path in tqdm(paths, total=len(paths), desc='Loading data', leave=False):
        # Load file and append arrays
        try:
            with np.load(path) as f:

                Xs.append(f['X'])

                ys.append(f['y'])

                size+=f['X'].shape[0]

        except BadZipFile:
            logging.warning(f"BadZipFile: {path}")

        # Chop down to specified amount of data
        if isinstance(num_data, int) and num_data > 0 and size>=num_data:
            break
    
    # Get X array by concatenating arrays
    if pad:
        max_len_axis1 = max([X.shape[1] for X in Xs])
        X = np.vstack([_pad_events_axis1(x[..., :ncol], max_len_axis1) for x in Xs])
    else:
        X = np.asarray([x[x[:, 0] > 0, :ncol] for X in Xs for x in X], dtype='O')

    # Get y array by concatenating arrays
    y = np.concatenate(ys)
    
    # Chop down to specified amount of data
    if isinstance(num_data, int) and num_data > 0 and size>=num_data:
        X = X[:num_data]
        y = y[:num_data]

    # define mask
    mask = X[..., 0] > 0

    # calculate jet features
    jet_features = jet_variables(X[..., [1,2,0]], mask=mask)

    if center_jet:
        for x in tqdm(X, desc='Centering jets', leave=False):
            _mask = x[:, 0] > 0
            yphi_avg = np.average(x[_mask, 1:3], weights=x[_mask, 0], axis=0)
            x[_mask, 1:3] -= yphi_avg
            if normalize_pt:
                x[_mask, 0] /= x[:, 0].sum()

    return X, y, jet_features

def load_latent_space(path_to_samples: Union[str, list], path_to_onnx: str=None, num_data: int = 100_000):
    """
    Load latent space data from samples and run inference using an ONNX model.

    Parameters:
    path_to_samples (Union[str, list]): Path to the sample files. Can be a string with a glob pattern or a list of file paths.
    path_to_onnx (str): Path to the ONNX model file.
    num_data (int, optional): Number of data points to load from each sample file. If None, all data points are loaded.

    Returns:
    tuple: A tuple containing:
        - outputs (dict): A dictionary where keys are output indices and values are concatenated inference results.
        - Y (np.ndarray): Concatenated ground truth labels from all sample files.
        - jets (pd.DataFrame): Concatenated jet data from all sample files.
    """
    
    # If path_to_samples is a string, convert it to a list of file paths using glob
    if isinstance(path_to_samples, str):
        path_to_samples = glob(path_to_samples if '*' in path_to_samples 
                               else f'{path_to_samples}/*.npz')

    # Load the ONNX model for inference
    if isinstance(path_to_onnx, str):
        ort_sess = load_onnx_in_inference(path_to_onnx, device='cpu')
    
    outputs, Y, jets = {}, [], []
    for path in path_to_samples:
        
        # Get data from the sample file
        _Xs, _ys, _jets = get_data(path, num_data=num_data)
        
        # Run inference on the data using the ONNX model
        if isinstance(path_to_onnx, str):
            _outputs = ort_sess.run(None, {'input': np.float32(_Xs)})
        
        # Collect outputs from the inference
        for i, output in enumerate(_outputs):
            if i not in outputs:
                outputs[i] = []
            outputs[i].append(output)
        
        # Collect ground truth labels and jet data
        Y.append(_ys)
        jets.append(_jets)
    
    # Concatenate outputs, ground truth labels, and jet data
    for i in outputs:
        outputs[i] = np.concatenate(outputs[i])
    
    if len(Y)==1:
        Y = Y[0]
    else:
        Y = np.concatenate(Y)

    jets = pd.concat(jets)

    return outputs, Y, jets

### for JETCLASS style datasets ###
def get_latent(model, dataloader, layers_removed_lst:list, n_batches=50, label_list=None):
    """
    Extract latent representations and labels from the model using the provided dataloader.

    Args:
        model: The model to use for extracting latent representations.
        dataloader: The dataloader providing the data.
        n_batches: Number of batches to process.
        label_list: List of labels to filter the data.
        layers_removed_lst: List of layers to remove from the classifier network.

    Returns:
        Tuple of latent representations and labels.
    """
    latent_representations = [] if layers_removed_lst is None else {i:[] for i in layers_removed_lst}
    labels_list = []

    model = model.eval()
    
    # turn it into a sequential model so it has forward
    # make a dict of the different outputs, use in for loop
    outputMLP = {i: T.nn.Sequential(*model.classifier.network[:i]) for i in layers_removed_lst}
    
    for batch_idx, batch in tqdm(enumerate(dataloader), 
                                 total=len(dataloader) if n_batches is None else n_batches, disable=True):
        if 'scalars' in batch:
            batch['ctxt'] = batch.pop('scalars')

        labels = batch.pop('labels')
        if label_list is not None:
            mask = np.isin(labels.cpu().numpy(), label_list)
            batch = {key: value.to('cuda')[mask] for key, value in batch.items()}
            labels = labels[mask]
        else:
            batch = {key: value.to('cuda') for key, value in batch.items()}
        
        with T.no_grad():
            latent = model.pc_classifier(**batch)
            for i in layers_removed_lst:
                latent_representations[i].append(outputMLP[i](latent).cpu().numpy())

            labels_list.append(labels.cpu().numpy())

        if n_batches is not None and batch_idx >= n_batches:
            break

    latent_representations = {
        i: np.concatenate(latent_representations, axis=0)
        for i, latent_representations in latent_representations.items()
        }
    
    # name the latent representations by their size might be an issue!
    latent_representations = {f'{j.shape[-1]}': j for i, j in latent_representations.items()}
    # latent_representations = {f'{j.shape[-1]}_{i}': j for i, j in latent_representations.items()}

    labels_list = np.concatenate(labels_list, axis=0)

    return latent_representations, labels_list

def load_model_and_config(model_path: str, layers_removed: int = None):
    """
    Load the model and its configuration.

    Args:
        model_path: Path to the model.
        layers_removed: Number of layers to remove from the classifier network.

    Returns:
        Tuple of the model and its data configuration.
    """
    # Load model
    model = load_lightning_module(model_path).eval()
    if layers_removed is not None:
        model.classifier.network = model.classifier.network[:layers_removed]

    # Load configuration
    config_path = Path(model_path) / ".hydra" / "config.yaml"
    data_config = misc.load_yaml(config_path).data
    data_config.val_set.n_jets = None

    return model, data_config

def get_latent_of_datasets(model_path: str, data_paths: dict, n_batches=50, label_list=None, layers_removed: int = None, layers_removed_lst:list=None, chunk_nr:int=None):
    """
    Get latent representations and labels for multiple datasets.

    Args:
        model_path: Path to the model.
        data_paths: Dictionary of dataset names and their paths.
        n_batches: Number of batches to process.
        label_list: List of labels to filter the data.
        layers_removed: Number of layers to remove from the classifier network.
        layers_removed_lst: List of layers to remove from the classifier network.

    Returns:
        Dictionary containing latent representations and labels for each dataset in get_latent.
    """
    # Load model and configuration
    model, data_config = load_model_and_config(model_path, layers_removed)

    data = {}

    for dataset_name, dataset_path in data_paths.items():
        dataset_config = deepcopy(data_config)
        data[dataset_name] = {}

        if dataset_path is not None:
            dataset_config.val_set.path = dataset_path

        dataset_instance = hydra.utils.instantiate(dataset_config)
        dataloader = dataset_instance.val_dataloader()
        dataloader.dataset.chunk_list = [chunk_nr]

        latent_representations, labels = get_latent(model, dataloader, n_batches=n_batches, label_list=label_list,
                                                    layers_removed_lst=layers_removed_lst)

        data[dataset_name]['labels'] = labels
        data[dataset_name]['latn'] = latent_representations
        data[dataset_name]['jets'] = dataloader.dataset.file['jets'][
            chunk_nr:chunk_nr+len(labels)
            ]

    return data

def inverse_norm(data:np.ndarray, norm_path:str):
    scaler = joblib.load(norm_path)
    
    data = scaler.inverse_transform(data)

    return data

def flow_norm(data:np.ndarray, model_path:str, to_base:bool=True, dataloader_kwargs=None):   
    
    if dataloader_kwargs is None:
        dataloader_kwargs = {
            'batch_size': 1024*10,
            'shuffle': False,
            'num_workers': 4,
            }

    model = load_flow_from_path(model_path)
    
    ten_val_data = T.utils.data.DataLoader(data, **dataloader_kwargs)
    gauss_val_data = []
    
    for batch in tqdm(ten_val_data, total=len(ten_val_data)):
        if to_base:
            batch = model.inverse(batch.to(model.device))
        else:
            batch = model.undo_inverse(batch.to(model.device))
        
        gauss_val_data.append(batch.detach().cpu().numpy())

    gauss_val_data = np.vstack(gauss_val_data)
    
    return gauss_val_data

def prepare_for_OTDataModule(data:partial, norm_path:str=None, flow_path:str=None, **kwargs):
    
    if norm_path is not None and flow_path is None:
        scaler = joblib.load(norm_path)
    
    if isinstance(data, (list, ListConfig)):
        train = data[0]()
        valid = data[1]()
        
        latn_key = list(train.keys())
        if len(latn_key)>1:
            raise ValueError('Only one key should be present in the data')

        train = train[latn_key[0]]
        valid = valid[latn_key[0]]
            
        if norm_path is not None and flow_path is None:
            train = scaler.transform(train)
            valid = scaler.transform(valid)
        elif flow_path is not None:
            train = flow_norm(train, flow_path, to_base=True)
            valid = flow_norm(valid, flow_path, to_base=True)

        return train, valid
    else:
        outputs = data()
    
        if 'data' in outputs:
            output = outputs['data']['latn']
        else:
            output = outputs['latn']

        return output


def ndim_gaussian(ndim:int, n_samples:int, save_path:str, **kwargs):
    
    mean_path = os.path.join(save_path, 'mean.npy')
    cov_path = os.path.join(save_path, 'cov.npy')
    
    # if they already exist load them
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
        cov = np.load(cov_path)
    else:
        # otherwise create them
        mean = np.random.rand(ndim)

        # Create a random matrix
        A = np.random.rand(ndim, ndim)
        # Make it symmetric positive definite
        cov = (np.dot(A, A.T) + ndim * np.eye(ndim)) / ndim

        # and save
        np.save(mean_path, mean)
        np.save(cov_path, cov)

    return np.random.multivariate_normal(mean, cov, size=n_samples)

def standard_multivariate_gaussian(ndim: int, n_samples: int, **kwargs):
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    return np.random.multivariate_normal(mean, cov, size=n_samples)


