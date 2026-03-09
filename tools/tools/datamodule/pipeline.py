import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import sys
from tqdm import tqdm
from functools import partial
from typing import Dict
import h5py


import pandas as pd
import numpy as np

import torch as T
import pytorch_lightning as L
from torch.utils.data import DataLoader
import hydra

from .. import misc
from . import prepare_data 
from ..visualization import general_plotting as plot


def pc_2_image(data: np.ndarray, mask:np.ndarray, style_kwargs: Dict = None):
    """
    Convert point cloud data to images by generating a series of 2D histograms.

    This function takes a 3D point cloud dataset and converts it into a series of 2D histograms
    representing different slices or frames of the data.

    Parameters:
    data : ndarray
        The point cloud data with shape (n_samples, n_points, 3), where the last dimension
        represents the (x, y, z) coordinates of the points.
    style_kwargs : dict, optional
        Additional keyword arguments to pass to np.histogram2d.

    Returns:
    images : ndarray
        An array of 2D histograms representing different frames or slices of the point cloud data.
        The shape of the resulting array is (n_samples * bins, bins).

    Example:
    >>> data = np.random.rand(100, 1000, 3)  # Random point cloud data
    >>> bin_range = [0, 1, 0, 1]  # Range for x and y dimensions
    >>> bins = 10  # Number of bins for both x and y
    >>> images = pc_2_image(data, bin_range, bins)
    >>> print(images.shape)  # (1000, 10)
    """
    if style_kwargs is None:
        style_kwargs={}
    images=[]
    if len(data.shape)==3:
        for i in tqdm(range(len(data)), disable=True):
            
            image = np.histogram2d(data[i, mask[i],0],data[i, mask[i],1], weights=data[i, mask[i],2],
                                   **style_kwargs)[0]

            images.append(image[None])
        images = np.concatenate(images, 0)
    else:
        images = np.histogram2d(data[:,0],data[:,1], weights=data[:,2],
                                **style_kwargs)[0]
    return images

def get_shape_of_dict_data(data, image_bool:bool=False):
    shape={}
    # [1:] to remove batch dim
    for i,j in data.items():
        # dict nested keys
        if isinstance(j, dict):
            for k,l in j.items():
                shape[f"{i}_{k}"]=list(l.shape)
        else:
            shape[i]=list(j.shape)
    if image_bool:
        for i,j in shape.items():
            if "image" in i:
                shape[i]=j[::-1]
    return shape

class Loader(L.LightningDataModule):
    def __init__(self, sample:T.Tensor, mask:T.Tensor=None,
                 ctxt:T.Tensor=None, **kwargs):
        super().__init__()
        if (len(sample)==2) and (isinstance(sample[0], np.ndarray)):
            sample = sample[0] 
        if not isinstance(sample, T.Tensor):
            sample = T.Tensor(sample)

        self.sample=sample
        self.mask=mask
        self.ctxt = ctxt
        if not hasattr(self, "loader_config"):
            self.loader_config = kwargs.get("loader_config", {})


    def train_dataloader(self):
        return DataLoader(self, **self.loader_config)

    def val_dataloader(self): # TODO should not have train/test in the same
        test_loader_config = self.loader_config.copy()
        test_loader_config["persistent_workers"]=False
        test_loader_config["shuffle"]=False
        test_loader_config["pin_memory"]=False
        test_loader_config.pop("pin_memory_device", None)
        # test_loader_config["batch_size"]=512
        return DataLoader(self, **test_loader_config)

    def __len__(self):
        return len(self.sample)
    
    def _shape(self):
        return list(self.sample.shape[1:])

    def __getitem__(self, idx):
        # add data to dict
        data = {"inpt": self.sample[idx]}

        # add mask if not None
        if self.mask is not None:
            data["mask"] = self.mask[idx]

        # add ctxt if anything there
        if self.ctxt is not None:
            data["ctxt"] = {}
            for i in self.ctxt:
                if "mask" in i:
                    data["ctxt"][i] = self.ctxt[i][idx]
                else:
                    data["ctxt"][i] = self.ctxt[i][idx]
        return data

    def __call__(self, gen_data, **kwargs):
        raise NotImplementedError
    
class ScalarLoader(Loader):
    hist_kwargs = {
        "percentile_lst": [0, 99.5],
        "style": {"bins": 40, "histtype": "step"},
        "dist_styles": [
            {"marker": "o", "color": "black", "label": "Target", "linewidth": 0},
            {"linestyle": "dotted", "color": "blue", "label": "Diffusion", "drawstyle": "steps-mid"},
        ]
    }
    scatter_styles = [
            {"alpha": 0.1, "color": "black", "label": "Target"},
            {"alpha": 0.1, "color": "blue", "label": "Diffusion"},
        ]
    def __call__(self, gen_data, **kwargs):
        log=kwargs.get("log", {})
        if isinstance(gen_data, T.Tensor):
            gen_data = gen_data.cpu().detach().numpy()
        
        # plot marginals
        log = self.plot_marginals(self.sample.numpy(),gen_data,
            col_name=[f"var_{i}" for i in range(self.sample.shape[-1])],
            hist_kwargs=self.hist_kwargs, log=log)

        if gen_data.shape[-1]==2:
            log = plot.scatter(self.sample,gen_data,
                         scatter_styles=self.scatter_styles,
                         name="2d_scatter", log=log,
                         figsize=(1.5*8, 1.5*6))
        return log


class PointCloudLoader(Loader):
    def __init__(self, paths, max_cnstits=None, standardize_bool:bool=True,
                 ctxt:list =None):
        self.paths = paths
        
        # used for non-padded data
        self.max_cnstits=max_cnstits
        self.standardize_bool=standardize_bool
        self.idx_number=None
        self.ctxt_labels=ctxt
        self.ctxt={}
        self.ctxt_shape={}

        # load data
        self.load_data()
        
        # get norm values
        self.calculate_norms()
        
        # do this after norm if the cnts are normed
        self.init_ctxt()
        
        super().__init__(self.sample, self.mask, self.ctxt)


    def load_data(self):

        # load data
        if ".csv" in self.paths[0]:
            self.df = [pd.read_csv(i, dtype=np.float32) for i in self.paths]
            self.df = pd.concat(self.df, axis=0)
            self.df=self.df.rename(columns={i:i.replace(" ", "") for i in self.df.columns})
        elif (".h5" in self.paths[0]) or ("hdf5" in self.path[0]):
            self.df = h5py.File(self.paths[0])

        # split and reshape
        # sample should be: (n_pc x features x constituents)
        if "mnist" in self.paths[0]:
            (self.sample, self.mask, self.min_cnstits, max_cnstits, self.n_pc, self.label
             ) = prepare_data.prepare_mnist(self.df)
        elif "shapenet" in self.paths[0]:
            (self.sample, self.mask, self.min_cnstits, max_cnstits, self.n_pc #, self.label
             ) = prepare_data.prepare_shapenet(self.df)
        else:
            raise ValueError("Unknown data path")

        # reduce max cnts
        if self.max_cnstits is None:
            self.max_cnstits=max_cnstits
        
        # make sure not cnts above max_cnts
        mask_max_cnts = self.mask.sum(1)<=self.max_cnstits

        # apply max mask and max cnts
        self.sample =  self.sample[mask_max_cnts, :self.max_cnstits]
        self.mask = self.mask[mask_max_cnts, :self.max_cnstits]
        self.n_pc = mask_max_cnts.sum()

        self.pc_shape=self.sample.shape[1:]
        
        print(f"Sample size {self.sample.shape}")
    
    def init_ctxt(self):
        
        # add scalar ctxt variables (user defined)
        # function to unpack the variables
        if self.ctxt_labels is not None:

            if "scalars" in self.ctxt_labels:
                # for scalar variables into ctxt
                self.ctxt["scalars"]={}
                for i in self.ctxt_labels["scalars"]:
                    if "label" in i:
                        self.ctxt["scalars"][i] = self.label
                    if "size" in i:
                        self.ctxt["scalars"][i] = self.mask.sum(1) 
                self.ctxt["scalars"]=np.concatenate([j[:,None] for _, j in self.ctxt["scalars"].items()],1)
                self.ctxt_shape["scalars"] = [len(self.ctxt_labels["scalars"])]

            if "cnts" in self.ctxt_labels:
                # cnts variables into ctxt
                self.ctxt["cnts"]=self.sample
                self.ctxt["mask"]=self.mask
                self.ctxt_shape["cnts"] = self.sample.shape[1:]
        

    def calculate_norms(self):
        self.mean, self.std = (self.sample[self.mask, :].mean(0),
                               self.sample[self.mask, :].std(0))

        # if self.ctxt is not None:
        #     if len(self.ctxt.shape)==2:
        #         self.ctxt_mean, self.ctxt_std = self.ctxt.mean(0), self.ctxt.std(0)
        #     else:
        #         raise NotImplementedError("ctxt for this shape not implemented yet")

        #norm sample
        if self.standardize_bool:
            self.sample = (self.sample-self.mean)/self.std

    def _shape(self):
        shape = {"inpt": list(self.sample.shape[1:])}
        shape.update({f"ctxt_{i}":j for i,j in self.get_ctxt_shape().items()})
        return shape

    def get_normed_ctxt(self):
        context = self.ctxt.copy()
        context["true_n_cnts"] = self.mask.sum(1)
        return context

    def get_ctxt_shape(self):
        return self.ctxt_shape

class ImageModule(Loader):
    def __init__(self, data: partial, img_enc=None, loader_config=None,
                 permute_dims=(1,2,0), use_label:bool=False, **kwargs):
        self.data = data
        self.loader_config = loader_config
        self.img_enc = img_enc
        self.permute_dims=permute_dims
        self.use_label=use_label
        
        # has to be the last thing
        self.test_data = self.__getitem__(0)
        
        # shape of label is not important
        self.test_data.pop("label", None)
        
        # get shape of inpt
        self.image_shape = get_shape_of_dict_data(self.test_data)
        self._n_epoch=0

        super().__init__(self.data)
    
    def get_ctxt_shape(self):
        shape = {}
        for i,j in self.image_shape.items():
            if "ctxt" in i:
                shape[i]=j
        return shape
    
    def _shape(self):
        return self.image_shape

    def __getitem__(self, idx):
        
        # run over dataloader
        image, label = self.data[idx]
        
        # data into dict
        data = {"inpt" : image.permute(*self.permute_dims)}

        if self.img_enc is not None:
            # run transformation
            data["ctxt"] = {"inpt": self.img_enc(image).permute(*self.permute_dims)}

        if self.use_label:
            data["label"] = label

        return data

class ImageModuleCSV(Loader):
    def __init__(self, path: str, img_enc=None, loader_config=None,
                 permute_dims=(1,2,0), use_label:bool=False,
                 flatten_bool:bool=False, **kwargs):
        self.path = path
        self.data = pd.read_csv(self.path).iloc[:kwargs.get('size',None)]
        self.loader_config = loader_config
        self.img_enc = img_enc
        self.permute_dims=permute_dims
        self.use_label=use_label
        self.flatten_bool=flatten_bool
        self.use_latent = kwargs.get("use_latent", False)

        # process data
        self.preprocess_data()
        
        # has to be the last thing
        self.test_data = self.__getitem__(0)
        
        # shape of label is not important
        self.test_data.pop("label", None)
        
        # get shape of inpt
        self.image_shape = get_shape_of_dict_data(self.test_data)
        self._n_epoch=0
        self.class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        super().__init__(self.data)

    def preprocess_data(self):
        self.y = self.data['label'].values
        self.data = self.data[list(self.data)[1:]].values

        ## normalize and reshape the predictors
        self.data = self.data / 255

        # img dimension
        if not self.flatten_bool:
            self.data = self.data.reshape(-1, 28,28)
            
            # Unsqueeze on axis 1
            self.data = np.expand_dims(self.data, axis=1)
            
        # convert to tensor and float
        self.data = T.tensor(self.data).float()

        # self.y = T.nn.functional.one_hot(T.tensor(self.y)).float()
        self.y = T.tensor(self.y).float()
        self.idx = T.arange(len(self.y)).float()
        
        self.latent_space = T.ones(len(self.data), 256)
        self.init_latent_space = self.latent_space.clone()

    def __getitem__(self, idx):
        
        # run over dataloader
        image = self.data[idx]
        label = self.y[idx]
        
        if not self.flatten_bool:
            image = image.permute(1,2,0)
        
        # data into dict
        data = {"inpt" : image}

        if self.use_latent:
            data['idx'] = self.idx[idx]
            # data['ctxt'] = self.latent_space[int(data['idx'])]

        if self.img_enc is not None:
            # run transformation
            data["ctxt"] = {"inpt": self.img_enc(image).permute(*self.permute_dims)}

        if self.use_label:
            data["labels"] = label

        return data

if __name__ == "__main__":
    # %matplotlib widget
    if False: # testing inpt
        import matplotlib.pyplot as plt
        config = misc.load_yaml("configs/data_cfg.yaml")
        data = hydra.utils.instantiate(config.train_set)
        dataloader = hydra.utils.instantiate(config.loader_cfg)(data)
        image_loader = hydra.utils.instantiate(config.img_enc)(dataloader)
        # downscale_func = T.nn.AvgPool2d(2)
        # image_loader = ImageEnhancement(dataloader, downscale_func)
        
        for i, cvx in image_loader:
            # print(i)
            break
        cvx= cvx.permute(0, 2, 3, 1)
        i= i.permute(0, 2, 3, 1)


        style = {"vmax":1, "vmin":0}
        figsize=(4*8, 6)
        _, ax = plt.subplots(1,4, figsize=figsize)
        for a, img in zip(ax, cvx):
            a.imshow(img, **style)

        _, ax = plt.subplots(1,4, figsize=figsize)
        for a, img in zip(ax, i):
            a.imshow(img, **style)
    elif False: # testing shapenet pc
        import h5py
        h5_file = h5py.File("/home/users/a/algren/scratch/diffusion/shapenet/shapenet.hdf5")
        data= {i:[] for i in ["train", "test", "val"]}
        for i in tqdm(h5_file.keys()):
            for j in data.keys():
                data[j].append(h5_file[i][j][:])
        for j in data.keys():
            _data = np.concatenate(data[j], 0)
            index = np.arange(0, len(_data), 1)
            np.random.shuffle(index)
            data[j] = _data[index, :, :]
        import matplotlib.pyplot as plt
        for i in range(9):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            n = 100

            ax.scatter(data["train"][i, :, 0], data["train"][i, :, 1],
                    data["train"][i, :, 2], marker="o")
    elif False: # create correct pc data size
        PATH = "/home/users/a/algren/scratch/diffusion/pileup/ttbar.csv"
        df = pd.read_csv(PATH, dtype=np.float32)
        (sample, mask, idx_number, max_cnstits, X_max, mean, std, n_pc
         ) = preprocess_jets(df)
        new_sample, new_mask= fill_data_in_pc(sample, n_pc, idx_number, max_cnstits)
        pd.DataFrame(new_sample.reshape(new_sample.shape[0], -1)).to_csv(
            PATH.replace(".csv", "_processed.csv")
        )
    else:
        PATH = "/home/users/a/algren/scratch/diffusion/pileup/ttbar_processed.csv"
        pc_jet = PointCloud(PATH)