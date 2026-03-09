from copy import deepcopy
import logging
from glob import glob
from collections.abc import Callable, Generator
from functools import partial
from itertools import starmap

import h5py
import numpy as np
import torch as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from src.datamodules.hdf import identity
from src.datamodules.hdf_utils import HDFRead, combine_slices

log = logging.getLogger(__name__)

class BatchSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        start_idx: int = 0,
        seed: int = 0,
    ):
        """Batch sampler for an h5 dataset.

        Parameters
        ----------
        dataset : torch.data.Dataset
            Input dataset, used only to determine the length
        batch_size : int
            Number of objects to batch
        shuffle : bool
            Shuffle the batches
        drop_last : bool
            Drop the last incomplete batch (if present)
        start_idx : int
            The starting index for the batch sampler
        seed : int
            The seed for the random number generator
        """
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length // self.batch_size  # full batches
        self.incl_last = not drop_last and (self.dataset_length % self.batch_size != 0)
        self.shuffle = shuffle
        self.start_idx = start_idx % (self.n_batches + self.incl_last)
        self.seed = seed

    def __len__(self) -> int:
        return self.n_batches + self.incl_last - self.start_idx

    def __iter__(self) -> Generator:
        # Create the batch ids
        if self.shuffle:
            gen = T.Generator().manual_seed(self.seed)
            self.batch_ids = T.randperm(self.n_batches, generator=gen)
        else:
            self.batch_ids = T.arange(self.n_batches)

        # Trim based on the starting index
        self.batch_ids = self.batch_ids[self.start_idx :]

        # yield full batches from the dataset
        for batch_id in self.batch_ids:
            start = batch_id * self.batch_size
            stop = (batch_id + 1) * self.batch_size
            yield np.s_[int(start) : int(stop)]

        # yield the partial batch at the end
        if self.incl_last:
            start = self.n_batches * self.batch_size
            stop = self.dataset_length
            yield np.s_[int(start) : int(stop)]

class JetHDFStream(Dataset):
    """A class for streaming in jets from HDF without loading buffers into memory.

    Should be combined with the RandomBatchSampler for training.
    """

    def __init__(
        self,
        *,
        path: str,
        n_classes: int,
        features: list[list] | None = None,
        csts_dim: int | None = None,
        n_jets: int | list = 0,
        transforms: Callable | list = identity,
        chunk_list:list=None,
        allowed_labels:list=None
    ) -> None:
        """Parameters
        ----------
        path : str
            The path containing all the HDF files.
        features : list of tuples
            The features to be loaded from the dataset.
            Should have three elements: the (key, dtype, slice).
        n_classes : int
            The number of classes in the dataset. Purely for convenience.
            Is not actually used in the class.
        n_jets: int or None, optional
            The total number of jets in the dataset.
        transforms : partial
            A callable function to apply during the getitem method
        """
        # Default features for jetclass
        if features is None:
            features = [
                # ["csts", "f", [128]],
                # ["mask", "bool", [128]],
                # ["csts_id", "l", [128]],
                ["csts", "f", [None]],
                ["mask", "bool", [None]],
                ["csts_id", "l", [None]],
                ["labels", "l", None],
                ["jets", "f", None],
            ]

        # Insert the csts dim into the features
        # This is a hack but we need the csts to change with hydra for now
        if csts_dim is not None:
            log.info("Warning! Explicitly setting the csts dimension")
            log.info("This is a hack and should be removed in the future!")
            c_idx = [i for i, f in enumerate(features) if f[0] == "csts"][0]
            curr = features[c_idx][-1]
            features[c_idx][-1] = [curr, [csts_dim]]
            log.info("New feature slice for csts:")
            log.info(features[c_idx])
                
        # idx for multiple files
        self.file_idx=0

        if '*' in path:
            # multiple files
            self.paths = glob(path)
            self.file = h5py.File(self.paths[0], mode="r")
        else:
            self.paths = [path]

            # single file
            self.file = h5py.File(path, mode="r")

        # define the chunk of data to take out of file
        self.chunk_list = chunk_list

        # Class attributes
        self.n_classes = n_classes
        self.allowed_labels=allowed_labels
        if len(allowed_labels) != n_classes:
            raise ValueError("Allowed labels have to be the same as n_classes")
        
        # can be any issue if
            # labels are not the same between files or
            # distribution of labels are not "uniform" 
        self.hot_encode_fix = False
        if np.max(self.file['labels'][:100_000]) != n_classes: 
            self.hot_encode_fix = True
            if self.allowed_labels is None:
                raise ValueError("Allowed labels have to be set if the labels are not the same as n_classes")

        self.features = list(starmap(HDFRead, features))

        # Set the number of jets (if not set, use all jets in the file)
        self.n_jets_in_file = len(next(iter(self.file.values())))

        if n_jets is not None:
            self.n_jets = min(n_jets, self.n_jets_in_file)
        else:
            self.n_jets = self.n_jets_in_file

        # Save the preprocessing as a list
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms

        log.info(f"Streaming from {path}")
        log.info(f"- selected {self.n_jets} jets")

    def __len__(self) -> int:
        return self.n_jets * len(self.paths)

    def __getitem__(self, idx: int | slice) -> tuple:
        """Retrieves an item and applies the pre-processing function."""
        
        if idx.start//self.n_jets > self.file_idx and len(self.paths)>1:
            # increase file index when file is done
            self.file_idx+=1
            
            # load new file
            self.file = h5py.File(self.paths[self.file_idx], mode="r")

        elif idx.start < self.n_jets and self.file_idx > 0:
            # scale back to zero after evaluation
            self.file_idx = 0
        
        # get constant for multiple files to be able to scale back to zero
        if self.chunk_list is None:
            constant = self.n_jets*self.file_idx
        else:
            # has to negative because of minus below in idx
            # self.chunk_list could be a list of two element but not implemented
            constant = -self.chunk_list[0]
        
        # scale by constant if multiple files
        idx = slice(idx.start - constant, idx.stop - constant, None)

        data = {
            d.key: self.file[d.key][combine_slices(idx, d.s_)].astype(d.dtype)
            for d in self.features
        }
        
        if "csts" in data:
            data["x"] = data.pop("csts")
        
        for fn in self.transforms:
            data = fn(data)
        
        if self.hot_encode_fix:
            data["labels"] = T.nn.functional.one_hot(
                T.tensor(data["labels"]).long(), num_classes=np.max(self.allowed_labels) + 1
                )[:, self.allowed_labels].argmax(1).numpy()

        return data

class LatnHDFStream(JetHDFStream):
    def __init__(self, key, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.key=key
    
    def __getitem__(self, idx):

        return self.file[self.key][idx]

class DomainHDFStream(JetHDFStream):
    """A class for streaming in jets from HDF without loading buffers into memory.

    Should be combined with the RandomBatchSampler for training.
    """

    def __init__(
        self,
        *,
        paths: list[list],
        n_classes: int,
        n_jets: int | list,
        idx_shift: int,
        features: list[list] | None = None,
        csts_dim: int | None = None,
        transforms: Callable | list = identity,
    ) -> None:
        """Parameters
        ----------
        path : str
            The path containing all the HDF files.
        features : list of tuples
            The features to be loaded from the dataset.
            Should have three elements: the (key, dtype, slice).
        n_jets: int 
            The total number of jets in the dataset.
        idx_shift: int
            The shift in the index for multiple files
        n_classes : int
            The number of classes in the dataset. Purely for convenience.
            Is not actually used in the class.
        transforms : partial
            A callable function to apply during the getitem method
        """
        super().__init__(path=paths[0], n_classes=n_classes, features=features, csts_dim=csts_dim, n_jets=n_jets, transforms=transforms)

        # idx for multiple files
        self.file_idx=0
        self.paths = paths

        # Set the number of jets (if not set, use all jets in the file)
        self.n_jets = n_jets
        self.idx_shift = idx_shift

        self.file = [h5py.File(i, mode="r") for i in self.paths]

        # Class attributes
        self.n_classes = n_classes
        # self.features = list(starmap(HDFRead, features))

        # Save the preprocessing as a list
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms

        log.info(f"Streaming from {paths}")
        log.info(f"- selected {self.n_jets} jets")

    def __len__(self) -> int:
        return self.n_jets

    def __getitem__(self, idx: int | slice) -> tuple:
        """Retrieves an item and applies the pre-processing function."""
        
        # scale by constant if multiple files
        idx = slice(self.idx_shift+idx.start//2, self.idx_shift+idx.stop//2, None)

        data = {}
        
        # loop over files and concat to a batch
        for nr, f in enumerate(self.file):
            for d in self.features:
                key = d.key
                value = f[key][combine_slices(idx, d.s_)].astype(d.dtype)
                
                # change labels to domain label
                if 'labels' in key:
                    value = np.ones((len(value),1))*nr
                
                # concat to batch
                data[key] = (np.concatenate((data[key], value), axis=0) if key in data 
                             else value)
        
        if "csts" in data:
            data["x"] = data.pop("csts")
        
        for fn in self.transforms:
            data = fn(data)

        return data


class StreamModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_set: partial,
        val_set: partial,
        test_set: partial,
        num_workers: int = 6,
        batch_size: int = 1000,
        pin_memory: bool = True,
        transforms: list | Callable = identity,
        name:str='jetclass'
    ) -> None:
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set()  # initialise now to calculate data shape
        self.test_set = test_set
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.transforms = deepcopy(transforms)
        self.val_transforms = deepcopy(transforms)
    
        if isinstance(self.transforms, list):
            # transform as list not supported!!!! 
            # [] is hardcoded many places!
            raise TypeError("Transforms should be a callable function - deepcopy has problems with lists of partials")

        # remove the mask amount from the validation set
        # if hasattr(self.val_transforms, 'keywords') and 'mask_amount' in self.val_transforms.keywords:
        #         self.val_transforms.keywords['mask_amount']=None

        self.name=name
        self.n_classes = self.val_set.n_classes
        self.val_set.transforms = [self.val_transforms]
        self.last_batch_idx = 0
        self.last_epoch = 0

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        if stage in {"fit", "train"}:
            self.train_set = self.train_set()
            self.train_set.transforms = [self.transforms]
        if stage in {"predict", "test"}:
            self.test_set = self.test_set()
            self.test_set.transforms = [self.val_transforms]

    def get_dataloader(self, dataset: Dataset, flag: str) -> DataLoader:
        is_train = flag == "train"
        return DataLoader(
            dataset=dataset,
            sampler=BatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=is_train,  # flag == "train", Honestly its so big...
                drop_last=is_train,
                start_idx=self.last_batch_idx * is_train,
                seed=self.last_epoch * is_train,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=None,  # batch size is handled by the sampler
            shuffle=False,  # shuffle is handled by the sampler
            collate_fn=None,  # collations should be handled by the dataset
        )

    def get_data_sample(self) -> tuple:
        """Get a data sample to help initialise the network."""
        return next(iter(self.val_set))

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_set, "train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.val_set, "val")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_set, "test")

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_batch_idx = state_dict["last_batch_idx"]
        self.last_epoch = state_dict["last_epoch"]

    def state_dict(self) -> dict:
        return {"last_batch_idx": self.last_batch_idx, "last_epoch": self.last_epoch}

    def on_before_batch_transfer(self, batch: dict, dataloader_idx: int) -> None:
        """Update the last batch index during validation."""
        if self.trainer.validating:
            self.last_batch_idx = self.trainer.global_step
            self.last_epoch = self.trainer.current_epoch
        return batch