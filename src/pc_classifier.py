

"Different diffusion models for training and evaluation"
# import numpy as np
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from typing import Callable,Tuple,Union
import pytorch_lightning as L
from glob import glob
import hydra

import torch as T
import torch.nn as nn
from tools.tools import schedulers, misc
from tools.tools.torch_utils import count_trainable_parameters
from tools.tools.modules import IterativeNormLayer
from tools.tools.torch_utils import activation_functions

def load_lightning_module(path: str, dev: str = "cuda", model_kwargs: dict = None):
    """
    Load a PyTorch Lightning module from a checkpoint.

    Args:
        path (str): The directory path where the model and configuration files are stored.
        dev (str): The device to load the model on. Default is "cuda".
        model_kwargs (dict, optional): Additional keyword arguments to pass to the model's `load_from_checkpoint` method. Default is None.

    Returns:
        model: The loaded PyTorch Lightning model.
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Load model configuration
    model_cfg = misc.load_yaml(f"{path}/.hydra/config.yaml")
    
    # Get the model class from the configuration
    model = hydra.utils.get_class(model_cfg.model._target_)

    # Set the device for the model
    model.device = dev

    # Find the checkpoint file
    model_paths = glob(f"{path}/checkpoints/best_*.ckpt")

    # Load the model from the checkpoint
    model = model.load_from_checkpoint(model_paths[-1],
                                       map_location=dev, device=dev, **model_kwargs)
    return model.eval()

def load_nn_module(path: str, config_name: str, weight_name:str):
    """
    Load a specific neural network module from a checkpoint.

    Args:
        path (str): The directory path where the model and configuration files are stored.
        config_name (str): The name of the network configuration to load from the configuration file.
        weight_name (str): The name of the network weights to load from the checkpoint.

    Returns:
        network: The loaded neural network module.
    """
    # Load model configuration
    model_cfg = misc.load_yaml(f"{path}/.hydra/config.yaml")
    
    # Instantiate the network from the configuration
    network = hydra.utils.instantiate(model_cfg.model[config_name])()
    
    # Find the checkpoint file
    model_paths = glob(f"{path}/checkpoints/best_*.ckpt")
    
    # Load the checkpoint
    checkpoint = T.load(model_paths[-1])
    
    # Load the state dictionary into the network
    network.load_state_dict({k.replace(f'{weight_name}.', ''): v for k, v in checkpoint["state_dict"].items() if k.startswith(f"{weight_name}.")})

    return network.eval()

class ClassifierHead(T.nn.Module):
    def __init__(self, input_dims:int, output_dims:int, hidden_dims:Union[list, None] =None,
                 act_func:str="relu", device:str="cuda", norm_inpt:bool=False) -> None:

        super().__init__()

        self.act_func=act_func
        self.input_dims=input_dims
        self.output_dims=output_dims
        self.device = device
        self.hidden_dims=hidden_dims
        self.norm_inpt=norm_inpt
        
        self.network = T.nn.ModuleList()

        if norm_inpt:
            self.input_norm = IterativeNormLayer((1,self.input_dims))

        # setup linear classifier head
        # init classifier head for downstream task
        if hidden_dims is not None:
            for nr, layer_size in enumerate(hidden_dims):
                if len(hidden_dims)==nr-1:
                    break
                elif nr==0:
                    self.network.append(T.nn.Linear(self.input_dims, layer_size))
                else:
                    self.network.append(T.nn.Linear(hidden_dims[nr - 1], layer_size))
                self.network.append(activation_functions(self.act_func))

            self.network.append(T.nn.Linear(self.hidden_dims[-1], self.output_dims))
        else:
            self.network.append(T.nn.Linear(self.input_dims, self.output_dims))
            self.network.append(activation_functions(self.act_func))
            

        if self.output_dims in [1, 2]:
            self.loss_func = T.nn.BCEWithLogitsLoss()
        else:
            self.loss_func = T.nn.CrossEntropyLoss()
        
        self.to(self.device)
    
    def _get_loss_and_predict(self, x:T.Tensor, target:T.Tensor) -> T.Tensor:

        # forward pass
        x = self.forward(x)
        
        #return loss
        return self.loss_func(x, target), x
    
    def forward(self, x:T.Tensor) -> T.Tensor:
        if self.norm_inpt:
            x = self.input_norm(x)

        for layer in self.network:
            x = layer(x)

        return x


class PCClassifier(L.LightningModule):
    def __init__(self, input_dims:int, output_dims:int,
                 pc_classifier:Callable,
                 classifier:Callable,
                 train_config:dict,
                 eval_fw:Callable=None, 
                #  tasks:List[Callable]=None, 
                 **kwargs) -> None:

        super().__init__()
        # self.save_hyperparameters(ignore=['backbone'])
        self.save_hyperparameters() # no reason to ignore backbone? 
        self.input_dims=input_dims
        self.output_dims=output_dims
        self.precision = kwargs.get("precision", '16-mixed')
        self.train_config=train_config
        self.eval_fw=eval_fw
        self.ctp = {} # count trainable parameters
        
        self.pc_classifier=pc_classifier(cls_dim_in = self.input_dims)
        self.classifier=classifier(output_dims=self.output_dims)

        # init loss function
        if self.classifier.output_dims>1:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
        
        # setup validation step 
        self.validation_dict = {i: [] for i in ["mask", "scalars",
                                                "labels", 'clf_out']}

        self.ctp["Classifier size"] = count_trainable_parameters(self.classifier)
        self.ctp["PC Classifier size"] = count_trainable_parameters(self.pc_classifier)

    def configure_optimizers(self):
        "configure optimizer and scheduler. If scheduler is not in train_config, it will be None."
        
        # all parameters
        parameters = self.parameters() # should contain everything
        
        # get params of embedder and diffusion model
        optimizer = {"optimizer": T.optim.AdamW(parameters, **self.train_config["opt_cfg"])}

        if "lr_scheduler" in self.train_config: # config for scheduler
            optimizer["lr_scheduler"] = schedulers.get_scheduler(optimizer=optimizer["optimizer"],
                                                                 **self.train_config.lr_scheduler)

        return optimizer

    def _shared_step(self, batch, batch_idx, log_name:str="train") -> Tuple[T.Tensor, T.Tensor, T.Tensor | None]:
        total_loss = T.tensor(0.0, requires_grad=True)
        
        labels = batch.pop('labels').long()
        
        if 'scalars' in batch:
            batch['ctxt']= batch.pop('scalars')

        # run full chain of model
        latn = self.pc_classifier(**batch)
        clf_out = self.classifier(latn)
        
        # calculate loss
        entropy = self.loss(clf_out, labels) # might need float for BCE

        total_loss = total_loss + entropy

        # calculate accuracy
        if self.classifier.output_dims>1:
            accuracy = T.mean((clf_out.argmax(1)==labels)*1.0)
        else:
            accuracy = T.mean(((clf_out>0)*1==labels)*1.0)

        # log losses
        if log_name is not None:
            self.log(f"{log_name}/entropy", entropy, prog_bar=True)
            self.log(f"{log_name}/accuracy", accuracy, prog_bar=True)
            self.log(f"{log_name}/total_loss", total_loss, prog_bar=False)
        
        return total_loss, accuracy

    def training_step(self, batch, batch_idx) -> T.Tensor:
        return self._shared_step(batch, batch_idx, log_name="train")[0]

    def on_train_batch_end(self, *args, **kwargs) -> None:
        """update ema with new parameters"""
        if self.lr_schedulers is not None:
            self.lr_schedulers().step() 
        
    def validation_step(self, batch:dict, batch_idx:int):
        "run validation batches and log results"
        return self._shared_step(batch, batch_idx, log_name='valid')[0]

    # def on_validation_epoch_end(self):
    #     """log validation results over all valid batches"""
    #     for i,j in self.validation_dict.items():
    #         if len(j)==0:
    #             continue
    #         self.validation_dict[i] = T.stack(j)
        
    #     # dont log sanity check
    #     if (self.global_step>0):
    #         # log validation entropy
    #         val_loss = self.loss(self.validation_dict['clf_out'].float(),
    #                             self.validation_dict['labels'].long())
    #         self.log(f"valid/entropy", val_loss, prog_bar=True)

    #         val_acc = T.mean((self.validation_dict['clf_out'].argmax(1) == self.validation_dict['labels'])*1.0)
    #         self.log(f"valid/accuracy", val_acc, prog_bar=True)
        
    #     # free memory
    #     for i in self.validation_dict:
    #         self.validation_dict[i] = []
