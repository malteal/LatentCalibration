import os
from torch import nn
import torch
import copy
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn import BCELoss
from sklearn.model_selection import train_test_split
from .torch_utils import activation_functions
from .visualization import general_plotting as plot
from .schedulers import get_scheduler
from . import misc

import logging

logging.basicConfig(level=logging.INFO)

def get_densenet(input_dim, output_dim, n_neurons, n_layers, activation_str,
                 batchnorm, **kwargs):
    
    # define activation
    activation = activation_functions(activation_str)
    
    # define norm
    norm_layer = nn.BatchNorm1d if batchnorm else nn.LayerNorm 

    # define dropout
    drp = kwargs.get("drp", 0)

    network = [nn.Linear(input_dim, n_neurons), nn.Dropout(drp), activation]

    # if batchnorm:
    network.append(norm_layer(n_neurons))

    hidden_layers = [
        [
        nn.Dropout(drp),
        nn.Linear(n_neurons, n_neurons),
        norm_layer(n_neurons),
        activation
        ] for _ in range(n_layers)]

    # else:
    #     hidden_layers = [[
    #         nn.Dropout(drp),
    #         nn.Linear(n_neurons, n_neurons), activation]
    #                      for _ in range(n_layers)]
    
    # extend with hidden layers
    network.extend(list(np.ravel(hidden_layers)))
    
    # output layer
    network.extend([nn.Linear(n_neurons, output_dim)])

    return network

class WeightedLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_pred, y_true, weight=None):
        return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, weight=weight.detach()) 

class DenseNet(nn.Module):
    """A very simple sense network that is initialised with an input and output dimension."""

    def __init__(self, input_dim, N=32, n_layers=4, sigmoid=False,
                 activation_str = "leaky_relu", output_dim=1,
                 device="cpu", network_type="clf", **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.device = device
        self.hidden_features = N
        self.sigmoid=sigmoid
        self.network_type=network_type
        self.train_loader = None
        self.valid_loader = None
        self.lr_scheduler = None
        self.lr_scheduler_args={}
        self.scale_path = kwargs.get("scale_path")
        self.save_path = kwargs.get("save_path")
        self.early_stoppping = kwargs.get("early_stoppping", 20)
        self.eval_str = kwargs.get("eval_str", "valid_loss")
        self.dataloader_args = {"pin_memory":True,"batch_size":512,
                                "shuffle":True, "drop_last": True}
        self.input_dim=input_dim
        kwargs.setdefault("batchnorm", True)
        kwargs.setdefault("old_layernorm_version", False)
        network = get_densenet(input_dim, output_dim, N,
                               n_layers, activation_str,
                               **kwargs)
        self.sigmoid_layer = nn.Sigmoid()
        if self.network_type == "clf":
            if kwargs.get("using_weights", False):
                if sigmoid:
                    raise ValueError("Sigmoid and using_weights cannot be used together. Loss function is using logits")
                self.loss = WeightedLoss()
            elif sigmoid:
                network.append(self.sigmoid_layer)
                self.loss = BCELoss()
            else:
                self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.network_type=="regression":
            self.loss = torch.nn.MSELoss()
        else:
            raise NameError("network_type is unknown. clf or regression!")

        self.network = nn.Sequential(*network).to(device)
        # eval
        self.best_predictions = None
        self.valid_truth_for_best=None
        
        # set optimizer
        self.set_optimizer(kwargs.get("adam", True),
                           kwargs.get("lr", 1e-3))

        self.scale_function=None
        if self.scale_path is not None:
            self.load_scale_dnn(self.scale_path)
        
        # define loss
        self.state_of_best=None
        self.nr_best_epoch=0
        self.init_new_log_dict()
        self.network.eval()

    def load_scale_dnn(self, path):
        
        nominal_cfg = misc.load_yaml(f"{path}/model_config.yml", hydra_bool=True)
        
        nominal_cfg["sigmoid"] = False
        
        nominal_cfg["device"] = self.device
        
        self.scale_function = DenseNet(**nominal_cfg)
        
        latest_model = sorted(glob(f"{path}/models/*"),key=os.path.getmtime)[-1]
        
        self.scale_function.load(latest_model, key_name="model")

        # Deactivate all trainable parameters
        for param in self.scale_function.parameters():
            param.requires_grad = False

    def set_optimizer(self, adam, lr=1e-3):
        if adam:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr)

    def init_new_log_dict(self):
        "Creating logging dict"
        self.loss_data = {f"{i}_{j}":[] for i in ["train", "valid"] for j in ["loss", "auc"]}
        self.loss_data["lr"] = []

    def load(self, path, key_name="model"):
        model_state = torch.load(path, map_location =self.device)
        self.load_state_dict(model_state[key_name])
        if "optimizer" in model_state:
            self.optimizer.state_dict = model_state["optimizer"]
        if "loss" in model_state:
            self.loss_data = model_state["loss"]
        self.network.eval()

    def save(self, output_path:str, **kwargs):
        """save state_dict

        Parameters
        ----------
        output_path : str
            path to output the saved model should end with .pt or .pth
        kwargs : 
            Should be state_dict values or loss like values
        """

        state_dict = {i: kwargs[i] for i in kwargs}
        state_dict["model"] = self.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["loss"] = self.loss_data
        torch.save(state_dict, output_path)
    
    def create_loaders(self, train, valid=None, valid_args:dict=None):
        "valid_args has to be args of train_test_split"
        if valid_args is not None and valid is None:
            train, valid = train_test_split(train, **valid_args)

        if valid is not None:
            self.valid_loader = torch.utils.data.DataLoader(valid,
                                                            **self.dataloader_args)

        self.train_loader = torch.utils.data.DataLoader(train, **self.dataloader_args)
    
    def test(self, test, truth_in_data=True):
        test_loader = torch.utils.data.DataLoader(test, batch_size = 512,
                                                  pin_memory = False,
                                                  shuffle=False)
        output_lst = {"truth":[], "output_ratio":[], "predictions":[]}
        for batch in tqdm(test_loader, total=len(test_loader)):
            x = batch[:,:self.input_dim].to(self.device)
            x.requires_grad = True
            output = self.forward(x)
            if truth_in_data:
                y = batch[:,self.input_dim:]
                output_lst["truth"].append(y.numpy())
            output_lst["output_ratio"].append(np.ravel(
                output[0].cpu().detach().numpy())
                )
            if self.network_type=="clf":
                output_lst["predictions"].append(
                    np.ravel(torch.sigmoid(output[1]).cpu().detach().numpy())
                    )
            else:
                output_lst["predictions"].append(
                    np.ravel(output[1].cpu().detach().numpy())
                    )
        output_lst = {i: np.concatenate(output_lst[i],0) for i in output_lst}
        return output_lst
    
    def set_lr_scheduler(self, lr_scheduler_name, lr_scheduler_args):
        self.lr_scheduler = get_scheduler(lr_scheduler_name, self.optimizer,
                                          lr_scheduler_args)
    
    def output_func_scale(self, x):
        if self.scale_function is not None:
            return self.scale_function(x)[-1]
        else:
            return torch.zeros(x.shape[0],1)

    def run_training(self, train_loader=None, valid_loader=None,
              n_epochs=50, load_best=True,
              **kwargs):

        if train_loader is None:
            if self.train_loader is None:
                raise ValueError("To use internal train_loader, run create_loaders()")
            if self.valid_loader is not None:
                valid_loader = self.valid_loader
            train_loader = self.train_loader

        if kwargs.get("standard_lr_scheduler", False):
            self.set_lr_scheduler("singlecosine",
                                  {"T_max": len(train_loader)*n_epochs,
                                   "eta_min": 5e-6})

        pbar = tqdm(range(n_epochs), leave=False, desc="Classifier epochs")
        
        for ep in pbar:
            self.network.train()
            self.loss_values=[]
            _clip=[]
            for batch in train_loader:
                self.optimizer.zero_grad()
                x = batch[:,:self.input_dim].float().to(self.device)
                y = batch[:,self.input_dim:].to(self.device)
                # x.requires_grad = True
                # y.requires_grad = True
                _, logit = self.forward(x)

                if y.shape[-1] == 2:
                    weight = y[:,0]
                    y = y[:,1]
                else:
                    weight = None
                if weight is None:
                    loss = self.loss(logit.view(-1), y.view(-1))
                else:
                    loss = self.loss(logit.view(-1), y.view(-1), weight=weight)
                loss.backward()

                if self.kwargs.get("clip_bool", False):
                    _clip.append(torch.nn.utils.clip_grad_norm_(self.parameters(), 5))

                self.optimizer.step()

                self.loss_values.append(loss.cpu().detach().numpy())

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            self.network.eval()

            if valid_loader is not None:
                self.evaluate_performance(valid_loader,ep_nr=ep)

            self.loss_data['train_loss'].append(float(np.mean(self.loss_values)))
            self.loss_data['lr'].append(self.optimizer.param_groups[0]['lr'])

            pbar.set_postfix({"Train loss": f"{str(round(self.loss_data['train_loss'][-1],4))}"})
            
            # early stopping
            if ep-self.nr_best_epoch>self.early_stoppping:
                logging.info(f"Early stopping at epoch: {ep}")
                break
        
        if load_best and (valid_loader is not None):
            print(f" Best epoch: {self.nr_best_epoch} ".center(30, "-"))
            if self.state_of_best is not None:
                self.load_state_dict(self.state_of_best)
        self.network.eval()

    def forward(self, x) -> tuple:
        p_hat = self.network(x)
        
        # p_bkg/p_sig
        if not self.sigmoid:
            r_hat = (1 - self.sigmoid_layer(p_hat))/self.sigmoid_layer(p_hat)
        else:
            r_hat = (1 - p_hat)/p_hat

        scale = self.output_func_scale(x)
        p_hat = p_hat+scale.to(p_hat.device)
        
        return r_hat, p_hat
    
    @torch.no_grad()
    def evaluate_performance(self,loader, ep_nr=None, log_val_bool=True):
        loss_lst = []
        output_lst = []
        valid_truth=[]
        self.data_from_loader=[]

        for batch in loader:
            x, y = (batch[:,:self.input_dim].float().to(self.device),
                   batch[:,self.input_dim:].to(self.device))
            # x.requires_grad = True
            # y.requires_grad = True
            _, logit = self.forward(x)

            if y.shape[-1] == 2:
                weight = y[:,0]
                y = y[:,1]
            else:
                weight = None

            if weight is None:
                loss = self.loss(logit.view(-1), y.view(-1))
            else:
                loss = self.loss(logit.view(-1), y.view(-1), weight=weight)
            
            loss_lst.append(loss.cpu().detach().numpy())
            output_lst.append(logit.cpu().detach().numpy())
            self.data_from_loader.append(x.cpu().detach().numpy())
            valid_truth.append(y.cpu().detach().numpy())

        self.data_from_loader = np.concatenate(self.data_from_loader,0)
        self.valid_truth = np.concatenate(valid_truth,0)
        self.output_lst = np.concatenate(output_lst,0)
        loss_lst = np.ravel(loss_lst)

        if self.network_type=="clf":
            valid_auc = calculate_auc(self.output_lst, self.valid_truth) # can use simple logit
            self.loss_data["valid_auc"].append(float(valid_auc))

        if log_val_bool:# for reg
            self.loss_data["valid_loss"].append(float(np.mean(loss_lst)))

        if log_val_bool: # for clf
            if self.eval_str == "valid_loss":
                mask_eval = np.mean(loss_lst)+1e-3 < self.loss_data["valid_loss"][:-1]
            elif self.eval_str == "valid_auc":
                mask_eval = valid_auc > self.loss_data["valid_auc"][:-1]
            else:
                raise ValueError(f"eval_str unknown: {self.eval_str}")

            if all(mask_eval):
                self.state_of_best = copy.deepcopy(self.state_dict())
                self.best_predictions = self.output_lst.copy()
                self.valid_truth_for_best = self.valid_truth.copy()
                self.nr_best_epoch = ep_nr
                if self.save_path is not None:
                    os.makedirs(f"{self.save_path}/clf_models/", exist_ok=True)
                    self.save(f"{self.save_path}/clf_models/checkpoint_{ep_nr}.pt")
                

        return float(valid_auc), float(np.mean(loss_lst))
    
    def plot_log(self, save_path=None, ep_nr=""):
        
        if save_path is None:
            save_path = self.save_path

        os.makedirs(f"{save_path}/plots_{ep_nr}/", exist_ok=True)

        # plot predictions
        fig = plt.figure()
        style={"alpha": 0.5, "bins":60,
                "range":np.percentile(self.output_lst, [1,99])}
        sig_counts, _, _ = plt.hist(self.output_lst[self.valid_truth==1],
                              label="sig", **style)
        bkg_counts, bins,_= plt.hist(self.output_lst[self.valid_truth==0], label="bkg",
                                  **style)
        # bins = (bins[1:]+bins[:-1])/2
        # outlier_size = np.sum(self.valid_truth==0)*0.01

        # try:
        #     low_cut = np.round(bins[(sig_counts-bkg_counts)<-outlier_size][-1],3)
        #     high_cut = np.round(bins[(sig_counts-bkg_counts)>outlier_size][0],3)
        #     plt.vlines(high_cut, 0, np.max(bkg_counts), colors="blue")
        #     plt.vlines(low_cut, 0, np.max(bkg_counts), colors="blue")
        # except IndexError: # if there is no bins
        #     low_cut = np.percentile(self.output_lst, 10)
        #     high_cut = np.percentile(self.output_lst, 90)
        #     plt.vlines(high_cut, 0, np.max(bkg_counts), colors="red")
        #     plt.vlines(low_cut, 0, np.max(bkg_counts), colors="red")
        low_cut = np.percentile(self.output_lst[self.valid_truth==1], 1)
        high_cut = np.percentile(self.output_lst[self.valid_truth==0], 99)
        plt.vlines(high_cut, 0, np.max(bkg_counts), colors="red")
        plt.vlines(low_cut, 0, np.max(bkg_counts), colors="red")

        plt.legend()
        plt.xlabel("Predictions")

        if save_path is not None:
            misc.save_fig(fig, f"{save_path}/plots_{ep_nr}/predictions_{ep_nr}.png")
            
        mask_low = np.ravel(low_cut>self.output_lst)
        mask_high = np.ravel(high_cut<self.output_lst)
        high_cut = str(np.round(high_cut,2))
        low_cut = str(np.round(low_cut,2))
        for i in range(self.data_from_loader.shape[1]):
            fig = plt.figure()
            style={"alpha": 0.5, "bins":50,
                    "range":np.percentile(self.data_from_loader[:,i], [0.1,99.9]),
                    'density':True}
            
            plt.hist(self.data_from_loader[mask_high][:,i],
                                label=f"Above {high_cut}", **style)
            plt.hist(self.data_from_loader[mask_low][:,i],
                                         label=f"Below {low_cut}", **style)
            # plt.hist(self.data_from_loader[:,i],
            #                              label=f"Standard dist", histtype="step",
            #                              **style)
            plt.legend()
            if save_path is not None:
                misc.save_fig(fig, f"{save_path}/plots_{ep_nr}/outliers_dist_{i}_{ep_nr}.png")
        

        for key,values in self.loss_data.items():
            if "valid" in key:
                continue
            fig = plt.figure()
            plt.plot(values, label="Train")
            plt.plot(self.loss_data[key.replace("train", "valid")], label="Valid")
            plt.ylabel(key.replace('train', ''))
            plt.xlabel('Epoch')
            plt.legend()

            # if all(np.array(values)>0) & all(np.array(self.loss_data[key.replace("train", "valid")])>0):
            #     plt.yscale("log")
                
            if save_path is not None:
                misc.save_fig(fig, f"{save_path}/plots_{ep_nr}/{key.replace('train', '')}.png")
    
    def plot_auc(self, path=None, epoch_nr="", plot_roc=False):
        fig = plt.figure()
        label = f"Minimum at {np.argmin(self.loss_data['valid_auc'])}, value at {np.min(self.loss_data['valid_auc'])}"
        plt.plot(self.loss_data["valid_auc"], label = label)
        plt.xlabel("Epoch")
        plt.ylabel(r"valid$_{AUC}$")
        if path is not None:
            if not ".png" in path:
                path += ".png"
            path = path.replace(".png", f"_{epoch_nr}.png")
            plt.savefig(path)
            plt.close(fig)
        if plot_roc:
            _, _, _, fig = plot.plot_roc_curve(self.valid_truth_for_best,
                                            self.best_predictions, label="ROC with ")
            path = path.replace(".png", "_roc_curve_best_epoch.png")
            plt.savefig(path)
            plt.close(fig)

def calculate_auc(pred, truth):
    "calculate auc in classification should be changed to torch function"
    fpr, tpr, _ = metrics.roc_curve(truth, pred)
    auc = metrics.auc(fpr, tpr)
    return auc

def train_valid_split(target, transport, valid_ratio, device="cpu"):
    """split into train and valid samples - can use numpy and torch.tensor
        This should not be split on the gpu!! So convert to cpu before running
    """
    index = np.arange(0, np.min([len(target), len(transport)]))
    np.random.shuffle(index)
    train_index = index[:-int(len(index)*valid_ratio)]
    valid_index = index[-int(len(index)*valid_ratio):]
    return (target[train_index].to(device),
            transport[train_index].to(device),
            target[valid_index].to(device),
            transport[valid_index].to(device))

def earth_mover_distance(fake, real):
    return torch.mean(fake)-torch.mean(real)

def define_loss(loss_str:str):
    if loss_str.lower() == "bce":
        loss = nn.BCEWithLogitsLoss() # combine sigmoid and BCE
    elif loss_str.lower() == "wasserstein":
        loss = earth_mover_distance
    elif loss_str.lower() == "auc":
        loss = calculate_auc
    else:
        raise ValueError("Unknown loss function name")
    return loss

# class RunDiscriminator():
#     "train discriminator to distinguish between two distributions"
#     def __init__(self,input_dim, valid_ratio=0.33, device="cuda",
#                  save_valid_curve_path:str=None, loss_str="bce", 
#                  valid_str="auc"):

#         self.input_dim = input_dim
#         self.device = device
#         self.valid_str = valid_str
#         self.loss_str = loss_str
#         self.loss_function = define_loss(loss_str=loss_str)
#         self.valid_function = define_loss(loss_str=valid_str)

#         self.dataloader_args = {"pin_memory":False,"batch_size":512,
#                             "shuffle":True, "drop_last": True
#                             }
#         self.save_valid_curve_path = save_valid_curve_path
#         self.valid_ratio = valid_ratio
#         self.transport_all = None
#         self.target_all = None
#         self._create_network()

#     @staticmethod
#     def _convert_to_tensor(distribution):
#         "convert number to torch tensor"
#         if not isinstance(distribution, torch.Tensor):
#             distribution = torch.tensor(distribution)
#         return distribution

#     def shuffle_parameters_and_data(self):
#         target, transport, valid_target, valid_transport = train_valid_split(self.target_all,
#                                                                              self.transport_all,
#                                                                              self.valid_ratio,
#                                                                              device=self.device)

#         # train
#         train = self._create_data(transport, target)
#         self.train_loader =  torch.utils.data.DataLoader(
#             torch.concat(train, 1), **self.dataloader_args)

#         # valid
#         self.distribution_valid = self._create_data(valid_transport, valid_target)
        

#     def _create_network(self):
#         "inti new network"
#         self.net = DenseNet(input_dim=self.input_dim).to(self.device)

#     def _create_data(self, transport, target):
#         "create torch data for discriminator"
#         if self.loss_str == "wasserstein":
#             data = (transport, target)
#         else:
#             target_truth = torch.ones((target.shape[0],1))
#             transport_truth = torch.zeros((transport.shape[0],1))
#             truth = torch.concat([target_truth,transport_truth], 0).to(self.device)
#             distribution = torch.concat([target,transport], 0)
#             data = (distribution, truth)
#         return data

#     def run_train_and_eval(self, epoch_nr, new_network=False, target=None, 
#                            transport=None, n_train_iter=1):
#         "run the training and evaluation framework"
#         if target is not None:
#             self.target_all = self._convert_to_tensor(target)
#         if transport is not None:
#             self.transport_all = self._convert_to_tensor(transport)

#         if (target is not None) | (transport is not None):
#             self.shuffle_parameters_and_data()

#         valid_list = []
#         for i in range(n_train_iter):
#             if new_network:
#                 self._create_network()
#             valid = self._train(nr = f"{epoch_nr}_{i}")
#             valid_list.append(valid)
#         return valid_list

#     def _train(self, nr, epochs=25):
#         "run the discriminator"
#         save_valid_curve_path = self.save_valid_curve_path
#         optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
#         valid_loss_list = []
#         loss = np.inf
#         for _ in range(epochs):
#             valid_dst, valid_target = self.distribution_valid
#             for x_train in self.train_loader:

#                 x, y = x_train[:, :self.input_dim], x_train[:,x_train.shape[1]-self.input_dim:]

#                 optimizer.zero_grad(set_to_none=True)
#                 output = self.net.forward(x)
#                 if self.loss_str == "wasserstein":
#                     y = self.net.forward(y)
#                 loss = self.loss_function(output, y)

#                 loss.backward()
#                 optimizer.step()

#             output = self.net.forward(valid_dst)
#             if self.loss_str == "wasserstein":
#                 valid_target = self.net.forward(valid_target)
#                 valid_loss_list.append(-self.valid_function(output, valid_target).cpu().detach().numpy())
#             else:
#                 valid_loss_list.append(self.valid_function(output, valid_target).cpu().detach().numpy())

#         if isinstance(save_valid_curve_path, str):
#             fig = plt.figure()
#             label = f"Minimum at {np.argmin(valid_loss_list)}, value at {np.min(valid_loss_list)}"
#             plt.plot(valid_loss_list, label = label)
#             plt.xlabel("Epoch")
#             plt.ylabel(r"valid$_{AUC}$")
#             if not ".png" in save_valid_curve_path:
#                 save_valid_curve_path += ".png"
#             save_valid_curve_path = save_valid_curve_path.replace(".png", f"_{nr}.png")
#             plt.savefig(save_valid_curve_path)
#             plt.close(fig)
#         auc = np.max(valid_loss_list)
#         return auc
