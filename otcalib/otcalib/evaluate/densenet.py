"To run evaluation discriminator"

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as T
from sklearn import metrics
from torch import nn
from tqdm import tqdm
from typing import List, Tuple

from otcalib.utils import misc


class DenseNet(nn.Module):
    """A very simple sense network that is initialised
    with an input and output dimension."""

    def __init__(
        self, input_dim, N=32, n_layers=4, output_size=1, device="cpu", **kwargs
    ):
        super().__init__()
        self.device = device
        self.hidden_features = N
        self.train_loader = None
        self.valid_loader = None
        self.lr_scheduler = None
        self.lr_scheduler_args = {}
        self.dataloader_style = {
            "pin_memory": False,
            "batch_size": 512,
            "shuffle": True,
            "drop_last": True,
        }
        activation = nn.LeakyReLU()

        self.input_dim = input_dim
        network = [
            nn.Linear(input_dim, N),
            activation,
        ]

        # extend with hidden layers
        hidden_layers = [[nn.Linear(N, N), activation] for _ in range(n_layers)]
        network.extend(list(np.ravel(hidden_layers)))

        # output layer
        network.extend([nn.Linear(N, output_size)])
        network.append(nn.Sigmoid())
        self.loss = nn.BCELoss()

        self.network = nn.Sequential(*network).to(device)
        # eval
        self.best_predictions = None
        self.valid_truth_for_best = None

        # eval lst
        self.loss_data = {}
        self.loss_lst = []
        self.output_lst = []
        self.valid_truth = []

        # set optimizer
        self.set_optimizer(kwargs.get("adam", True), kwargs.get("lr", 1e-3))
        # define loss
        self.state_of_best = None
        self.init_new_log_dict()

    def set_optimizer(self, adam, lr=1e-3):
        if adam:
            self.optimizer = T.optim.Adam(self.parameters(), lr)
        else:
            self.optimizer = T.optim.SGD(self.parameters(), lr)

    def init_new_log_dict(self):
        "Creating logging dict"
        self.loss_data = {
            f"{i}_{j}": [] for i in ["train", "valid"] for j in ["loss", "auc"]
        }
        self.loss_data["lr"] = []

    def load(self, path, key_name="model"):
        model_state = T.load(path, map_location=self.device)
        self.load_state_dict(model_state[key_name])
        if "optimizer" in model_state:
            self.optimizer.state_dict = model_state["optimizer"]
        if "loss" in model_state:
            self.loss_data = model_state["loss"]

    def save(self, output_path: str, **kwargs):
        """save state_dict

        Parameters
        ----------
        output_path : str
            path to output the saved model should end with .pt or .pth
        kwargs :
            Should be state_dict values or loss like values
        """

        state_dict = {i: j for i, j in kwargs.items()}
        state_dict["model"] = self.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["loss"] = self.loss_data
        T.save(state_dict, output_path)

    def run_training(self, train_loader, valid_loader, n_epochs=50, **kwargs):
        pbar = tqdm(range(n_epochs), disable=not kwargs.get("verbose", True))

        for _ in pbar:
            loss_lst = []
            for batch in train_loader:
                self.optimizer.zero_grad()
                x = batch[:, : self.input_dim].to(self.device)
                y = batch[:, self.input_dim :].to(self.device)
                x.requires_grad = True
                y.requires_grad = True
                output = self.forward(x)

                loss = self.loss(output[1].view(-1), y.view(-1))
                loss.backward()
                if kwargs.get("clip_bool", False):
                    T.nn.utils.clip_grad_norm_(self.parameters(), 10)
                self.optimizer.step()
                loss_lst.append(loss.cpu().detach().numpy())
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if valid_loader is not None:
                self.evaluate(valid_loader, log_val_bool="valid")

            self.loss_data["train_loss"].append(float(np.mean(loss_lst)))
            self.loss_data["lr"].append(self.optimizer.param_groups[0]["lr"])
            pbar.set_postfix(
                {"Train loss": f"{str(round(self.loss_data['train_loss'][-1],4))}"}
            )

        self.load_state_dict(self.state_of_best)

    def forward(self, x):
        p_hat = self.network(x)

        r_hat = (1 - p_hat) / p_hat

        return r_hat, p_hat

    def evaluate(self, loader: T.utils.data.DataLoader,
                             log_val_bool:str=None) -> float:
        loss_lst = []
        output_lst = []
        valid_truth = []
        for batch in loader:
            x, y = (
                batch[:, : self.input_dim].to(self.device),
                batch[:, self.input_dim :].to(self.device),
            )
            x.requires_grad = True
            y.requires_grad = True
            output = self.forward(x)
            loss = self.loss(output[1].view(-1), y.view(-1))

            loss_lst.append(loss.cpu().detach().numpy())
            output_lst.append(output[1].cpu().detach().numpy())
            valid_truth.append(y.cpu().detach().numpy())

        self.valid_truth = np.concatenate(valid_truth, 0)
        self.output_lst = np.concatenate(output_lst, 0)
        self.loss_lst = np.ravel(loss_lst)

        valid_auc = calculate_auc(self.output_lst, self.valid_truth)

        if isinstance(log_val_bool, str):
            if all(valid_auc > self.loss_data[f"{log_val_bool}_auc"]):
                self.state_of_best = copy.deepcopy(self.state_dict())
                self.best_predictions = self.output_lst.copy()
                self.valid_truth_for_best = self.valid_truth.copy()

        if isinstance(log_val_bool, str):
            self.loss_data[f"{log_val_bool}_auc"].append(float(valid_auc))

            self.loss_data[f"{log_val_bool}_loss"].append(float(np.mean(self.loss_lst)))

        return valid_auc

    def plot_log(self, save_path=None):
        os.makedirs(f"{save_path}/plots/", exist_ok=True)

        for key, values in self.loss_data.items():
            if "valid" in key:
                continue
            fig = plt.figure()
            plt.plot(values, label="Train")
            plt.plot(self.loss_data[key.replace("train", "valid")], label="Valid")
            plt.legend()

            if all(np.array(values) > 0) & all(
                np.array(self.loss_data[key.replace("train", "valid")]) > 0
            ):
                plt.yscale("log")

            if save_path is not None:
                misc.save_fig(fig, f"{save_path}/plots/{key.replace('train', '')}.png")


def calculate_auc(pred, truth):
    "calculate auc in classification should be changed to torch function"
    fpr, tpr, _ = metrics.roc_curve(truth, pred)
    auc = metrics.auc(fpr, tpr)
    return auc
