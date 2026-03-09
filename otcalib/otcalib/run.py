"""This scripts will run the training loop for optimal transport"""
import json
import logging
import os
from time import time
from typing import Any, Union
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from .torch import torch_utils as utils
from .torch.layers import GRL
from .utils import plotutils
from .utils import transformations as trans
from .evaluate import densenet as dn
from .evaluate import evaluate
import corner
from sklearn.decomposition import PCA

from tools.tools import misc


class DualTraining:
    """Dual training for two (P)ICNN to find the optimal transport between two
    continuous densities"""

    def __init__(
        self,
        f_func: callable,
        g_func: callable,
        device: str = "cuda",
        verbose: bool = False,
        GRL_bool: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the start parameter of OT training

        Parameters
        ----------
        f_func : callable
            ML model
        g_func : callable
            ML model
            list of conditional bins to evaluate in. Only works for 1d conds
        outdir : str, optional
            output folder , by default "tb"
        verbose : bool, optional
            How much to print, by default False
        """
        self.f_per_g = kwargs.get("f_per_g", 1)
        self.g_per_f = kwargs.get("g_per_f", 1)
        self.grad_clip = kwargs.get("grad_clip", {"f": 5, "g": 5})
        self.epoch_size = kwargs.get("epoch_size", 100)
        self.burn_in = kwargs.get("burn_in", 0)
        self.GRL_bool = GRL_bool
        self.GRL = GRL(alpha=1) if self.GRL_bool else torch.nn.Identity()

        self.ckpt_path = kwargs.get("ckpt_path")
        self.epoch_nr = kwargs.get("epoch_nr", 0)

        self.optimizer_args_f = kwargs.get(
            "optimizer_args_f",
            {
                "name": "adamw",
                "args": {
                    "lr": 1e-4,
                    "betas": (0.0, 0.9),
                },
            },
        )
        self.optimizer_args_g = kwargs.get(
            "optimizer_args_g",
            {
                "name": "adamw",
                "args": {
                    "lr": 1e-4,
                    "betas": (0.0, 0.9),
                },
            },
        )
        self.sch_args_g = kwargs.get("sch_args_g", None)
        self.sch_args_f = kwargs.get("sch_args_f", None)

        # set new lenght of CosineAnnealingLR
        if self.sch_args_g is not None:
            if (
                ("CosineAnnealingLR" in self.sch_args_g.get("name", ""))
                and
                (kwargs.get("nepochs", ) is not None)
                and ('T_max' not in self.sch_args_g["args"])
                ) :
                self.sch_args_g["args"]["T_max"] = self.epoch_size*kwargs.get("nepochs")
                self.sch_args_f["args"]["T_max"] = self.sch_args_g["args"]["T_max"]

        self.outdir = kwargs.get("outdir", "/")

        self.log = kwargs.get("log", {"steps_f": [0], "steps_g": [0]})

        self.device = device
        self.verbose = verbose
        self.f_func = f_func
        self.g_func = g_func

        self.noncvx_dim = (
            0
            if not hasattr(self.f_func, "weight_uutilde")
            else self.f_func.weight_uutilde[0].shape[1]
        )
        self.cvx_dim = self.f_func.cvx_dim

        if not os.path.exists(self.outdir):
            self.save_config()

        # creating folders
        os.makedirs(self.outdir + "/training_setup", exist_ok=True)
        os.makedirs(self.outdir + "/plots", exist_ok=True)
        os.makedirs(self.outdir + "/plots/discriminator", exist_ok=True)

        # save setup
        utils.save_config(
            outdir=self.outdir,
            values=self.__dict__.copy(),
            drop_keys=[
                "metric",
                "discriminator",
                "distribution",
                "f_func_optim",
                "f_scheduler",
                "f_func",
                "g_func_optim",
                "g_scheduler",
                "g_func",
                "GRL",
            ],
            file_name="train_config",
        )
        self.init_training()
        
    def init_training(self):  # TODO bad name
        # init standard optimizer & scheduler
        self.f_func_optim, self.f_scheduler = utils.get_optimizer(
            self.f_func.parameters(), self.optimizer_args_f, self.sch_args_f
        )
        self.g_func_optim, self.g_scheduler = utils.get_optimizer(
            self.g_func.parameters(), self.optimizer_args_g, self.sch_args_g
        )
        
        # load optimizer and scheduler for checkpoint 
        if self.ckpt_path:
            # load old log
            self.log = misc.load_json(f"{self.ckpt_path}/log.json")
            
            # load old state of optimizer and scheduler
            self.load_state(self.ckpt_path, last_epoch=self.epoch_nr)
            

        # init Metrics
        self.metric = None
        # self.metric = Metrics(geomloss_input=[])

        # define discriminator
        self.dense_net_args = {
            "input_dim": self.f_func.cvx_dim + self.f_func.noncvx_dim,
            "N": 64,
            "n_layers": 4,
            "output_size": 1,
            "device": self.device,
        }

        self.discriminator = dn.DenseNet(**self.dense_net_args)

    def load_state(self, old_model, last_epoch):
        "load old state of optimizer and scheduler"
        
        if last_epoch==0:
            # get last epoch - remember zero indexed
            last_epoch = len(self.log['loss_g'])
            # last_epoch = len(glob(f"{old_model}/training_setup/*"))-1
            
            # used in the progress bar with should start at the next epoch
            self.epoch_nr = last_epoch+1
        
        state_dict = torch.load(f"{old_model}/training_setup/checkpoint_{last_epoch}.pth")
        
        # load state for optimizers
        self.f_func_optim.load_state_dict(state_dict['f_optimizer'])
        self.g_func_optim.load_state_dict(state_dict['g_optimizer'])

        # load state for schedulers
        if not isinstance(state_dict['f_scheduler'], str):
            self.f_scheduler.load_state_dict(state_dict['f_scheduler'])
            self.g_scheduler.load_state_dict(state_dict['g_scheduler'])
        
        # load state for models
        self.f_func.load_state_dict(state_dict['f_func'])
        self.g_func.load_state_dict(state_dict['g_func'])

    def load_model(self, f_path: str, g_path: str):
        self.f_func.load(f_path)
        self.g_func.load(g_path)

    def evaluate(
        self,
        data: dict,
        epoch: int,
        datatype: str = "",
        plot_figures: Union[bool,int]  = False,
        run_metrics: bool = False,
        discriminator_str: str = None,
        **kwargs,
    ):
        """
        evaluate model performance
        This is shity evaluate function and should be rewritting (LOL) - M.A

        Parameters
        ----------
        data : dict
            list of torch.Tensor of source, target, valid_source, valid_target
        epoch : int
            the epoch number
        datatype : str
            toy or ftag
        Returns
        -------
        dict
            return the updated log dict
        """
        if self.verbose:
            logging.info("Evaluationing performance")
        self.g_func.eval()
        self.f_func.eval()

        # evaluate performance
        metric = {}

        # classification
        # Transport values
        for conds_val, val in data.items():
            for name, values in val.items():
                if conds_val not in metric:
                    metric[conds_val] = {}

                if name == "truth":
                    continue

                transport = self.g_func.chunk_transport(
                    values["transport"],
                    values.get("conds", None),
                    values.get("sig_mask", None),
                    n_chunks=len(values["transport"]) // 10_000 + 1,
                )

                sig_mask = values.get("sig_mask", np.ones(len(transport)) == 1)

                # calculate transport cost
                transport_cost = np.mean(
                    ((transport[sig_mask] - values["transport"][sig_mask]) ** 2)
                    .detach()
                    .numpy()
                )

                # add transport to dict
                data[conds_val][name]["eval_transport"] = transport.cpu().detach()

                metric[conds_val][name] = {"transport_cost": transport_cost}

                if self.metric is not None:  # TODO not being used atm
                    metric[conds_val][f"o-{name}"] = {}
                    eval_truth = data[conds_val]["truth"]["transport"]
                    if "ftag" in datatype:
                        eval_truth = trans.dl1r(
                            trans.probsfromlogits(data[conds_val]["truth"]["transport"])
                        )
                    for sub_dict, value in zip(
                        ["o-", ""], [values["transport"], values["eval_transport"]]
                    ):
                        if "ftag" in datatype:
                            value = trans.dl1r(trans.probsfromlogits(value))

                        metric[conds_val] = self.metric.run(
                            source=value,
                            target=eval_truth,
                            append=metric[conds_val],
                            sub_dict=sub_dict + name,
                            # column_name = f"_{sub_dict}"
                        )

        if run_metrics & (self.metric is not None):  # TODO not sure if needed
            average_wasserstein = {}
            # run over all conds keys if the conds are split
            for conds_val in data.keys():
                for name, values in metric[conds_val].items():
                    if "total" in name:
                        continue
                    metric_name = "wasserstein"

                    metric[conds_val][name]["average_wasserstein"] = np.mean(
                        values[metric_name]
                    )
                    if name not in average_wasserstein:
                        average_wasserstein[name] = []
                    average_wasserstein[name].append(np.mean(values[metric_name]))
            average_wasserstein = {
                f"{i}_average_wasserstein": np.mean(j)
                for i, j in average_wasserstein.items()
            }
            metric.update(average_wasserstein)

        if isinstance(discriminator_str, str) & (epoch > self.burn_in):
            metric.update(
                evaluate.run_clf_evaluation(
                    self.discriminator,
                    discriminator_str=discriminator_str,
                    data=data,
                    first_training=(epoch == self.burn_in + 1),
                )
            )

            self.discriminator.plot_log(
                f"{self.outdir}/plots/discriminator/plots_{epoch}",
            )  # TODO plot each loss batch instead of epoch

        else:
            metric["AUC"] = 1.0
            metric["train_AUC"] = 1.0

        # update logging
        self.update_log(metric, epoch)

        torch.cuda.empty_cache()

        # save that prediction
        self.save(epoch=epoch)

        # plot figures
        if plot_figures and (epoch%plot_figures == 0): # and epoch > self.burn_in:
            plotutils.plot_training_setup(
                kwargs["source_iter"],
                kwargs["target_iter"],
                eval_data=data,
                outdir=f"{self.outdir}/plots/",
                plot_var=kwargs["condsnames"] + kwargs["transnames"],
                dist_labels=["Source", "Truth", "Transport"],
                generator=self.g_func,
                # n_samples=200_000 // 10,
                n_bins=30,
                log_bool=False,
                epoch=epoch,
                datatype=datatype,
                plot_mc_eval=kwargs.get("plot_mc_eval", False),
            )
        
            if kwargs.get('pca_bool', False) and (epoch>=50):
                os.makedirs(f"{self.outdir}/plots/pca", exist_ok=True)

                transport = data[conds_val][discriminator_str]["eval_transport"].numpy()

                target = data[conds_val]['truth']['transport'].numpy()

                pca = PCA(
                    n_components=data[conds_val][discriminator_str]["eval_transport"].shape[1]
                    )

                pca_target = pca.fit_transform(target)
                pca_transport = pca.transform(transport)

                logging.info(" Corner plot!")

                # corner plots between pythia and herwig
                ranges = [tuple(np.percentile(pca_target[:, i], [0.1,99.9])) for i in range(pca_target.shape[1])]
                labels = [f'Latent {i+1}' for i in range(pca_target.shape[1])]

                figure=None

                kwargs = {'plot_density': False, 'plot_datapoints': False, 'range': ranges,
                            'bins':30}

                figure= corner.corner(pca_target, color = 'red', **kwargs)

                corner.corner(pca_transport, color = 'blue',
                            fig=figure, labels=labels, labelpad=0.1,
                            **kwargs)
                
                misc.save_fig(figure, f"{self.outdir}/plots/pca/pca_{epoch}.png")


        torch.cuda.empty_cache()

        return metric

    def update_log(self, logging_values, epoch_nr: int = -1) -> dict:
        """Saving training information

        Parameters
        ----------
        logging_values : list
            list of list of values that should be logged

        Returns
        -------
        dict
            return dict where the new metrics have been added
        """
        if "epoch" in logging_values:
            logging_values.pop("epoch")

        if epoch_nr == 0:
            for i in logging_values.keys():
                if isinstance(logging_values[i], dict):
                    self.log[i] = {}
                    for j in logging_values[i].keys():
                        self.log[i][j] = {}
                        for k in logging_values[i][j].keys():
                            # if isinstance(logging_values[i][j], dict):
                            self.log[i][j][k] = []
                            # else:
                            #     self.log[i][j] = []
                else:
                    self.log[i] = []
        else:
            keys = np.array(list(logging_values.keys()))

            sub_keys = keys[np.in1d(keys, list(self.log.keys()))]
            mask_dict = np.array(
                [isinstance(logging_values[i], dict) for i in sub_keys]
            )
            if (len(sub_keys) > 0) and any(mask_dict):
                for i in sub_keys[mask_dict]:
                    sub_keys_logging = np.array(list(logging_values[i].keys()))
                    sub_keys_logging = sub_keys_logging[
                        ~np.in1d(sub_keys_logging, list(self.log[i].keys()))
                    ]
                    for j in sub_keys_logging:
                        self.log[i][j] = []

            add_keys = keys[~np.in1d(keys, list(self.log.keys()))]
            for i in add_keys:
                if isinstance(logging_values[i], dict):
                    self.log[i] = {}
                    for j in logging_values[i].keys():
                        # if True:  # isinstance(logging_values[i][j], dict):
                        self.log[i][j] = {}
                        for k in logging_values[i][j].keys():
                            self.log[i][j][k] = []
                        self.log[i][j] = []
                else:
                    self.log[i] = []

        for i, j in logging_values.items():
            if isinstance(j, dict):
                for j, k in j.items():
                    if isinstance(k, dict):
                        for k, l in k.items():
                            self.log[i][j][k].append(
                                l if isinstance(l, list) else np.float64(l)
                            )
                    else:
                        self.log[i][j].append(
                            k if isinstance(k, list) else np.float64(k)
                        )
            else:
                self.log[i].append(j if isinstance(j, list) else np.float64(j))

    def dual_formulation(
        self, 
        model_disc: torch.nn.Module,
        model_transport: torch.nn.Module,
        source: list,
        target: list = None,
        run_disc: bool = False,
    ):
        """Kantorovich dual formulation

        Parameters
        ----------
        model_disc : torch.nn.Module
            discriminator model
        model_transport : torch.nn.Module
            Generator
        source : list
            condtions, transport_var & sig_mask
        target : list
            condtions, target_var & sig_mask
        run_disc : bool
            Which formulation to run. Train either f or g

        Returns
        -------
        _type_
            _description_
        """
        conditionals, totransport, signal_bool = source

        signal_bool = signal_bool.flatten()

        # nabla g(x, theta), g(x, theta)
        trans = totransport.clone()
        trans[signal_bool] = model_transport.transport(
            totransport[signal_bool],
            conditionals[signal_bool],
        )[0]

        # nabla f(nabla g(x, theta), theta), f(nabla g(x, theta), theta)
        cvx_disc_trans = model_disc(self.GRL(trans), conditionals)

        if self.GRL_bool:
            targetconds, targettrans, _ = target
            loss = (model_disc(targettrans, targetconds)
                    - self.GRL.alpha * torch.sum(trans * totransport, keepdim=True, dim=1)
                    - cvx_disc_trans)
        elif run_disc:
            targetconds, targettrans, _ = target
            # nabla f(x, theta'), f(x, theta')
            cvx_disc = model_disc(targettrans, targetconds)
            loss = cvx_disc - cvx_disc_trans
            # loss = cvx_disc.mean() - cvx_disc_trans.mean()
        else:
            # nabla f(nabla g(x, theta), theta), f(nabla g(x, theta), theta)
            loss = cvx_disc_trans - torch.sum(trans * totransport, keepdim=True, dim=1)

        loss = loss.mean()

        if torch.isnan(loss):
            print('trans:', trans)
            print('cvx trans:', cvx_disc_trans)
            if run_disc:
                print('cvx dics:', cvx_disc)
            # raise ValueError(f"NaN in loss functions of {'f' if run_disc else 'g'}")

        return loss, cvx_disc_trans

    def save(self, epoch: int) -> None:
        """save models and info TODO save onnx model every time?

        Parameters
        ----------
        epoch : int
            The epoch number
        """
        # save log file
        with open(self.outdir + "/log.json", "w", encoding="utf-8") as file_parameters:
            json.dump(self.log, file_parameters)

        # save network states
        states_to_save = [
            self.f_func_optim.state_dict(),
            self.g_func_optim.state_dict(),
            self.f_func.state_dict(),
            self.g_func.state_dict(),
        ]

        # save schedulers
        if self.f_scheduler is not None:
            states_to_save.extend(self.f_scheduler.state_dict())
        if self.g_scheduler is not None:
            states_to_save.extend(self.g_scheduler.state_dict())

        # fill it into dict
        save_state = {}
        for i, name in zip(
            states_to_save,
            [
                "f_optimizer",
                "g_optimizer",
                "f_func",
                "g_func",
                "f_scheduler",
                "g_scheduler",
            ],
        ):
            save_state[name] = i

        torch.save(save_state, f"{self.outdir}/training_setup/checkpoint_{epoch}.pth")

    def pretrain_models(
        self,
        sourcedset: iter,
        targetdset: iter,
        stop_training: float = 1e-5,
        n_epochs=5_000,
        pretrain_activated=True,
    ): 
        "pretrain model to the identity"
        g_optimizer = torch.optim.AdamW(self.g_func.parameters(), lr=1e-4,
                                        weight_decay=0)
        f_optimizer = torch.optim.AdamW(self.f_func.parameters(), lr=1e-4,
                                        weight_decay=0)

        f_loss = np.inf
        g_loss = np.inf

        pbar = tqdm(range(n_epochs))

        for _ in pbar:
            source = next(sourcedset)
            target = next(targetdset)

            g_optimizer.zero_grad(set_to_none=True)
            f_optimizer.zero_grad(set_to_none=True)
            
            # sort to truth
            source = [source[1], source[0]]
            target = [target[1], target[0]]

            # MSE to the identity
            g_loss = (
                ((self.g_func.transport(*source)[0] - source[0]) ** 2).sum(dim=1).mean()
            )

            # f_loss = (
            #     ((self.f_func.transport(*target)[0] - target[0]) ** 2).sum(dim=1).mean()
            # )

            if g_loss.isnan() or g_loss.isinf():
                print("nan or inf loss")
            if pretrain_activated:
                # f_loss.backward()
                g_loss.backward()

                g_optimizer.step()
                # f_optimizer.step()

            pbar.set_postfix({"Loss g": g_loss.item()})
            # pbar.set_postfix({"Loss f": f_loss.item(), "Loss g": g_loss.item()})

            if (g_loss.item() < stop_training): #and (f_loss.item() < stop_training):
                break
        print(f"Final Loss: {g_loss.item()}")
        # print(f"Final Loss: {f_loss.item()} / {g_loss.item()}")

    def _train_step(
        self, sourcedset: iter, targetdset: iter, pbar: callable = None
    ) -> dict:
        """run a epoch

        Parameters
        ----------
        sourcedset : iter
            source dataset
        targetdset : iter
            target dataset
        epoch : int
            the epoch number

        Returns
        -------
        dict
            return the information dict
        """
        start_time = time()

        loss_dict_all = {}

        # load_time={"data":[], "loss":[]}
        loss_dict_iter = {
            "f": {"loss": [], "lr": [], "cycle_error": [], "clip": []},
            "g": {"loss": [], "lr": [], "cycle_error": [], "clip": []},
        }
        optimization_order = [False, True]
        for _ in tqdm(range(self.epoch_size), leave=False,
                      desc="Epoch iterations", ):
            for nr, (iterations, wdist) in enumerate(
                zip([self.g_per_f, self.f_per_g], optimization_order)
            ):
                model_name = "g" if nr == 0 else "f"
                if nr == 0:  # g network
                    self.f_func.eval()
                    self.g_func.train()
                else:  # f network
                    self.f_func.train()
                    self.g_func.eval()

                for iteration in range(iterations):

                    # ensure no gradient
                    self.g_func_optim.zero_grad(set_to_none=True)
                    self.f_func_optim.zero_grad(set_to_none=True)

                    source = next(sourcedset)
                    
                    # ICNN do not need the target conditions to be align
                    # course there are none
                    target = None
                    if (source[0].shape[-1] != 0 or model_name=='f'
                        or self.GRL_bool):
                        target = next(targetdset)
                        
                        if len(target[1]) != len(source[1]):
                            raise ValueError("The sampled source and target do not have the same length")


                    loss, cvx = self.dual_formulation(
                        self.f_func, self.g_func, source, target, wdist
                    )

                    loss.backward()
                    
                    # clip gradient
                    if self.GRL_bool:
                        f_clip = torch.nn.utils.clip_grad_norm_(
                            self.f_func.parameters(), self.grad_clip[model_name]
                        )
                        g_clip = torch.nn.utils.clip_grad_norm_(
                            self.g_func.parameters(),self.grad_clip[model_name]
                        )
                        clip = f_clip if wdist else g_clip
                    else:
                        clip = torch.nn.utils.clip_grad_norm_(
                            self.f_func.parameters() if wdist else self.g_func.parameters(),
                            self.grad_clip[model_name]
                        )

                    if torch.isnan(clip):
                        raise ValueError(
                            f"NaN in {model_name}_func at {iteration} iteration"
                            )

                    # gradient step for networks
                    if self.GRL_bool:
                        self.g_func_optim.step()
                        self.f_func_optim.step()
                    elif nr == 0:
                        self.g_func_optim.step()
                    else:
                        self.f_func_optim.step()

                    # log values
                    loss_dict_iter[model_name]["loss"].append(loss.detach())
                    loss_dict_iter[model_name]["clip"].append(clip)

                if self.GRL_bool:
                    self.g_scheduler.step()
                    self.f_scheduler.step()
                elif nr == 0:  # always start by training g
                    self.g_scheduler.step()
                else:
                    self.f_scheduler.step()

        # update log
        for model_name in ["f", "g"]:
            loss = torch.stack(loss_dict_iter[model_name]["loss"]).cpu().numpy()
            clip = torch.stack(loss_dict_iter[model_name]["clip"]).cpu().numpy()
            loss_dict_all[f"loss_{model_name}"] = np.mean(loss)
            loss_dict_all[f"loss_{model_name}_abs_log"] = np.log(np.mean(np.abs(loss)))
            loss_dict_all[f"{model_name}_clip"] = np.mean(clip)
            loss_dict_all[f"steps_{model_name}"] = (
                self.log[f"steps_{model_name}"][-1]
                + self.g_per_f * self.f_per_g * self.epoch_size
            )
        loss_dict_all[f"lr_g"] = self.g_func_optim.param_groups[0]["lr"]
        loss_dict_all[f"lr_f"] = self.f_func_optim.param_groups[0]["lr"]
        self.update_log(loss_dict_all)

        if self.verbose:
            print("Epoch time: ", time() - start_time)

        if pbar is not None:
            pbar.set_postfix(
                {
                    "Log loss F": f"{str(round(loss_dict_all['loss_f_abs_log'],3))}",
                    "Loss G": f"{str(round(loss_dict_all['loss_g'],3))}",
                }
            )

        if (
            np.isnan(loss_dict_all["loss_g"])
            or np.isnan(loss_dict_all["loss_f"])
            or loss_dict_all["loss_f_abs_log"] > 30
        ):
            raise ValueError("NaN in loss functions")


        # eval mode
        self.f_func.eval()
        self.g_func.eval()

        return pbar

    def train(self, *args: Any, **kwds: Any) -> Any:
        return self._train_step(*args, **kwds)
