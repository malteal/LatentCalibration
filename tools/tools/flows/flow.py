import pathlib
from datetime import datetime
from glob import glob
import os

from tqdm import tqdm
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import hydra
from sklearn.model_selection import train_test_split
try:
    import pytorch_lightning as L
    lightning_activated=True
except ImportError:
    lightning_activated=False
from .get_flow import stacked_norm_flow
from ..schedulers import get_scheduler
from ..create_log_file import CreateLog
from .. import misc
from ..visualization import cflow_plotting as visual
from nflows.transforms.base import InputOutsideDomain

class Flow_old:
    # deprecated f.ow class. should use the lightning instead
    def __init__(self,flow_config, train_config, save_path=None, device="cuda",
                 **kwargs):
        self.device=device
        self.save_path=save_path
        self.flow_config=flow_config
        flow_config["device"]=device
        self.train_config=train_config
        self.scheduler=None
        self.verbose = kwargs.get("verbose", False)
        self.dataloader_style = kwargs.get("dataloader_style", 
                                           {"pin_memory":False,"batch_size":512,
                                            "shuffle":True, "drop_last": True
                                            })
        self.valid_dataloader=None
        self.y_scale = kwargs.get("y_scale", None)
        
        # init model
        self.xz_dim = flow_config["xz_dim"]
        self.ctxt_dim = flow_config.get("ctxt_dim", 0)
        self.flow = stacked_norm_flow(**flow_config).to(device)
        
        #init training
        self.optimizer = T.optim.Adam(self.flow.parameters(), lr=train_config["lr"])

        self.epoch_start_nr=0
        self.plotting=None
        if kwargs.get("old_path", None) is not None:
            self.load_old_model(kwargs["old_path"], kwargs.get("new_model_bool", False) )

        if save_path is not None:
            self.path_init(kwargs.get("additional_dirs", ["models",
                                                        "figures",
                                                        "figures/1d_marginals"]),
                        kwargs.get("add_time_to_dir", True))
        
            misc.save_yaml(train_config, self.save_path + "/train_config.yaml", hydra=True)
            misc.save_yaml(flow_config, self.save_path + "/model_config.yaml", hydra=True)

            self.log_file = CreateLog(self.save_path, ["loss", "lr", "mean_clip"])
            # if kwargs.get("new_model_bool", True):
                # init log

    def create_loaders(self, train, valid=None, valid_args:dict={}):
        "valid_args has to be args of train_test_split"
        if len(valid_args) and isinstance(valid_args, dict) and (valid is None):
            train, valid = train_test_split(train, **valid_args)
        if valid is not None:
            self.valid_dataloader = T.utils.data.DataLoader(valid,
                                                            **self.dataloader_style)
        self.train_dataloader = T.utils.data.DataLoader(train, **self.dataloader_style)
    
    def create_plotting(self, dist, conds_dist, **kwargs):
        self.plotting = visual.Plotting(self.flow,
                                   dist,
                                   conds_dist,
                                   device=self.device,
                                   **kwargs #n_times_sample=1, scaler=norm,
                                   #run_scaler=False, inverse_scale=True
                                   )

    def create_scheduler(self, dataloader=None):
        sch_attr = self.train_config.get("sch_config", {"eta_min": 1e-7})
        if dataloader is None:
            sch_attr["T_max"] = self.train_config["n_epochs"]*self.train_config["epoch_length"]
        else:
            sch_attr["T_max"] = self.train_config["n_epochs"]*len(dataloader)
        self.scheduler = get_scheduler("singlecosine", self.optimizer, attr=sch_attr)


    def path_init(self, additional_dirs, add_time_to_dir=True) -> None:
        if add_time_to_dir:
            self.save_path += f"/{datetime.today().strftime('%m_%d_%Y_%H_%M_%S_%f')}"
        
        os.makedirs(self.save_path, exist_ok=True)

        # create folders
        for i in additional_dirs:
            os.makedirs(f"{self.save_path}/{i}", exist_ok=True)
       
    
    def load_old_model(self, old_model, new_model_bool=False):
        if (old_model is not None) & (not new_model_bool):
            model_config = sorted(glob(f"{old_model}/models/*"), key=os.path.getmtime)[-1]
            model_state = T.load(model_config, map_location=self.device)
            self.flow.load_state_dict(model_state["flow"])
            self.optimizer.load_state_dict(model_state["optimizer_state_dict"])
            path_old_log = f"{old_model}/log.json"
        elif new_model_bool:
            model_config = sorted(glob(f"{old_model}/models/*"), key=os.path.getmtime)[-1]
            model_state = T.load(model_config, map_location=self.device)
            self.flow.load_state_dict(model_state["flow"])
            path_old_log = None
        else:
            path_old_log = None
        self.log_file = CreateLog(self.save_path, ["loss", "lr", "mean_clip"],
                            path_old_log=path_old_log)

        self.epoch_start_nr = len(self.log_file.log["loss"])
    
    def log_prob(self, inputs, context, weights=None):
        InputOutsideDomain_bool=True

        if (weights is None) or (weights.shape[-1]==0):
            weights = T.ones(inputs.shape[0], device=self.device)

        while InputOutsideDomain_bool:
            try:
                return weights*self.flow.log_prob(inputs, context)
            except InputOutsideDomain:
                print(f"Out of domain, trying again {inputs, context}")
    
    def flow_sample(self, n_samples, y):
        InputOutsideDomain_bool=True
        while InputOutsideDomain_bool:
            try:
                return self.flow.sample(n_samples, y)
            except InputOutsideDomain:
                print(f"Out of domain, trying again {x, y}")
        
    def train(self,train_dataloader=None, valid_dataloader=None):
        if train_dataloader is None:
            if self.train_dataloader is None:
                raise ValueError("To use internal train_dataloader, run create_loaders()")
            train_dataloader = self.train_dataloader
        if self.scheduler is None:
            self.create_scheduler(train_dataloader)
        if (valid_dataloader is not None):
            # valid_dataloader = self.valid_dataloader
            if "valid_loss" not in self.log_file.log:
                self.log_file.log["valid_loss"]=[]
            total_loss = self.log_file.log["valid_loss"].copy()
        else:
            total_loss = self.log_file.log["loss"].copy()

        pbar = tqdm(range(self.epoch_start_nr,self.train_config["n_epochs"]))

        for i in pbar:

            train_loss = []
            clip_lst = []
            for data_iter in train_dataloader:
                y = data_iter[:, :self.xz_dim].to(self.device)
                x = data_iter[:, self.xz_dim:self.xz_dim+self.ctxt_dim].to(self.device)
                weights = data_iter[:, self.xz_dim+self.ctxt_dim:].to(self.device)
                self.optimizer.zero_grad()
                # print(x.min(0))
                loss = -self.log_prob(inputs=x, context=y, weights=weights).mean()
                train_loss.append(loss.cpu().detach().numpy())
                loss.backward()
                clip_values = T.nn.utils.clip_grad_norm_(self.flow.parameters(), 10)
                self.optimizer.step()
                self.scheduler.step()
                clip_lst.append(clip_values.cpu().numpy())

            if valid_dataloader is not None:
                valid_loss = [np.ravel(
                    -self.log_prob(
                        inputs=i[:, self.xz_dim:self.xz_dim+self.ctxt_dim].to(self.device),
                        context=i[:, :self.xz_dim].to(self.device),
                        weights=i[:, self.ctxt_dim+self.xz_dim:].to(self.device)
                        ).cpu().detach().numpy())
                    for i in valid_dataloader
                    ]
                epoch_loss = np.float64(np.concatenate(valid_loss,0).mean())
                total_loss.append(epoch_loss)
                log_values = [np.float64(np.mean(train_loss)),
                              self.optimizer.param_groups[0]["lr"],
                              np.mean(clip_lst), epoch_loss]
            else:
                epoch_loss = np.float64(np.mean(train_loss))
                total_loss.append(epoch_loss)
                log_values = [epoch_loss, self.optimizer.param_groups[0]["lr"], np.mean(clip_lst)]

            if (i==0) or (np.min(total_loss[:-1]) > total_loss[-1]):
                if self.plotting is not None:
                    self.plotting.update_sample(self.flow)
                    self.plotting.marginals()
                    fig, _, _ = self.plotting.plot_marginals(y_scale=self.y_scale)
                    misc.save_fig(
                        fig,
                        f"{self.save_path}/figures/1d_marginals/epoch_nr_{i}.png",
                        close_fig=True,
                    )
                best_model_path = self.save_path+f"/models/model_setup_{i}.pt"
                T.save(
                    {
                        "flow": self.flow.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                    },
                    best_model_path,
                )

            pbar.set_postfix({"Flow loss": f"{str(round(total_loss[-1],4))}"})


            log = dict(zip(self.log_file.keys(), log_values))
            self.log_file.update_log(log)
            self.log_file.save()
            
            fig = plt.figure()
            for i in [i for i in self.log_file.log if "loss" in i]:
                plt.plot(self.log_file.log[i], label=i)
            plt.xlabel("Epoch number")
            plt.ylabel("Loss")
            plt.legend()
            if not (np.min(total_loss) < 0):
                plt.yscale("log")
            misc.save_fig(fig, self.save_path+"/figures/flow_loss.png",close_fig=True)

if lightning_activated:

    class Flow(L.LightningModule):
        def __init__(self, train_config):
            super().__init__()
            self.train_config = train_config
            self.plotting = None

        def configure_optimizers(self):
            "configure optimizer and scheduler. If scheduler is not in train_config, it will be None."
            
            optimizer = {"optimizer":
                T.optim.AdamW(self.parameters(), lr=self.train_config["lr"])}

            if "sch_config" in self.train_config: # config for scheduler
                optimizer["lr_scheduler"] = {
                    "scheduler": self.train_config["sch_config"](optimizer=optimizer["optimizer"]),
                    "interval": "step",
                    "frequency": 1
                    }

            return optimizer

        def get_progress_bar_dict(self):
            tqdm_dict = self.get_progress_bar_dict()
            tqdm_dict.pop("v_num", None)
            return tqdm_dict

        def get_flow_ctxt(self, **kwargs) -> tuple:
            raise NotImplementedError

        def create_plotting(self, dist, conds_dist, **kwargs):
            self.plotting = visual.Plotting(self.flow,
                                    dist,
                                    conds_dist,
                                    device=self.device,
                                    **kwargs #n_times_sample=1, scaler=norm,
                                    #run_scaler=False, inverse_scale=True
                                    )
        
        def load_old_model(self, old_model, new_model_bool=False):
            if (old_model is not None) & (not new_model_bool):
                model_config = sorted(glob(f"{old_model}/models/*"), key=os.path.getmtime)[-1]
                model_state = T.load(model_config, map_location=self.device)
                self.flow.load_state_dict(model_state["flow"])
                self.optimizer.load_state_dict(model_state["optimizer_state_dict"])
                path_old_log = f"{old_model}/log.json"
            elif new_model_bool:
                model_config = sorted(glob(f"{old_model}/models/*"), key=os.path.getmtime)[-1]
                model_state = T.load(model_config, map_location=self.device)
                self.flow.load_state_dict(model_state["flow"])
                path_old_log = None
            else:
                path_old_log = None
            self.log_file = CreateLog(self.save_path, ["loss", "lr", "mean_clip"],
                                path_old_log=path_old_log)

            self.epoch_start_nr = len(self.log_file.log["loss"])

        def _shared_step(self, batch:dict, batch_idx:int):

            x = batch["images"].to(self.device)

            if "ctxt" in batch:
                y = self.get_flow_ctxt(batch["ctxt"])

            loss = -self.flow.log_prob(inputs=x, context=y).mean()
            return loss


        def training_step(self, batch:dict, batch_idx:int):
            
            loss= self._shared_step(batch, batch_idx)
            self.log("train/loss", loss, prog_bar=True)
            
            # # step with lr scheduler
            # if self.lr_schedulers() is not None:
            #     self.lr_schedulers().step()

            return loss

        def validation_step(self, batch:dict, batch_idx:int):
            loss= self._shared_step(batch, batch_idx)
            self.log("valid/loss", loss, prog_bar=True)
            return loss
else:
    class Flow():
        def __init__(self, **kwargs):
            raise NotImplementedError("Pytorch lightning is not installed. Please install it to use the Flow class or use Flow_old")

    # def train(self,train_dataloader=None, valid_dataloader=None):
    #     if train_dataloader is None:
    #         if self.train_dataloader is None:
    #             raise ValueError("To use internal train_dataloader, run create_loaders()")
    #         train_dataloader = self.train_dataloader
    #     if self.scheduler is None:
    #         self.create_scheduler(train_dataloader)
    #     if (valid_dataloader is not None):
    #         # valid_dataloader = self.valid_dataloader
    #         if "valid_loss" not in self.log_file.log:
    #             self.log_file.log["valid_loss"]=[]
    #         total_loss = self.log_file.log["valid_loss"].copy()
    #     else:
    #         total_loss = self.log_file.log["loss"].copy()

    #     pbar = tqdm(range(self.epoch_start_nr,self.train_config["n_epochs"]))

    #     for i in pbar:

    #         train_loss = []
    #         clip_lst = []
    #         for data_iter in train_dataloader:
    #             y = data_iter[:, :self.ctxt_dim].to(self.device)
    #             x = data_iter[:, self.ctxt_dim:].to(self.device)
    #             self.optimizer.zero_grad()
    #             # print(x.min(0))
    #             loss = -self.flow.log_prob(inputs=x, context=y).mean()
    #             train_loss.append(loss.cpu().detach().numpy())
    #             loss.backward()
    #             clip_values = T.nn.utils.clip_grad_norm_(self.flow.parameters(), 10)
    #             self.optimizer.step()
    #             self.scheduler.step()
    #             clip_lst.append(clip_values.cpu().numpy())

    #         if valid_dataloader is not None:
    #             valid_loss = [
    #                 -self.flow.log_prob(inputs=i[:, self.ctxt_dim:].to(self.device),
    #                                     context=i[:, :self.ctxt_dim].to(self.device)
    #                                     ).mean().cpu().detach().numpy()
    #                 for i in valid_dataloader
    #                 ]
    #             epoch_loss = np.float64(np.mean(valid_loss))
    #             total_loss.append(epoch_loss)
    #             log_values = [np.float64(np.mean(train_loss)),
    #                           self.optimizer.param_groups[0]["lr"],
    #                           np.mean(clip_lst), epoch_loss]
    #         else:
    #             epoch_loss = np.float64(np.mean(train_loss))
    #             total_loss.append(epoch_loss)
    #             log_values = [epoch_loss, self.optimizer.param_groups[0]["lr"], np.mean(clip_lst)]

    #         if (i==0) or (np.min(total_loss[:-1]) > total_loss[-1]):
    #             if self.plotting is not None:
    #                 self.plotting.update_sample(self.flow)
    #                 self.plotting.marginals()
    #                 fig, _, _ = self.plotting.plot_marginals(y_scale=self.y_scale)
    #                 misc.save_fig(
    #                     fig,
    #                     f"{self.save_path}/figures/1d_marginals/epoch_nr_{i}.png",
    #                     close_fig=True,
    #                 )
    #             best_model_path = self.save_path+f"/models/model_setup_{i}.pt"
    #             T.save(
    #                 {
    #                     "flow": self.flow.state_dict(),
    #                     "optimizer_state_dict": self.optimizer.state_dict(),
    #                     "scheduler_state_dict": self.scheduler.state_dict(),
    #                 },
    #                 best_model_path,
    #             )

    #         pbar.set_postfix({"Flow loss": f"{str(round(total_loss[-1],4))}"})


    #         log = dict(zip(self.log_file.keys(), log_values))
    #         self.log_file.update_log(log)
    #         self.log_file.save()
            
    #         fig = plt.figure()
    #         for i in [i for i in self.log_file.log if "loss" in i]:
    #             plt.plot(self.log_file.log[i], label=i)
    #         plt.xlabel("Epoch number")
    #         plt.ylabel("Loss")
    #         plt.legend()
    #         if not (np.min(total_loss) < 0):
    #             plt.yscale("log")
    #         misc.save_fig(fig, self.save_path+"/figures/flow_loss.png",close_fig=True)


if __name__ == "__main__":
    bkg_path = train_context_template("ftag_h5_bkg")
    sig_path = train_context_template("ftag_h5_sig")
