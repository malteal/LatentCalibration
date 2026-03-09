"train flow to estimate p(N)"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
import wandb

import hydra
import numpy as np
import torch as T
import pytorch_lightning as L
import matplotlib.pyplot as plt


# from src.eval_utils import EvaluateFramework, get_percentile

# from tools.tools.flows import stacked_norm_flow
from src.utils import fig2img
from tools.tools.visualization import general_plotting as plot
from tools.tools.modules import IterativeNormLayer, MinMaxLayer
from tools.tools.omegaconf_utils import instantiate_collection, save_config
from tools.tools import misc
import normflows as nf

def load_flow_from_path(path:str, dev:str="cuda"):
    "load flow from path and return it"
    flow_cfg = misc.load_yaml(f"{path}/.hydra/config.yaml")
    # flow = hydra.utils.instantiate(flow_cfg.model, device=dev)
    # state = T.load(f"{path}/checkpoints/last.ckpt", map_location=dev)
    flow = hydra.utils.get_class(flow_cfg.model._target_)
    flow.device=dev
    flow = flow.load_from_checkpoint(f"{path}/checkpoints/last.ckpt", map_location=dev, device=dev)
    return flow

def create_flow(input_dim:int, n_couplings:int, spline_kwargs:dict):
    
    flows = []
    for _ in range(n_couplings):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(input_dim, **spline_kwargs)]
        flows += [nf.flows.Permute(input_dim, mode='shuffle')]

    # Set base distribuiton
    q0 = nf.distributions.DiagGaussian(input_dim, trainable=False)
        
    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    
    return nfm

class Flow(L.LightningModule):
    def __init__(self, data_dims:dict, flow_cfg:dict,
                 train_config:dict, device="cuda", **kwargs):
        super().__init__()

        # save hyperparameters
        # self.save_hyperparameters()

        self.data_dims = data_dims
        self.flow_cfg = flow_cfg
        self.train_config = train_config
        self.target_norm = kwargs.get("target_norm", "standard")
        self.get_network()
        # self.to(device)
        
        # # validation step outputs
        self.validation_dict = {i: [] for i in ['sample']}
        # save hp
        self.save_hyperparameters()
        
    def get_network(self):
        # init flow and embedding network
        self.flow = create_flow(**self.flow_cfg).to(self.device)

        # Initialise the individual normalisation layers
        self.normaliser = IterativeNormLayer(
            (1,self.data_dims),
            max_iters=50_000)
        
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

    def training_step(self, batch:dict, batch_idx:int):
        
        loss= self._shared_step(batch, batch_idx)

        self.log("train/loss", loss, prog_bar=True)
        
        return loss
   
    def _shared_step(self, batch:dict, batch_idx:int):
        # normalise
        batch = self.normaliser(batch)

        # Compute loss
        loss = self.flow.forward_kld(batch)
        
        return loss
    
    def sample(self, n:int, ctxt:dict=None):

        return self.flow.sample(n)
    
    def inverse(self, x:T.Tensor):
        return self.flow.inverse(self.normaliser(x))
    
    def undo_inverse(self, x:T.Tensor):
        return self.normaliser.reverse(self.flow(x))

    def validation_step(self, batch:dict, batch_idx:int):
        "run validation batches and log results"
        # get val loss
        loss= self._shared_step(batch, batch_idx)
        self.log("valid/loss", loss, prog_bar=True)
        
        # norm batch - when plotting it is nice to see the normed, so we can check if tails are correct
        batch = self.normaliser(batch)

        self.validation_dict['sample'].append(batch)

        return loss

    def on_validation_epoch_end(self):
        """log validation results over all valid batches"""
        for i,j in self.validation_dict.items():
            if len(j)==0:
                continue
            self.validation_dict[i] = T.vstack(j).detach().cpu().numpy()

        if self.global_step>0:
            log_vals={}
            z, _ = self.flow.sample(10_000)
            # z = self.normaliser.reverse(z)
            z = z.detach().cpu().numpy()
            
            fig, ax = plt.subplots(1, z.shape[1], figsize=(z.shape[1]*6,6))

            for i in range(z.shape[1]):
                fig, plot.plot_hist(
                    z[:,i],
                    self.validation_dict['sample'][:, i], 
                    ax=ax[i])

            plt.tight_layout()
            
            log_vals["sampled_hist"] =  wandb.Image(fig2img(fig))
            
            # log images
            if self.logger is not None:
                self.logger.experiment.log(log_vals, commit=False)
            
        # free memory
        for i in self.validation_dict:
            self.validation_dict[i] = []


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(root / "configs/flow"), config_name="config")
def main(config) -> None:
    T.set_float32_matmul_precision('medium')
    T.autograd.set_detect_anomaly(True)
   
    log.info("Instantiating the data")
    data = hydra.utils.instantiate(config.data)

    log.info("Instantiating the callbacks")
    callbacks = instantiate_collection(config.callbacks)
    
    log.info("Instantiating the WandB")
    wandb = hydra.utils.instantiate(config.wandb)

    log.info("Instantiating the Trainer")
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=wandb)

    log.info("Instantiating the models")
    with trainer.init_module():

        model = hydra.utils.instantiate(config.model,
                                        output_dims=data.n_classes,
                                        )

    log.info("Saving config so job can be resumed")
    save_config(config)

    # train model
    log.info("Start training:")

    trainer.fit(model=model, datamodule=data)
    
    if trainer.state.status == "finished":
        log.info("Declaring job as finished!")
        misc.save_declaration("train_finished")

if __name__ == "__main__":
    main()
