"Training the calibration algorithm"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
from tqdm import tqdm
import torch
import hydra
import copy
from glob import glob
from tools.tools import misc
from omegaconf import OmegaConf

# ot-framework
from otcalib.otcalib.torch.torch_utils import load_model_w_hydra

def hydra_resume(ckpt_path: str):
    from hydra.core.hydra_config import HydraConfig
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
   
    checkpoint_cfg = OmegaConf.load(ckpt_path + '/.hydra/config.yaml')
    ConfigStore.instance().store('checkpoint_cfg', node=checkpoint_cfg)
    cfg = hydra.compose(config_name="checkpoint_cfg", overrides=HydraConfig.get().overrides.task)
    
    return cfg

@hydra.main(config_path=str(root / "configs/ot/"), config_name="config", version_base=None)
def main(config) -> None:
    logging.info("Running OTcalib framework")
    logging.info("torch version: %s", torch.__version__)
    logging.info("Outdir: %s", config.path.save_path)

    torch.set_default_dtype(torch.float32)

    ckpt_cfg = copy.deepcopy(config['ckpt'], )
    if ckpt_cfg.get('path') and config.ckpt.full_resume:
        ckpt_path = ckpt_cfg.get('path')

        logging.info("Resume training from checkpoint: %s", ckpt_path)

        new_config = config.ckpt.get('new_config', {})
        config = hydra_resume(ckpt_path)
        config = OmegaConf.merge(config, new_config)

    logging.info("Initializing data!")
    # load data
    datamodule = hydra.utils.instantiate(config.data)
    
    source, target, eval_data, conds_bins = datamodule()
    
    if datamodule.bootstraps_iters>0:
        misc.save_json(
            datamodule.bootstrap_w_replacements.tolist(),
            f"{config.save_path}/bootstrap_w_replacements.json")
    
    if ckpt_cfg.get('path'):
        f_func, g_func = load_model_w_hydra(ckpt_path, index_to_load=-1, 
                                            device=config.device)
        
        if not ckpt_cfg.full_resume:
            # load old config
            hydra_config = misc.load_yaml(f"{ckpt_path}/.hydra/config.yaml")
            
            # save config to new path
            misc.save_yaml(hydra_config, f"{config.path.save_path}/.hydra/config.yaml")
    else:
        # create model
        f_func = hydra.utils.instantiate(copy.deepcopy(config.model)
                                        )(cvx_dim=datamodule.cvx_dim,
                                        noncvx_dim=datamodule.noncvx_dim)

        g_func = hydra.utils.instantiate(copy.deepcopy(config.model)
                                        )(cvx_dim=datamodule.cvx_dim,
                                        noncvx_dim=datamodule.noncvx_dim)
                                        
    # init training loop
    training_loop = hydra.utils.instantiate(config.train,
                                        f_func=f_func,
                                        g_func=g_func,
                                        outdir=config.path.save_path,
                                        conds_bins=conds_bins,
                                        ckpt_path=ckpt_path if ckpt_cfg.full_resume else None,
                                        )

    # define progress bar
    pbar = tqdm(range(training_loop.epoch_nr if ckpt_cfg.full_resume else 0,
                      config.train.nepochs))
    
    for epoch in pbar:
        
        #evaluate performance
        training_loop.evaluate(eval_data, epoch, conds_range=conds_bins,
                            source_iter=source, target_iter=target,
                            transnames=datamodule.transportnames,
                            condsnames=datamodule.condnames,
                            **config.eval)
        
        #train networks
        pbar = training_loop.train(source, target, pbar=pbar)
        
        # for sampling new data events for training
        source, target = datamodule.create_tp_samplers(datamodule.target_sample, shuffle=True)
        
if __name__ == "__main__":
    main()
