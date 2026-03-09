
import os
from omegaconf import OmegaConf,DictConfig
import sys
import ast
import logging
from pathlib import Path
try:
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
except ModuleNotFoundError: 
    rank_zero_only=None
from . import misc

log = logging.getLogger(__name__)

def hydra_init(config_path:str, verbose:bool=False) -> DictConfig:
    """Initialize hydra config with command line arguments
    config_path : str - path to config.yaml file
    
    Also takes in args. If args are in the form of key=value, it will update the config with the key value pair
    
    return : DictConfig - OmegaConf DictConfig object with args from command line
    
    """
    cfg = misc.load_yaml(config_path)
    
    logging.root.setLevel(logging.INFO if verbose else logging.CRITICAL)

    if verbose:
        logging.debug(f"Command line import: {sys.argv}")

    for arg in sys.argv[1:]:
        if verbose:
            print(f"Command line arg: {arg}")
        if '=' in arg:
            key, value = arg.split('=')
            
            # handling lists can be difficult......
            if value.startswith('[') and value.endswith(']'): # converts to list
                try:
                    value = ast.literal_eval(value)
                except ValueError:
                    logging.exception(f"ValueError with {arg}. Converting using split/replace")
                    value = value.replace("[","").replace("]","").split(",")
                    logging.exception(f"Results: {value}")

            elif value.isdigit(): # converts to int
                value = int(value)
            elif value.lower() == 'true': # converts to bool
                value = True
            elif value.lower() == 'false': # converts to bool
                value = False
            elif (value.lower() == 'none'): # converts to None
                value = None
            OmegaConf.update(cfg, key, value, merge=True)
            
    # update with defaults
    if "defaults" in cfg:
        
        # get glob path of config
        glob_path = "/".join(config_path.split("/")[:-1])

        # run over list of defaults
        for i in cfg.defaults:
            if i == '_self_':
                continue
            
            # only works for - defaults: {key: value}
            if isinstance(i, (dict,DictConfig)):
                keys = list(i.keys())
                if len(keys)>1: 
                    raise ValueError(f"Only one key is allowed in the defaults section. Found {len(keys)}")

                path = "/".join(keys)
                file = "/".join(list(i.values()))

                value = misc.load_yaml(f"{glob_path}/{path}/{file}")
                
                OmegaConf.update(cfg, keys[0], value, merge=True)
            # only works for - defaults: value
            elif isinstance(i, (str)):
                value = misc.load_yaml(f"{glob_path}/{i}")
                cfg = OmegaConf.merge(cfg, value)

    return cfg

def get_wandb_id(path:str) -> str:
    """Get the wandb id from the path"""
    wandb_run_id = misc.load_json(f"{path}/wandb/wandb-resume.json")
    return wandb_run_id['run_id']

if rank_zero_only is not None:
    @rank_zero_only
    def reload_original_config(
        path: str = ".",
        file_name: str = "full_config.yaml",
        set_ckpt_path: bool = True,
        ckpt_flag: str = "*",
        set_wandb_resume: bool = True,
    ) -> OmegaConf:
        """Return the original config used to start the job.

        Will also set the chkpt_dir to the latest version of the last or best checkpoint
        """
        log.info(f"Looking for previous job config in {path}")
        try:
            orig_cfg = OmegaConf.load(Path(path, file_name))
        except FileNotFoundError:
            log.warning("No previous job config found! Running with current one.")
            return None

        log.info(f"Looking for checkpoints in folder matching {ckpt_flag}")
        if set_ckpt_path:
            try:
                orig_cfg.ckpt_path = str(
                    max(
                        Path(path).glob(f"checkpoints/{ckpt_flag}"),
                        key=os.path.getmtime,
                    )
                )

                log.info(f"Setting checkpoint path to {orig_cfg.ckpt_path}")

                if set_wandb_resume:
                    wandb_kwargs = {"resume": 'auto', 'id': get_wandb_id(path)}
                    log.info("Attempting to set the same WandB ID to continue logging run")
                    if hasattr(orig_cfg, "wandb"):
                        orig_cfg.wandb.update(wandb_kwargs)
                    elif hasattr(orig_cfg.loggers, "wandb"):
                        orig_cfg.logger.wandb.update(wandb_kwargs)

            except IndexError:
                log.warning("No checkpoint found! Will not set the checkpoint path.")

        return orig_cfg
