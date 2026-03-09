import wandb
import logging
import hydra
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
from typing import List, Any
from glob import glob

from . import misc

log = logging.getLogger(__name__)

# example
# OmegaConf.register_new_resolver("plus_10", lambda x: x + 10)
# c = OmegaConf.create({'key': '${plus_10:990}'})
# c.key
# 1000

OmegaConf.register_new_resolver("eval", eval)

#logit
OmegaConf.register_new_resolver("geq", lambda x,y: x>=y)
OmegaConf.register_new_resolver("leq", lambda x,y: x<=y)
OmegaConf.register_new_resolver("great", lambda x,y: x>y)
OmegaConf.register_new_resolver("less", lambda x,y: x<y)


def check_config(config, resume:bool=False):
    """print config and return it
        Also checks if the output_subdir is None, if so it loads the config from the save_path to resume run
    """
    # Set settings
    settings = wandb.Settings(start_method="fork")
        
    if HydraConfig.get().output_subdir is None:

        if "resume" not in config:
            raise ValueError("output_subdir is None but no resume(True/False) config is given")

        # get values from config
        old_name = config.wandb.name.replace("/", "")

        sync_dir = misc.sort_by_creation_time(glob(f"{config.save_path}/wandb/run*"))[0]

        # Change sync_dir
        settings.update({"sync_dir":sync_dir,
                         "run_id": sync_dir.split("-")[-1],
                         "symlink": False
                         })

        settings.__dict__["timespec"] =  sync_dir.split("-")[-2]

        resume = sync_dir.split("-")[-1]

        # load old config from save_path
        config = misc.load_yaml(f"{config.save_path}/.hydra/config.yaml")
    
        # set values in new config
        config.wandb.update({"name":old_name,
                             "id":resume,
                             })

    # print config
    print(OmegaConf.to_yaml(config))
    
    return config

def instantiate_collection(cfg_coll: DictConfig) -> List[Any]:
    """Use hydra to instantiate a collection of classes and return a list."""
    objs = []

    if not cfg_coll:
        log.warning("List of configs is empty")
        return objs

    if not isinstance(cfg_coll, DictConfig):
        raise TypeError("List of configs must be a DictConfig!")

    for _, cb_conf in cfg_coll.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating <{cb_conf._target_}>")
            objs.append(hydra.utils.instantiate(cb_conf))

    return objs

def save_config(cfg: OmegaConf) -> None:
    """Save the config to the output directory.

    This is necc ontop of hydra's default conf.yaml as it will resolve the entries
    allowing one to resume jobs identically with elements such as ${now:%H-%M-%S}.

    Furthermore, hydra does not allow resuming a previous job from the same dir. The
    work around is reload_original_config but that will fail as hydra overwites the
    default config.yaml file on startup, so this backup is needed for resuming.
    """
    # In order to be able to resume the wandb logger session, save the run id
    if wandb.run is not None and hasattr(cfg, "loggers"):
        if hasattr(cfg.loggers, "wandb"):
            cfg.loggers.wandb.id = wandb.run.id
        else:
            log.warning("WandB is running but cant find config/loggers/wandb!")
            log.warning("This is required to save the ID for resuming jobs.")
            log.warning("Is the name of the logger set correctly")

    # save config tree to file
    OmegaConf.save(cfg, Path(cfg.paths.save_path, "full_config.yaml"), resolve=True)



