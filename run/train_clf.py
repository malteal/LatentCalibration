import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

# import lightning as L # having import issues - should be here
# import numpy as np
import hydra
import torch as T

from tools.tools.omegaconf_utils import instantiate_collection, save_config
import tools.tools.misc as misc

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(root / "configs/classifier"), config_name="config")
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
