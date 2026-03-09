"""Testing the model pipeline"""
import json
import unittest

from otcalib import models
from otcalib.otcalib.data.toy_data import CreateToyData
from otcalib.otcalib.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class TestingModelPipeline(unittest.TestCase):
    """Testing the model pipeline"""

    def setUp(self):  # remember large U
        "Initializing the testing setup"
        logger.debug("Loading model config file")
        with open("otcalib/tests/configs/model_config.json", "r+") as model_dict_config:
            self.model_config = json.load(model_dict_config)

    def testing_model_setup(self):
        "Testing model building"
        logger.debug(f"Initializing models: {self.model_config}")
        f_func = models.PICNN(**self.model_config)
        self.assertTrue(
            all(
                [
                    j.shape[0] == i
                    for i, j in zip(
                        self.model_config["convex_layersizes"][1:], f_func.weight_zz
                    )
                ]
            ),
            "The convex network is missing layers",
        )
        # TODO might need more checks

    def testing_training(self):
        "testing the training pipeline"
        logger.debug(f"Initializing models: {self.model_config}")
        f_func = models.PICNN(**self.model_config)
        g_func = models.PICNN(**self.model_config)

        with open("otcalib/tests/configs/train_config.json", "r+") as train_config:
            training_config = json.load(train_config)

        logger.debug(
            f"Initializing training step with training config: {training_config}"
        )
        model = models.Training(f_func=f_func, g_func=g_func, **training_config)

        # TODO this should run after toy data TEST
        batch_size = 512
        input_simulated = {
            "sim_type": "gauss",
            "sim_para": {
                "mean": [-4, 1],
                "cov": [[1, 3], [3, 2]],
                "size": 5 * batch_size,
            },
        }

        source = CreateToyData(
            **input_simulated, batch_size=batch_size, zero_cond=False
        )

        input_simulated = {
            "sim_type": "gauss",
            "sim_para": {
                "mean": [-4, 5],
                "cov": [[1, 3], [3, 4]],
                "size": 5 * batch_size,
            },
        }
        target = CreateToyData(
            **input_simulated, batch_size=batch_size, zero_cond=False
        )
        info = {
            "loss_f": [],
            "loss_g": [],
            "loss": [],
            "lr_f": [],
            "lr_g": [],
            "train": {},
            "valid": {},
            "o-train": {},
            "o-valid": {},
        }
        info = model.step(source, target, info=info, epoch=0)
        self.assertIsInstance(info, dict, msg="Testing output type")


if __name__ == "__main__":
    with open("otcalib/tests/configs/model_config.json", "r+") as model_dict_config:
        model_config = json.load(model_dict_config)
    f_func = models.PICNN(**model_config)
    g_func = models.PICNN(**model_config)

    with open("otcalib/tests/configs/train_config.json", "r+") as train_config:
        training_config = json.load(train_config)

    model = models.Training(f_func=f_func, g_func=g_func, **training_config)

    # TODO this should run after toy data TEST
    batch_size = 512
    input_simulated = {
        "sim_type": "gauss",
        "sim_para": {
            "mean": [-4, 1],
            "cov": [[1, 3], [3, 2]],
            "size": 5 * batch_size,
        },
    }

    source = CreateToyData(**input_simulated, batch_size=batch_size, zero_cond=False)

    input_simulated = {
        "sim_type": "gauss",
        "sim_para": {
            "mean": [-4, 5],
            "cov": [[1, 3], [3, 4]],
            "size": 5 * batch_size,
        },
    }
    target = CreateToyData(**input_simulated, batch_size=batch_size, zero_cond=False)
    info = {
        "loss_f": [],
        "loss_g": [],
        "loss": [],
        "lr_f": [],
        "lr_g": [],
        "train": {},
        "valid": {},
        "o-train": {},
        "o-valid": {},
    }
    info = model.step(source, target, info=info, epoch=0)
    # self.assertIsInstance(info, dict, msg="Testing output type")
