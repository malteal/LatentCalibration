"Testing the plotting pipeline"
import unittest

import matplotlib.pyplot as plt
import numpy as np

from otcalib.otcalib.data.toy_data import CreateToyData
from otcalib.otcalib.utils import logger, set_log_level, uplot

set_log_level(logger, "DEBUG")
STR_CONVERGENCE = "Testing convergence of "
STR_TYPE = "Testing output type"
STR_METRIC_DICT = "Testing if elements were correctly added"


class TestingPlotting(unittest.TestCase):
    "Testing the plotting pipeline"

    def setUp(self):
        "Initializing the testing setup"
        logger.debug("Loading init for plotting")
        self.batch_size = 2048
        input_simulated = {
            "sim_type": "gauss",
            "sim_para": {"mean": [5], "cov": [[3]], "size": 5 * self.batch_size},
        }
        self.source = (
            CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            )
            .valid_dist()
            .detach()
            .numpy()
        )
        self.target = (
            CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            )
            .valid_dist()
            .detach()
            .numpy()
        )

        input_simulated = {
            "sim_type": "gauss",
            "sim_para": {
                "mean": [5, 1],
                "cov": [[1, 3], [4, 3]],
                "size": 5 * self.batch_size,
            },
        }
        self.source2 = (
            CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            )
            .valid_dist()
            .detach()
            .numpy()
        )
        self.target2 = (
            CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            )
            .valid_dist()
            .detach()
            .numpy()
        )

        input_simulated = {
            "sim_type": "gauss",
            "sim_para": {
                "mean": [5, 1, 2],
                "cov": [[1, 3, 2], [4, 3, 1], [1, 4, 1]],
                "size": 5 * self.batch_size,
            },
        }
        self.source3 = (
            CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            )
            .valid_dist()
            .detach()
            .numpy()
        )
        self.target3 = (
            CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            )
            .valid_dist()
            .detach()
            .numpy()
        )

    def testing_1d_plots(self):
        "Testing the output of the 1d plots functions"
        fig, fig_cern, mae_trans, mae_source = uplot.plot_1d_hist(
            self.source, self.target, self.source, "Dummy name"
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(fig_cern, plt.Figure)
        self.assertIsInstance(mae_trans, np.ndarray)
        self.assertIsInstance(mae_source, np.ndarray)
