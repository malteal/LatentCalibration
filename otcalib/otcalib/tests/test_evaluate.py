"""Testing the evaluation metric"""
import unittest

import numpy as np

from otcalib.otcalib.data.toy_data import CreateToyData
from otcalib.otcalib.utils import Metrics, logger, set_log_level

set_log_level(logger, "DEBUG")
STR_CONVERGENCE = "Testing convergence of "
STR_TYPE = "Testing output type"
STR_METRIC_DICT = "Testing if elements were correctly added"


class TestingMetric(unittest.TestCase):
    """Testing the metric system to OT"""

    def setUp(self):
        "Initializing the testing setup"
        logger.debug("Loading model config file")
        self.all_metrics = Metrics(
            kl_input={"reduction": "batchmean"}, geomloss_input=[]
        )
        self.batch_size = 2048
        input_simulated = {
            "sim_type": "gauss",
            "sim_para": {"mean": [5], "cov": [[3]], "size": 5 * self.batch_size},
        }
        self.target = CreateToyData(
            **input_simulated, batch_size=self.batch_size, zero_cond=False
        ).valid_dist()

        input_simulated = {
            "sim_type": "gauss",
            "sim_para": {
                "mean": [5, 1],
                "cov": [[1, 3], [4, 3]],
                "size": 5 * self.batch_size,
            },
        }
        self.target2 = CreateToyData(
            **input_simulated, batch_size=self.batch_size, zero_cond=False
        ).valid_dist()
        self.ks_closeness = [0, 0, 0.05]

    def testing_individual_metric_1d(self):
        """Testing the 1d metric"""
        wass_last = 10
        kl_last = 10
        for mean, clo in zip(np.linspace(0, 5, 3), self.ks_closeness):
            input_simulated = {
                "sim_type": "gauss",
                "sim_para": {"mean": [mean], "cov": [[3]], "size": 5 * self.batch_size},
            }

            source = CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            ).valid_dist()

            # ks test
            ks_test = self.all_metrics.ks_test1d(source, self.target)
            self.assertGreaterEqual(
                np.min(ks_test), clo, msg=STR_CONVERGENCE + "1d ks test"
            )
            self.assertIsInstance(ks_test, list, msg=STR_TYPE)

            # wasserstein
            wass = self.all_metrics.nd_wasserstein_distance(source, self.target)
            self.assertLess(wass, wass_last, msg=STR_CONVERGENCE + "1d wasserstein")
            self.assertIsInstance(wass, np.float64, msg=STR_TYPE)
            wass_last = wass

            # KL-divergence
            kl_div = self.all_metrics.kl_div(source, self.target)
            self.assertLess(kl_div, kl_last, msg=STR_CONVERGENCE + "1d kl div")
            self.assertIsInstance(kl_div, np.float64, msg=STR_TYPE)
            kl_last = kl_div

    def testing_individual_metric_2d(self):
        """Testing 2d Metrics"""
        wass_last = 10
        kl_last = 10
        for mean, clo in zip(np.linspace(0, 5, 3), self.ks_closeness):
            input_simulated = {
                "sim_type": "gauss",
                "sim_para": {
                    "mean": [mean, 1],
                    "cov": [[1, 3], [4, 3]],
                    "size": 5 * self.batch_size,
                },
            }

            source = CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            ).valid_dist()

            # ks test
            ks_test = self.all_metrics.ks_test1d(source, self.target2)
            self.assertGreaterEqual(
                np.min(ks_test),
                clo,
                msg=STR_CONVERGENCE
                + (
                    "2d ks test - if this failes just"
                    + "run again (KS testing can have large variance)"
                ),
            )  # this might sometimes fire due to kstest - just run it again
            self.assertIsInstance(ks_test, list, msg=STR_TYPE)

            # wasserstein
            wass = self.all_metrics.nd_wasserstein_distance(source, self.target2)
            # print(wass)
            self.assertLess(wass, wass_last, msg=STR_CONVERGENCE + "2d wasserstein")
            self.assertIsInstance(wass, np.float64, msg=STR_TYPE)
            wass_last = wass

            # KL-divergence
            kl_div = self.all_metrics.kl_div(source, self.target2)
            self.assertLess(kl_div, kl_last, msg=STR_CONVERGENCE + "2d kl div")
            # print(kl_div)
            self.assertIsInstance(kl_div, np.float64, msg=STR_TYPE)
            kl_last = kl_div

    def testing_combined_metrics_1d(self):
        """Testing the combination of metric in 1d"""
        info = {"test": {}}
        for length_of_out, mean in enumerate(np.linspace(0, 5, 3)):
            input_simulated = {
                "sim_type": "gauss",
                "sim_para": {"mean": [mean], "cov": [[3]], "size": 5 * self.batch_size},
            }

            source = CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            ).valid_dist()

            info = self.all_metrics.run(
                source, self.target, append=info, sub_dict="test"
            )
            self.assertEqual(len(info["test"]["wasserstein"]), length_of_out + 1)
            self.assertEqual(len(info["test"]["kl_div"]), length_of_out + 1)

            info_single = self.all_metrics.run(source, self.target)
            self.assertEqual(
                info_single["wasserstein"], info["test"]["wasserstein"][length_of_out]
            )
            self.assertEqual(
                info_single["kl_div"], info["test"]["kl_div"][length_of_out]
            )

    def testing_combined_metrics_2d(self):
        """Testing the combination of metric in 2d"""
        info = {"test": {}}
        for length_of_out, mean in enumerate(np.linspace(0, 5, 3)):
            input_simulated = {
                "sim_type": "gauss",
                "sim_para": {
                    "mean": [mean, 1],
                    "cov": [[1, 3], [4, 3]],
                    "size": 5 * self.batch_size,
                },
            }

            source = CreateToyData(
                **input_simulated, batch_size=self.batch_size, zero_cond=False
            ).valid_dist()

            info = self.all_metrics.run(
                source, self.target2, append=info, sub_dict="test"
            )
            logger.debug(info)
            self.assertEqual(
                len(info["test"]["wasserstein"]), length_of_out + 1, msg=STR_METRIC_DICT
            )
            self.assertEqual(
                len(info["test"]["kl_div"]), length_of_out + 1, msg=STR_METRIC_DICT
            )

            info_single = self.all_metrics.run(source, self.target2)
            self.assertEqual(
                info_single["wasserstein"],
                info["test"]["wasserstein"][length_of_out],
                msg=STR_METRIC_DICT,
            )
            self.assertEqual(
                info_single["kl_div"],
                info["test"]["kl_div"][length_of_out],
                msg=STR_METRIC_DICT,
            )
