"""This script will test the setup of toy data pipeline"""
import unittest

from otcalib.otcalib.data.toy_data import CreateToyData
from otcalib.otcalib.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class TestingToyData(unittest.TestCase):
    """Testing the toy data setup"""

    def test_toy_data(self):
        """Testing the toy data setup with a 2d gaussian"""
        batch_size = 512
        input_simulated = {
            "sim_type": "gauss",
            "sim_para": {
                "mean": [-4, 1],
                "cov": [[1, 3], [3, 2]],
                "size": 100 * batch_size,
            },
        }

        simulation = CreateToyData(
            **input_simulated, batch_size=batch_size, zero_cond=False
        )

        output = next(iter(simulation))
        self.assertIsInstance(
            output, tuple, "This should be a tuple, containing (conditions, source)"
        )
        self.assertEqual(
            len(output), 2, "This tuple should contain (conditions, source)"
        )
        self.assertEqual(
            output[0].shape[0],
            batch_size,
            f"The source should have the same batch size of {batch_size}",
        )
