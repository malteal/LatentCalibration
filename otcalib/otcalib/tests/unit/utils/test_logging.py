"""
Unit test script for the functions in utils/logging.py
"""

import unittest

from otcalib.otcalib.utils import get_log_level, logger, set_log_level

set_log_level(logger, "DEBUG")


class GetLogLevelTestCase(unittest.TestCase):
    """Test class for the get_log_level function."""

    def test_wrong_key(self):
        """Test wrong key passed to `get_log_level()`."""
        with self.assertRaises(KeyError):
            get_log_level("test")
