# flake8: noqa
# pylint: skip-file
__version__ = "0.0"

from otcalib import *
try:
    from otcalib.torch import *
    from otcalib.utils import *
except (AttributeError, ModuleNotFoundError):
    from otcalib.otcalib.utils import *
    from otcalib.otcalib.torch import torch_utils, loader, layers, PICNN
    from otcalib.otcalib.evaluate import *
