"""Configuration for logger of otcalib and tf as well as reading global config."""
import logging
import os
import pathlib

from otcalib.utils.yaml_tools import YAML

# import matplotlib


def get_log_level(level: str):
    """Get logging levels with string key.

    Parameters
    ----------
    level : str
        Log level as string.

    Returns
    -------
    logging level
        logging object with log level info

    Raises
    ------
    KeyError
        if non-valid option is given
    """
    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    if level not in log_levels:
        raise KeyError(f"The 'DebugLevel' option {level} is not valid.")
    return log_levels[level]


class Configuration:
    """
    This is a global configuration to allow certain settings which are
    hardcoded so far.
    """

    def __init__(self):
        super().__init__()
        self.yaml_config = (
            f"{pathlib.Path(__file__).parent.absolute()}/../configs/global_config.yaml"
        )
        self.load_config_file()
        self.logger = self.set_logging_level()
        self.set_tf_debug_level()
        self.set_mlp_plotting_backend()
        self.get_configuration()

    def load_config_file(self):
        """Load config file from disk."""
        yaml = YAML(typ="safe", pure=True)
        with open(self.yaml_config, "r") as conf:
            self.config = yaml.load(conf)
        print(self.config)

    def get_configuration(self):
        """Assign configuration from file to class variables.

        Raises
        ------
        KeyError
            if required config is not present in passed config file
        """
        config_items = [
            "DebugLevel",
            "TFDebugLevel",
            "MPLPlottingBackend",
        ]
        for item in config_items:
            if item in self.config:
                self.logger.debug(f"Setting {item} to {self.config[item]}.")
                setattr(self, item, self.config[item])
            else:
                raise KeyError(f"You need to specify {item} in your config file!")

    def set_mlp_plotting_backend(self):
        """Setting the plotting backend of matplotlib."""
        self.logger.debug(
            f"Setting Matplotlib's backend to {self.config['MPLPlottingBackend']}"
        )

        # matplotlib.use(self.config["MPLPlottingBackend"])

    def set_tf_debug_level(self):
        """Setting the Debug level of tensorflow.
        For reference see https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
        """  # noqa # pylint: disable=C0301
        self.logger.debug(f"Setting TFDebugLevel to {self.config['TFDebugLevel']}")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(self.config["TFDebugLevel"])

    def set_logging_level(self):
        """Set DebugLevel for logging.

        Returns
        -------
        logger
            logger object with new level set
        """
        otcalib_logger = logging.getLogger("otcalib")
        otcalib_logger.setLevel(get_log_level(self.config["DebugLevel"]))
        ch_handler = logging.StreamHandler()
        ch_handler.setLevel(get_log_level(self.config["DebugLevel"]))
        ch_handler.setFormatter(CustomFormatter())

        otcalib_logger.addHandler(ch_handler)
        otcalib_logger.propagate = False
        return otcalib_logger


def set_log_level(otcalib_logger, log_level: str):
    """Setting log level

    Parameters
    ----------
    otcalib_logger : logger
        logger object
    log_level : str
        logging level corresponding CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    """

    otcalib_logger.setLevel(get_log_level(log_level))
    for handler in otcalib_logger.handlers:
        handler.setLevel(get_log_level(log_level))


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors
    using implementation from
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output"""  # noqa # pylint: disable=C0301

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    green = "\x1b[32;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    debugformat = (
        "%(asctime)s - %(levelname)s:%(name)s: %(message)s (%(filename)s:%(lineno)d)"
    )
    date_format = "%(levelname)s:%(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + debugformat + reset,
        logging.INFO: green + date_format + reset,
        logging.WARNING: yellow + date_format + reset,
        logging.ERROR: red + debugformat + reset,
        logging.CRITICAL: bold_red + debugformat + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


global_config = Configuration()
logger = global_config.logger
logger.debug(f"Loading global config {global_config.yaml_config}")
