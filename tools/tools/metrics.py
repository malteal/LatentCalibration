"Calculate various metrics"

import numpy as np

def log_squash(data: np.ndarray) -> np.ndarray:
    """Apply a log squashing function for distributions with high tails."""
    return np.sign(data) * np.log(np.abs(data) + 1)


def undo_log_squash(data: np.ndarray) -> np.ndarray:
    """Undo the log squash function above."""
    return np.sign(data) * (np.exp(np.abs(data)) - 1)

def IQR(values):
    "Calculate the interquartile range"
    percentiles = np.percentile(values, [75,25], 0)
    return (percentiles[0]-percentiles[1])/1.349 # 1.349 is the IQR of a normal distribution