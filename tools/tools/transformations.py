"transformations"
import numpy as np
from typing import Union
import torch as T

def log_squash(data: np.ndarray) -> np.ndarray:
    """Apply a log squashing function for distributions with high tails."""
    return np.sign(data) * np.log(np.abs(data) + 1)


def undo_log_squash(data: np.ndarray) -> np.ndarray:
    """Undo the log squash function above."""
    return np.sign(data) * (np.exp(np.abs(data)) - 1)

def logit(data: Union[np.ndarray, T.Tensor]) -> Union[np.ndarray, T.Tensor]:
    """Apply a logit transformation to the data."""
    if isinstance(data, T.Tensor):
        return T.log(data/(1-data))
    else:
        return np.log(data/(1-data))

def logistic(data: Union[np.ndarray, T.Tensor]) -> Union[np.ndarray, T.Tensor]:
    """Apply a logistic transformation to the data."""
    if isinstance(data, T.Tensor):
        return 1/(1+T.exp(-data))
    else:
        return 1/(1+np.exp(-data))

def logit_normal(normal_dst: Union[np.ndarray, T.Tensor]) -> Union[np.ndarray, T.Tensor]:
    """Logit normal distribution."""
    return logistic(normal_dst)

def softmax(data: np.ndarray) -> np.ndarray:
    """Apply softmax transformation to the data."""
    exp_data = np.exp(data - np.max(data))
    return exp_data / np.sum(exp_data, axis=-1, keepdims=True)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize the data."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x = np.random.normal(0,1, 100_000)
    xuni = np.random.normal(0,1, 100_000)
    s = logit_normal(xuni)
    plt.hist(logistic(x), bins=50, alpha=0.5)
    plt.hist(s, bins=50, alpha=0.5)

