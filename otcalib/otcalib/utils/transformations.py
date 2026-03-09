"useful transformations"

import numpy as np
import pandas as pd
import torch as T
from typing import Union


def numpy_check(data: T.Tensor) -> np.ndarray:
    """should take a tensor a turn it to numpy

    Parameters
    ----------
    data : T.Tensor
        tensor

    Returns
    -------
    np.ndarray
        a numpy matrix

    Raises
    ------
    TypeError
        is the type is unknown
    """
    if isinstance(data, T.Tensor):
        if data.is_cuda:
            data = data.cpu()
        data = data.detach().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, list):
        data = np.array(data)
    else:
        raise TypeError("Data type unknown in numpy_check")
    return data


def logit(xs_input:Union[np.ndarray, T.Tensor]) -> Union[np.ndarray, T.Tensor]:
    """logit transformation torch - should be able to take any type a output the logit

    Parameters
    ----------
    xs_input : Tensor or array
        vector of value

    Returns
    -------
    np.ndarray
        logit transformed vector

    Raises
    ------
    TypeError
        if the type is unknown
    """
    if isinstance(xs_input, T.Tensor):
        logit_trans = T.log(xs_input / (1 - xs_input + 1e-10))  # pylint: disable=E1101
    elif isinstance(xs_input, (np.ndarray, pd.DataFrame, pd.Series)):
        logit_trans = np.log(xs_input / (1 - xs_input + 1e-10))
    else:
        raise TypeError("Data type unknow in logit")
    return logit_trans

def probsfromlogits(logitps: Union[np.ndarray, T.Tensor]) -> Union[np.ndarray, T.Tensor]:
    """reverse transformation from logits to probs using logistic

    Parameters
    ----------
    logitps : np.ndarray
        arrray of logit

    Returns
    -------
    np.ndarray
        probabilities from logit
    """
      
    norm = 1
    if isinstance(logitps, T.Tensor):
        ps_value = 1.0 / (1.0 + T.exp(-logitps))
        # ps_value = T.exp(ps_value)
        # ps_value = ps_value/ps_value.sum(1).view(-1,1)
        if (ps_value.shape[-1] > 1) and (len(ps_value.shape) > 1):
            norm = T.sum(ps_value, axis=1)
            norm = T.stack([norm] * logitps.shape[1]).T
    else:
        logitps = numpy_check(logitps)
        ps_value = 1.0 / (1.0 + np.exp(-logitps))
        # ps_value = np.exp(ps_value)
        # ps_value = ps_value/np.sum(ps_value, 1)
        if (ps_value.shape[-1] > 1) and (len(ps_value.shape) > 1):
            norm = np.sum(ps_value, axis=1)
            norm = np.stack([norm] * logitps.shape[1]).T
    return ps_value / norm


# def softmax(logps: np.ndarray) -> np.ndarray:
#     """Softmax transformation

#     Parameters
#     ----------
#     logps : np.ndarray
#         log values that should be turned to proba

#     Returns
#     -------
#     np.ndarray
#         proba vales
#     """
#     ps_value = np.exp(logps)
#     denom = np.sum(ps_value, axis=1).reshape(-1, 1)
#     return ps_value / denom


def dl1r(ps_value: np.ndarray, dl1r_c=False, fc=None) -> np.ndarray:
    """dl1r transformation

    Parameters
    ----------
    ps_value : np.ndarray
        probabilities

    Returns
    -------
    np.ndarray
        return the dl1r values
    """
    if fc is None:
        if dl1r_c:
            fc=0.3 # from VHCC
        else:
            fc=0.018

    if isinstance(ps_value, T.Tensor):
        pb_value = ps_value[:, 0]
        pc_value = ps_value[:, 1]
        pu_value = ps_value[:, 2]
        if dl1r_c:
            output = T.log(pc_value) - T.log(fc * pb_value + (1 - fc) * pu_value)
        else:
            output = T.log(pb_value) - T.log(fc * pc_value + (1 - fc) * pu_value)
        output = output.view(-1, 1)
    elif len(ps_value.shape) == 1:
        ps_value[ps_value < 1e-5] = 1e-5
        pb_value = ps_value[0]
        pc_value = ps_value[1]
        pu_value = ps_value[2]
        if dl1r_c:
            output = np.array(
                [np.log(pc_value) - np.log(fc * pb_value + (1 - fc) * pu_value)]
            )
        else:
            output = np.array(
                [np.log(pb_value) - np.log(fc * pc_value + (1 - fc) * pu_value)]
            )
        output.shape = (1,)
    else:
        pb_value = ps_value[:, 0]
        pc_value = ps_value[:, 1]
        pu_value = ps_value[:, 2]
        if dl1r_c:
            output = np.log(pc_value) - np.log(fc * pb_value + (1 - fc) * pu_value)
        else:
            output = np.log(pb_value) - np.log(fc * pc_value + (1 - fc) * pu_value)
    return output
