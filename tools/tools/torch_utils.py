from copy import deepcopy
import numpy as np

import torch as T
from .groupsort import GroupSort
from .modules import GatedLinearUnits

import torch.nn as nn
from copy import deepcopy

def make_ema(model: nn.Module, requires_grad: bool = False) -> nn.Module:
    """
    Creates an Exponential Moving Average (EMA) copy of the given model.

    Args:
        model (nn.Module): The original model to copy.
        requires_grad (bool): If True, the parameters of the EMA model will require gradients.
                              Defaults to False.

    Returns:
        nn.Module: A deep copy of the original model with the specified requires_grad setting.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("The model must be an instance of nn.Module")

    try:
        ema_model = deepcopy(model)
    except Exception as e:
        raise RuntimeError("Failed to deepcopy the model. Ensure all components are deepcopyable.") from e

    if ema_model is not None:
        for param in ema_model.parameters():
            param.requires_grad = requires_grad
    
    return ema_model


@T.no_grad()
def ema(ema_model, model, ema_ratio):
    """
    Apply Exponential Moving Average (EMA) to the parameters of a model.

    Args:
        ema_model (torch.nn.Module): The model to store the EMA parameters.
        model (torch.nn.Module): The original model whose parameters are being averaged.
        ema_ratio (float): The ratio for the EMA update. Typically a value close to 1 (e.g., 0.999).

    Returns:
        torch.nn.Module: The EMA model with updated parameters.
    """
    ema_state_dict = ema_model.state_dict()
    model_state_dict = model.state_dict()
    
    for (param_name, param), (ema_param_name, ema_param) in zip(model_state_dict.items(),
                                                                ema_state_dict.items()):
        ema_state_dict[ema_param_name] = ema_ratio * ema_param + (1 - ema_ratio) * param

    ema_model.load_state_dict(ema_state_dict)
    
    return ema_model

def masked_pooling(kv_seq: T.Tensor, mask: T.Tensor, pooling_style:str='average') -> T.Tensor:
    """
    Perform average pooling on a masked point cloud.
    
    Args:
    - kv_seq (torch.Tensor): Point cloud tensor of shape (b, n, k).
    - mask (torch.Tensor): Mask tensor of shape (b, n) with 1 for valid points and 0 for masked points.
    - pooling_style (str): The pooling style to use. Can be 'average' or 'max'.
    
    Returns:
    - torch.Tensor: Pooled tensor of shape (b, k).
    """
    # Ensure mask has the same dtype as kv_seq
    mask = mask.type_as(kv_seq)
    
    # Expand mask to match the dimensions of kv_seq
    mask_expanded = mask.unsqueeze(-1).expand_as(kv_seq)
    
    # Zero out the masked elements
    masked_kv_seq = kv_seq * mask_expanded
    
    # Sum the masked point cloud along the n dimension
    if 'max' in pooling_style:
        pc_pooled = masked_kv_seq.max(dim=1)[0]
    else:
        sum_pooled = masked_kv_seq.sum(dim=1)
        
        # Count the non-masked elements
        count_non_masked = mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        
        # Compute the average
        pc_pooled = sum_pooled / count_non_masked
    
    return pc_pooled


def activation_functions(activation_str: str, params:dict={}) -> callable:
    """output different activation function

    Parameters
    ----------
    activation_str : str
        name of activation function

    Returns
    -------
    callable
        return the callable activation function

    Raises
    ------
    ValueError
        if the activation function is unknown
    """
    if activation_str.lower() == "softplus":
        act_func = T.nn.Softplus(**params)
    elif activation_str.lower() == "relu":
        act_func = T.nn.ReLU()
    elif activation_str.lower() == "celu":
        act_func =  T.nn.CELU(**params)
    elif activation_str.lower() == "selu":
        act_func =  T.nn.SELU(**params)
    elif activation_str.lower() == "tanh":
        act_func =  T.nn.Tanh(**params)
    elif activation_str.lower().replace("_", "") == "leakyrelu":
        act_func = T.nn.LeakyReLU(0.2, **params)
    elif activation_str.lower() == "elu":
        act_func = T.nn.ELU(**params)
    elif activation_str.lower() == "gelu":
        act_func = T.nn.GELU(**params)
    elif activation_str.lower() == "silu":
        act_func = T.nn.SiLU(**params)
    elif activation_str.lower() == "gs":
        act_func = GroupSort()
    elif activation_str.lower() == "glu":
        act_func = GatedLinearUnits(**params)
    else:
        raise ValueError(f"Did not recognize the activation_str: {activation_str}")
    return act_func

def save_torch_model(epoch, model, optimizer, loss, path, **kwargs):
    save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }
    if kwargs is not None:
        save_dict.update(kwargs)
    T.save(save_dict, path)
    
def onnx_export_model(model: T.nn.Module, inputs: dict, path: str, input_names:list, output_names:list, dynamic_axes:dict=None, verbose=False):
    "export model to onnx format"
    T.onnx.export(model=model,
                      args=inputs,
                      f=path,
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True,
                      opset_version=14,
                      verbose=verbose,
                      dynamic_axes=dynamic_axes
                      )

def count_trainable_parameters(model):
    sum_trainable = np.sum([i.numel() for i in model.parameters() if i.requires_grad])
    return sum_trainable

def append_dims(x: T.Tensor, target_dims: int, add_to_front: bool = False) -> T.Tensor:
    """Appends dimensions of size 1 to the end or front of a tensor until it
    has target_dims dimensions.

    Parameters
    ----------
    x : T.Tensor
        The input tensor to be reshaped.
    target_dims : int
        The target number of dimensions for the output tensor.
    add_to_front : bool, optional
        If True, dimensions are added to the front of the tensor.
        If False, dimensions are added to the end of the tensor.
        Defaults to False.

    Returns
    -------
    T.Tensor
        The reshaped tensor with target_dims dimensions.

    Raises
    ------
    ValueError
        If the input tensor already has more dimensions than target_dims.

    Examples
    --------
    >>> x = T.tensor([1, 2, 3])
    >>> x.shape
    torch.Size([3])

    >>> append_dims(x, 3)
    tensor([[[1]], [[2]], [[3]]])
    >>> append_dims(x, 3).shape
    torch.Size([3, 1, 1])

    >>> append_dims(x, 3, add_to_front=True)
    tensor([[[[1, 2, 3]]]])
    >>> append_dims(x, 3, add_to_front=True).shape
    torch.Size([1, 1, 3])
    """
    dim_diff = target_dims - x.dim()
    if dim_diff < 0:
        raise ValueError(f"x has more dims ({x.ndim}) than target ({target_dims})")
    if add_to_front:
        return x[(None,) * dim_diff + (...,)]  # x.view(*dim_diff * (1,), *x.shape)
    return x[(...,) + (None,) * dim_diff]  # x.view(*x.shape, *dim_diff * (1,))



# if __name__ == "__main__":
    