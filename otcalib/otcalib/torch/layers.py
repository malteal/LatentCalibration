"""
Copyright (c)2021 Chris Pollard and Philipp Windischhofer
an implementation of the partially convex neural networks
introduced in https://arxiv.org/abs/1609.07152
"""
import numpy as np
import torch as T
import torch.nn as nn
from typing import Union
from torch.autograd import Function


def apply_linear_layer(
    input_value: T.Tensor, weights: T.Tensor, bias: T.Tensor=None) -> T.Tensor:  # torch.nn.functional.softplus
    """apply linear mapping

    Parameters
    ----------
    input_value : torch.Tensor
         x input for the network
    weights : torch.Tensor
        weight input of the neurons
    biases : int, optional
        weight input of the neurons, by default 0

    Returns
    -------
    torch.Tensor
         torch.Tensor: Output the linear mapping
    """
    # try:
    if bias is not None:
        return input_value.matmul(weights.t()) + bias
    return input_value.matmul(weights.t()) 

def concat_and_sum(*args):
    """
    concatenate and sum the tensors
    
    This is required for ONNX compatibility
    """
    return T.stack(args, dim=0).sum(dim=0)

class Linear(nn.Linear):
    def __init__(self,in_features, out_features, **kwargs):
        super().__init__(in_features=in_features, out_features=out_features, **kwargs)
    
    def forward(self, input: T.Tensor) -> T.Tensor:
        return apply_linear_layer(input, self.weight, self.bias)


def get_act_funcs(activation_str: str, params: dict = {}, device="cuda") -> callable:
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
    # return Softplus(zeroed=True, device=device)

    activation_str = activation_str.lower().replace("_", "")
    if activation_str == "softplus":
        act_func = Softplus(device=device)
    elif activation_str == "softpluszeroed":
        act_func = Softplus(zeroed=True, device=device)
    elif activation_str == "relu":
        act_func = nn.ReLU()
    elif activation_str == "celu":
        act_func = CELU()
    elif activation_str == "tanh":
        act_func = nn.Tanh()
    elif activation_str == "leakyrelu":
        act_func = nn.LeakyReLU(0.2)
    elif activation_str == "symsoftplus":
        act_func = Softplus(symmetric=True, device=device)
    elif activation_str == "elu":
        act_func = nn.ELU()
    elif activation_str == "sigmoid":
        act_func = nn.Sigmoid()
    elif activation_str == "":
        act_func = lambda x: x  # dummy functions
    else:
        raise ValueError(f"Did not recognize the activation_str: {activation_str}")
    return act_func


class Softplus(nn.Module):
    def __init__(
        self,
        beta=1,
        threshold=20,
        zeroed=False,
        symmetric=False,
        linear=False,
        trainable_beta=False,
        onnx_export: bool=False,
        device="cuda",
    ):
        super(Softplus, self).__init__()

        "onnx compatible"
        self.threshold = threshold
        # self.zero = T.tensor([0.0], device=device)
        self.zeroed = zeroed
        self.symmetric = symmetric
        self.device = device
        self.linear = linear
                
        # the torch_softplus is not completely stable when training and taking the gradient
        # softplus is not implemented in onnx so when export is true we use the torch_softplus
        self.func_softplus = self.torch_softplus if onnx_export else nn.Softplus()


    def torch_softplus(self, x):
        # return T.nn.functional.softplus(x)
        # return T.log(1+T.exp(x))
        y = x.clamp(max=self.threshold)
        return T.where(
            x < self.threshold,
            # T.log(1 / (0.1 * self.beta.exp()) + T.exp((0.1 * self.beta.exp()) * y)),
            T.log(1 + T.exp(y)),
            x,
        )

    def forward(self, x):
        x_sp = self.func_softplus(x)

        if self.zeroed:
            return x_sp - 0.69314718056
        elif self.symmetric:
            return x_sp - 0.5 * x
        elif self.linear:
            gain = 1 / x.size(1)
            return x_sp * x * gain
        else:
            return x_sp

    def __repr__(self):
        if self.zeroed:
            return "Zeroed Softplus"
        elif self.symmetric:
            return "Symmetric Softplus"
        elif self.linear:
            return "Linear Softplus"
        else:
            return "Softplus"

class CELU(nn.Module):
    def __init__(self, device="cuda", onnx_export: bool=False):
        super(CELU, self).__init__()
        self.device = device

        self.onnx_export = onnx_export 
        
        # the torch_celu is not completely stable when training and taking the gradient
        # celu is not implemented in onnx so when export is true we use the torch_celu
        self.func_celu = self.torch_celu if onnx_export else nn.CELU()
    
    def torch_celu(self, x):
        return T.where(
            x > 0,
            x,
            (T.exp(x) - 1),
        )

    def forward(self, x):
        return self.func_celu(x)


class ELU(nn.Module):
    def __init__(self, alpha=1.0, device="cuda"):
        super(CELU, self).__init__()
        self.device = device
        self.alpha = alpha

    def forward(self, x):
        return T.where(x < 0, self.alpha * (T.exp(x) - 1), x)


def weight(
    in_size: int, out_size: int, device: str = "cpu", requires_grad=True
) -> T.Tensor:
    """Initialize the weight parameter of neurons randomly

    Parameters
    ----------
    in_size : int
        Number of colums
    out_size : int
        Number of rows
    device : str, optional
        Which device to init on, by default "cpu"

    Returns
    -------
    T.Tensor
        output a random weight tensor with size y times in_size.
    """
    weights = T.empty((out_size, in_size))
    if in_size > 0 and out_size > 0:
        nn.init.kaiming_normal_(weights)
    # if "softplus" in act_func:
    weights = weights / weights.size(0)  # divide with input size
    parameters = nn.parameter.Parameter(weights.to(device), requires_grad=requires_grad)
    return parameters


class PositiveLayer(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        act_func: str,
        deactivate: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.deactivate = deactivate
        self.in_size = in_size
        self.device = device
        self.out_size = out_size
        self.act_func = act_func

        self.linear_layer = weight(self.in_size, self.out_size, device=self.device)
        self.act_enforce_cvx = get_act_funcs(
            self.act_func, device=self.device
            )

        if self.deactivate:
            self.deactivate_layer()

    def deactivate_layer(self):
        # the first cvx layer in PICNN/ICNN should be zeroed
        self.linear_layer.data.copy_(
            T.zeros_like(self.linear_layer)  # pylint: disable=E1101
        )
        self.linear_layer.requires_grad = False

    def forward(self, x):

        # run matrix multi
        linear_proj = x.matmul(self.act_enforce_cvx(self.linear_layer.t()))

        if "softplus" in self.act_func:
            gain = 1 / x.size(1)
            return linear_proj * gain
        else:
            return linear_proj


class IterativeNormLayer(nn.Module):
    # from matt
    def __init__(self, in_shape: int, max_iters:int=100_000, ema_sync:float=0.99,
                 extra_dims:Union[tuple, int] = (),
                 track_grad_forward:bool=True, track_grad_reverse:bool=False
                 ):
        super().__init__()
        if not isinstance(in_shape, list):
            self.inpt_dim = in_shape
        self.max_iters=max_iters
        self.ema_sync=ema_sync
        self.extra_dims=extra_dims
        self.do_ema=True
        # init: n is the number of samples seen so far
        n = T.tensor(0)

        # Init buffers are needed for saving/loading the layer
        self.register_buffer(
            "means",
            T.zeros(self.inpt_dim, dtype=T.float32)
            )
        self.register_buffer(
            "vars",
            T.ones(self.inpt_dim, dtype=T.float32)
            )
        self.register_buffer("n", n)

        # If the means are set here then the model is "frozen" and never updated
        self.register_buffer(
            "frozen",
            T.as_tensor( self.n > self.max_iters),
            )

        # Gradient tracking options
        self.track_grad_forward = track_grad_forward
        self.track_grad_reverse = track_grad_reverse


    def __repr__(self):
        return f"IterativeNormLayer({list(self.means.shape)})"

    def __str__(self) -> str:
        return f"IterativeNormLayer(m={self.means.squeeze()}, v={self.vars.squeeze()})"

    def _mask(self, inpt: T.Tensor, mask: Union[T.BoolTensor, None] = None) -> T.Tensor:
        if mask is None:
            return inpt
        return inpt[mask]

    def _unmask(
        self, inpt: T.Tensor, output: T.Tensor, mask: Union[T.BoolTensor, None] = None
    ) -> T.Tensor:
        if mask is None:
            return output
        masked_out = inpt.clone()  # prevents inplace operation, bad for autograd
        masked_out[mask] = output.type(masked_out.dtype)
        return masked_out

    def _check_attributes(self) -> None:
        if self.means is None or self.vars is None:
            raise ValueError(
                "Stats have not been initialised and fit() has not been run!"
            )

    def fit(
        self, inpt: T.Tensor, mask: Union[T.BoolTensor, None] = None, freeze: bool = True
    ) -> None:
        """Set the stats given a population of data."""
        inpt = self._mask(inpt, mask)
        _vars, _means = T.var_mean(
            inpt, dim=(0, *self.extra_dims), keepdim=True
        )
        if (self.means.shape != _means.shape) or (self.vars.shape != _vars.shape):
            print("means shape:", self.means.shape, _means.shape)
            raise ValueError("""New means/vars shapes does not match the shape of the layer!
                             batch dimension might be missing or extra dimensions are present!""")
        else:
            self.means= _means
            self.vars = _vars
        
        self.n += T.tensor(self._n, device=self.means.device)
        # self.m2 = self.vars * self.n
        if freeze:
            self.frozen.fill_(True)
    
    def check_device(self, inpt:T.Tensor):
        "push tensor to device if not same as means"
        if self.means.device!=inpt:
            return inpt.to(self.means.device)
        else:
            return inpt

    def forward(self, inpt: T.Tensor, mask: Union[T.BoolTensor, None] = None,
                training:bool=True) -> T.Tensor:
        """Apply standardisation to a batch of inputs.

        Uses the inputs to update the running stats if in training mode.
        """

        # Save and check the gradient tracking options
        grad_setting = T.is_grad_enabled()
        T.set_grad_enabled(self.track_grad_forward)

        # check that on same device
        inpt = self.check_device(inpt)
        if mask is not None:
            mask = self.check_device(mask)
        
        # define the batch size before mask is applied
        self._n = len(inpt)

        # Mask the inputs and update the stats
        sel_inpt = self._mask(inpt, mask)

        # Only update if in training mode
        if self.training and training:
            self.update(sel_inpt)


        # Apply the mapping

        normed_inpt = (sel_inpt - self.means) / (self.vars.sqrt() + 1e-8)

        # Undo the masking
        normed_inpt = self._unmask(inpt, normed_inpt, mask)

        # Revert the gradient setting
        T.set_grad_enabled(grad_setting)

        return normed_inpt

    def reverse(self, inpt: T.Tensor, mask: Union[T.BoolTensor, None] = None) -> T.Tensor:
        """Unnormalises the inputs given the recorded stats."""

        # check if correct device
        inpt = self.check_device(inpt)
        if mask is not None:
            mask = self.check_device(mask)
            
        # Save and check the gradient tracking options
        grad_setting = T.is_grad_enabled()
        T.set_grad_enabled(self.track_grad_reverse)

        # Mask, revert the inputs, unmask
        sel_inpt = self._mask(inpt, mask)
        unnormed_inpt = sel_inpt * self.vars.sqrt() + self.means
        unnormed_inpt = self._unmask(inpt, unnormed_inpt, mask)

        # Revert the gradient setting
        T.set_grad_enabled(grad_setting)

        return unnormed_inpt

    def update(self, inpt: T.Tensor) -> None:
        """Update the running stats using a batch of data."""

        # Freeze the model if we already exceed the requested stats
        T.fill_(self.frozen, self.n >= self.max_iters)
        if self.frozen:
            return

        # For first iteration, just run the fit on the batch
        if self.n == 0:
            self.fit(inpt, freeze=False)
            return

        # Otherwise update the statistics
        if self.do_ema:
            self._apply_ema_update(inpt)
        else:
            self._apply_welford_update(inpt)

    @T.no_grad()
    def _apply_ema_update(self, inpt: T.Tensor) -> None:
        """Use an exponential moving average to update the means and vars."""
        self.n += self._n
        nm = inpt.mean(dim=(0, *self.extra_dims), keepdim=True)
        self.means = self.ema_sync * self.means + (1 - self.ema_sync) * nm
        nv = (inpt - self.means).square().mean((0, *self.extra_dims), keepdim=True)
        self.vars = self.ema_sync * self.vars + (1 - self.ema_sync) * nv

class GRF(Function):
    """A gradient reversal function.

    - The forward pass is the identity function
    - The backward pass multiplies the upstream gradients by -alpha
    """

    @staticmethod
    def forward(ctx, inpt, alpha) -> T.Tensor:
        """Pass inputs without chaning them."""
        ctx.alpha = alpha
        return inpt.clone()

    @staticmethod
    def backward(ctx, grads) -> tuple:
        """Inverse the gradients."""
        alpha = ctx.alpha
        neg_grads = -alpha * grads
        return neg_grads, None


class GRL(nn.Module):
    """A gradient reversal layer.

    This layer has no parameters, and simply reverses the gradient in the backward pass.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = T.tensor(alpha, requires_grad=False)

    def forward(self, inpt):
        """Pass to the GRF."""
        return GRF.apply(inpt, self.alpha)


if __name__== '__main__':
    import matplotlib.pyplot as plt
    x = T.linspace(-5,5, 1000)
    
    for beta in [1,2,5]:
        softplus = T.nn.Softplus(beta=beta, threshold=30/(beta))
        plt.plot(x.numpy(), softplus(x).numpy(), label= beta)
    plt.xlim([-3,3])
    plt.legend()
    
    