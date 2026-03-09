# Taken from https://github.com/bayesiains/nflows/blob/daee01ade3a32c0921476c1a5db4c9ae56a494d3/nflows/distributions/uniform.py
# because the context didnt work

from typing import Union

import torch as T

from nflows.distributions.base import Distribution
from nflows.utils import torchutils

import numpy as np
from torch.distributions.transforms import SoftmaxTransform


class BoxUniform(Distribution):
    def __init__(
            self,
            low: Union[T.Tensor, float],
            high: Union[T.Tensor, float],
            device:str="cpu"
    ):
        """Multidimensionqal uniform distribution defined on a box.

        Args:
            low (Tensor or float): lower range (inclusive).
            high (Tensor or float): upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
        """
        super().__init__()

        if not T.is_tensor(low):
            low = T.tensor(low, dtype=T.float32,device=device)
        if not T.is_tensor(high):
            high = T.tensor(high, dtype=T.float32,device=device)

        if low.shape != high.shape:
            raise ValueError(
                "low and high are not of the same size"
            )

        if not (low < high).byte().all():
            raise ValueError(
                "low has elements that are higher than high"
            )
        self.device=device
        self._shape = low.shape
        self._low = low
        self._high = high
        self._log_prob_value = -T.sum(T.log(high - low))

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        return self._log_prob_value.expand(inputs.shape[0]).to(context.device)

    def _sample(self, num_samples, context):
        context_size = 1 if context is None else context.shape[0]
        low_expanded = self._low.expand(context_size * num_samples, *self._shape)
        high_expanded = self._high.expand(context_size * num_samples, *self._shape)
        samples = low_expanded + T.rand(context_size * num_samples, *self._shape,
                                            device=self.device) * (high_expanded - low_expanded)

        if context is None:
            return samples
        else:
            return torchutils.split_leading_dim(samples, [context_size, num_samples])
        
class Dirichlet(Distribution):
    def __init__(self, alpha, logit=False, drop_dim=0, device="cuda"):
        super().__init__()
        self.alpha = alpha
        self.logit=logit
        self.drop_dim=drop_dim
        self.base_dist = T.distributions.dirichlet.Dirichlet(alpha.to(device))
        self._shape = alpha.shape
        self.batch_shape=512

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if self.drop_dim >0:
            inputs = T.concat([inputs,1-inputs.sum(1).view(-1,1)],1)
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        if self.logit:
            inputs = self.probsfromlogits(inputs)
        self._log_prob_value = self.base_dist.log_prob(inputs)
        return self._log_prob_value.view(len(inputs),-1).to(inputs.device)

    @staticmethod
    def probsfromlogits(logitps: np.ndarray) -> np.ndarray:
        """reverse transformation from logits to probs

        Parameters
        ----------
        logitps : np.ndarray
            arrray of logit

        Returns
        -------
        np.ndarray
            probabilities from logit
        """
        norm=1
        ps_value = 1.0 / (1.0 + T.exp(-logitps))
        # ps_value = T.exp(ps_value)
        # ps_value = ps_value/ps_value.sum(1).view(-1,1)
        if (ps_value.shape[-1]>1) and (len(ps_value.shape)>1):
            norm = T.sum(ps_value, axis=1)
            norm = T.stack([norm] * logitps.shape[1]).T
        return ps_value / norm

    def _sample(self, num_samples, context):
        context_size = 1 if context is None else context.shape[0]
        samples = self.base_dist.sample([context_size*num_samples])
        
        if self.logit:
            samples = T.log(samples/(1-samples))

        #make the distribution unbound
        samples = samples[:,self.drop_dim:]

        if context is None:
            return samples
        else:
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

class SoftmaxTrans(SoftmaxTransform):
    def __init__(self):
        super().__init__()
        
    def jacobian(self, s):
        return T.diag(s) - T.mm(s.T, s)
    
    def log_abs_det_jacobian(self, x,y ):
        s = self._call(x)
        J = self.jacobian(s)
        return J.det().abs().log()

