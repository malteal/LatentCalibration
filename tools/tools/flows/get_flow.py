"""
Functions and classes used to define the learnable and invertible transformations used
"""

from copy import deepcopy
import logging
import torch as T
from torch import nn
from ..discriminator import DenseNet
from .base_distribution import BoxUniform, Dirichlet

from nflows.transforms import (
    CompositeTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    MaskedAffineAutoregressiveTransform,
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
    LULinear,
    BatchNorm,
    ActNorm,
    standard
)
from nflows.flows.base import Flow as nFlow
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation

from ..torch_utils import activation_functions as get_act

class DenseNetwork(DenseNet):
    def __init__(self, input_dim, ctxt, N=32, n_layers=4, sigmoid=False, activation_str="leaky_relu", output_dim=1, device="cpu", network_type="clf", **kwargs):
        self.ctxt=ctxt
        self.input_dim=input_dim
        input_dims = input_dim+ctxt
        
        super().__init__(input_dims, N, n_layers, sigmoid, activation_str, output_dim, device, network_type, **kwargs)

    def forward(self, inputs: T.Tensor, ctxt = None) -> T.Tensor:
        """Pass through all layers of the dense network."""

        # Reshape the context if it is available. Equivalent to performing
        # multiple ctxt.unsqueeze(1) until the dim matches the input.
        # Batch dimension is kept the same.
        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)
            inputs = T.concat([inputs, ctxt], 1)

        # Pass through the input block
        # inputs = self.input_block(inputs, ctxt)
        output = self.network(inputs)

        return output



def key_change(dic: dict, old_key: str, new_key: str, new_value=None) -> None:
    """Changes the key used in a dictionary inplace only if it exists"""

    ## If the original key is not present, nothing changes
    if old_key not in dic:
        return

    ## Use the old value and pop. Essentially a rename
    if new_value is None:
        dic[new_key] = dic.pop(old_key)

    ## Both a key change AND value change. Essentially a replacement
    else:
        dic[new_key] = new_value
        del dic[old_key]


def change_kwargs_for_made(old_kwargs):
    """Converts a dictionary of keyword arguments for configuring a mattstools
    DenseNetwork to one that can initialise a MADE network for the nflows package
    with similar (not exactly the same) hyperparameters
    """
    new_kwargs = deepcopy(old_kwargs)

    ## Certain keys must be changed
    key_change(new_kwargs, "ctxt_dim", "context_features")
    key_change(new_kwargs, "drp", "dropout_probability")
    key_change(new_kwargs, "do_res", "use_residual_blocks")

    ## Certain keys are changed and their values modified
    if "act_h" in new_kwargs:
        new_kwargs["activation"] = get_act(new_kwargs.pop("act_h"))
    if "nrm" in new_kwargs:  ## MADE only supports batch norm!
        new_kwargs["use_batch_norm"] = new_kwargs.pop("nrm") is not None

    ## Some options are missing
    missing = ["ctxt_in_all", "n_lyr_pbk", "act_o", "do_out"]
    for miss in missing:
        if miss in new_kwargs:
            del new_kwargs[miss]

    ## The hidden dimension passed to MADE as an arg, not a kwarg
    if "hddn_dim" in new_kwargs:
        hddn_dim = new_kwargs.pop("hddn_dim")
    ## Otherwise use the same default value for mattstools.modules.DenseNet
    else:
        hddn_dim = 32

    return new_kwargs, hddn_dim


def get_simple_flow_network(n_features, num_layers=5):
    context_encoder = T.nn.Sequential(
        T.nn.Linear(1, 16),
        T.nn.ReLU(),
        T.nn.Linear(16, 16),
        T.nn.ReLU(),
        T.nn.Linear(16, n_features * 2),
    )

    base_dist = ConditionalDiagonalNormal(
        shape=[n_features], context_encoder=context_encoder
    )

    transforms = []
    for _ in range(num_layers):
        # transforms.append(RandomPermutation(features=n_features))
        transforms.append(ReversePermutation(features=n_features))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=n_features, hidden_features=8, context_features=1
            )
        )
    transform = CompositeTransform(transforms)

    flow = nFlow(transform, base_dist)
    return flow


def stacked_norm_flow(
    xz_dim: int,
    ctxt_dim: int = 0,
    nstacks: int = 3,
    param_func: str = "made",
    invrt_func: str = "rqs",
    permutation: str = "",
    nrm: str = "",
    base_dist: str = "normal",
    net_kwargs: dict = None,
    rqs_kwargs: dict = None,
    device:str="cpu",
    **kwargs
) -> nFlow:
    """
    Create a stacked flow using a either autoregressive or coupling layers to learn the
    paramters which are then applied to elementwise invertible transforms, which can
    either be a rational quadratic spline or an affine layer.

    After each of these transforms, there can be an extra invertible
    linear layer, followed by some normalisation.

    args:
        xz_dim: The number of input X (and output Z) features
    kwargs:
        ctxt_dim: The dimension of the context feature vector
        nstacks: The number of NSF+Perm layers to use in the overall transform
        param_func: To use either autoregressive or coupling layers
        invrt_func: To use either spline or affine transformations
        do_lu: Use an invertible linear layer inbetween splines to encourage mixing
        nrm: Do a scale shift normalisation inbetween splines (batch or act)
        net_kwargs: Kwargs for the network constructor (includes ctxt dim)
        rqs_kwargs: Keyword args for the invertible spline layers
    """

    ## Dictionary default arguments (also protecting dict from chaning on save)
    net_kwargs = deepcopy(net_kwargs) or {}
    rqs_kwargs = deepcopy(rqs_kwargs) or {}

    ## We add the context dimension to the list of network keyword arguments
    net_kwargs["ctxt_dim"] = ctxt_dim

    ## For MADE netwoks change kwargs from mattstools to nflows format
    if param_func == "made":
        made_kwargs, hddn_dim = change_kwargs_for_made(net_kwargs)

    ## For coupling layers we need to define a custom network maker function
    elif param_func == "cplng":
        # raise TypeError("not working with denseNetwork from mattstools")
        def net_mkr(inpt, outp):
            return DenseNetwork(input_dim=inpt,
                                output_dim=outp,
                                **net_kwargs)

    ## Start the list of transforms out as an empty list
    trans_list = []

    ## Start with a mixing layer
    if "lu" in  permutation.lower():
        trans_list.append(LULinear(xz_dim))
    # else:
    #     trans_list.append(RandomPermutation(features=xz_dim))

    ## Cycle through each stack
    tail_bound = rqs_kwargs.get("tail_bound", None)
    tails = rqs_kwargs.get("tails", None)

    for i in range(nstacks):
        # if tails:
        #     tb = tail_bound
        # else:
        #     tb = tail_bound if i == 0 else None
        # rqs_kwargs["tail_bound"]=tb
        ## For autoregressive funcions
        # if ((tails == None)
        #     and (not tail_bound == None)
        #     and (i == nstacks - 1)):
        #     trans_list += [standard.PointwiseAffineTransform(shift=-tail_bound, scale=2 * tail_bound)] # X = X * scale + shift.

        if param_func == "made":
            if invrt_func == "aff":
                trans_list.append(
                    MaskedAffineAutoregressiveTransform(xz_dim, hddn_dim, **made_kwargs)
                )

            elif invrt_func == "rqs":
                trans_list.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        xz_dim, hddn_dim, **made_kwargs, **rqs_kwargs
                    )
                )

        ## For coupling layers
        elif param_func == "cplng":

            ## Alternate between masking first half and second half (rounded up)
            mask = T.abs(T.round(T.arange(xz_dim) / (xz_dim - 1)).int() - i % 2)

            if invrt_func == "aff":
                trans_list.append(AffineCouplingTransform(mask, net_mkr))

            elif param_func == "cplng" and invrt_func == "rqs":
                trans_list.append(
                    PiecewiseRationalQuadraticCouplingTransform(
                        mask, net_mkr, **rqs_kwargs
                    )
                )


        ## Add the mixing layers
        if "lu" in permutation.lower():
            trans_list.append(LULinear(xz_dim))
        elif "reverse" in permutation.lower():
            trans_list.append(ReversePermutation(features=xz_dim))
            

        ## Normalising layers (never on last layer in stack)
        if i < nstacks - 1:
            if nrm == "batch":
                trans_list.append(BatchNorm(xz_dim))
            elif nrm == "act":
                trans_list.append(ActNorm(xz_dim))

    ## Return the list of transforms combined
    base_input_dim=xz_dim+kwargs.get("drop_dim", 0)
    
    if "trainable_normal" in base_dist.lower():
        context_encoder = T.nn.Sequential(
            T.nn.Linear(ctxt_dim, 16),
            T.nn.ReLU(),
            T.nn.Linear(16, 16),
            T.nn.ReLU(),
            T.nn.Linear(16, base_input_dim * 2),
        )

        base_dist = ConditionalDiagonalNormal(
            shape=[base_input_dim], context_encoder=context_encoder
        )
    elif "uniform" in base_dist.lower():

        tail_bound = rqs_kwargs.get("tail_bound", [0.0,1.0])

        if tail_bound==1:
            tail_bound = [0, 1]
        if isinstance(tail_bound, (float, int)):
            tail_bound = [-tail_bound, tail_bound]
        # else:
        #     tail_bound = [0.0,1.0]
        base_dist = BoxUniform([tail_bound[0]]*base_input_dim,
                               [tail_bound[1]]*base_input_dim,
                               device=device)
        # base_dist = T.distributions.uniform.Uniform(
        #     T.tensor(0.0, device = device),
        #     T.tensor(1.0, device = device)
        #     )
        if  kwargs.get("logit", False):
            transforms = [T.distributions.transforms.SigmoidTransform().inv]
            base_dist = T.distributions.transformed_distribution.TransformedDistribution(
                base_dist, transforms)

    elif "dirichlet" in base_dist.lower():
        base_dist = Dirichlet(T.tensor([1.0]*(base_input_dim)),
                              logit = kwargs.get("logit", False),
                              device=device,
                              drop_dim=kwargs.get("drop_dim", 0))
    else:
        base_dist = StandardNormal(shape=[base_input_dim])
    flow_model = nFlow(CompositeTransform(trans_list), base_dist)
    
    # calculate the number of trainable parameters
    
    flow_model = nFlow(CompositeTransform(trans_list), base_dist)
    num_trainable_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    logging.info("Number of trainable parameters in the flow model: %d", num_trainable_params)
    return flow_model


# def spline_flow(inp_dim, nodes, num_blocks=2, nstack=3, tail_bound=None, tails=None, activation=F.relu, lu=0,
#                 num_bins=10, context_features=None):
#     transform_list = []
#     for i in range(nstack):
#         # If a tail function is passed apply the same tail bound to every layer, if not then only use the tail bound on
#         # the final layer
#         tpass = tails
#         if tails:
#             tb = tail_bound
#         else:
#             tb = tail_bound if i == 0 else None
#         transform_list += [
#             transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, nodes, num_blocks=num_blocks,
#                                                                                tail_bound=tb, num_bins=num_bins,
#                                                                                tails=tpass, activation=activation,
#                                                                                context_features=context_features)]
#         if (tails == None) and (not tail_bound == None) and (i == nstack - 1):
#             transform_list += [transforms.standard.PointwiseAffineTransform(-tail_bound, 2 * tail_bound)]

#         if lu:
#             transform_list += [transforms.LULinear(inp_dim)]
#         else:
#             transform_list += [transforms.ReversePermutation(inp_dim)]

#     return transforms.CompositeTransform(transform_list[:-1])


# def coupling_inn(inp_dim, maker, nstack=3, tail_bound=None, tails=None, lu=0, num_bins=10, mask=[1, 0],
#                  unconditional_transform=False, spline=True, curtains_transformer=False):
#     transform_list = []
#     for i in range(nstack):
#         # If a tail function is passed apply the same tail bound to every layer, if not then only use the tail bound on
#         # the final layer
#         tpass = tails
#         if tails:
#             tb = tail_bound
#         else:
#             tb = tail_bound if i == 0 else None
#         if spline:
#             transform_list += [
#                 transforms.PiecewiseRationalQuadraticCouplingTransform(mask, maker, tail_bound=tb, num_bins=num_bins,
#                                                                        tails=tpass,
#                                                                        apply_unconditional_transform=unconditional_transform)]
#             if (tails == None) and (not tail_bound == None) and (i == nstack - 1):
#                 transform_list += [transforms.standard.PointwiseAffineTransform(-tail_bound, 2 * tail_bound)]
#         else:
#             transform_list += [
#                 transforms.AffineCouplingTransform(mask, maker)]
#             if unconditional_transform:
#                 warnings.warn('Currently the affine coupling layers only consider conditional transformations.')

#         if lu:
#             transform_list += [transforms.LULinear(inp_dim)]
#         else:
#             transform_list += [transforms.ReversePermutation(inp_dim)]

#     if not (curtains_transformer and (nstack % 2 == 0)):
#         # If the above conditions are satisfied then you want to permute back to the original ordering such that the
#         # output features line up with their original ordering.
#         transform_list = transform_list[:-1]

#     return transforms.CompositeTransform(transform_list)

if __name__ == "__main__":
    flow_test = stacked_norm_flow(1, 1)
    print(flow_test)
