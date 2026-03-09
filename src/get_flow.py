"""
Functions and classes used to define the learnable and invertible transformations used
"""

from copy import deepcopy
import torch as T
import os
from glob import glob

# ot-framework imports
from otcalib.otcalib.torch.layers import get_act_funcs as get_act
from otcalib.otcalib.utils.misc import load_yaml


# nflows imports
from nflows.transforms import (
    CompositeTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    MaskedAffineAutoregressiveTransform,
    # AffineCouplingTransform,
    # PiecewiseRationalQuadraticCouplingTransform,
)
from tools.tools.flows import BoxUniform
from nflows.flows.base import Flow as nFlow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation

# from tools.torch_utils import activation_functions as get_act

# class DenseNetwork(DenseNet):
#     def __init__(self, input_dim, ctxt, N=32, n_layers=4, sigmoid=False, activation_str="leaky_relu", output_dim=1, device="cpu", network_type="clf", **kwargs):
#         self.ctxt=ctxt
#         self.input_dim=input_dim
#         input_dims = input_dim+ctxt
        
#         super().__init__(input_dims, N, n_layers, sigmoid, activation_str, output_dim, device, network_type, **kwargs)

#     def forward(self, inputs: T.Tensor, ctxt = None) -> T.Tensor:
#         """Pass through all layers of the dense network."""

#         # Reshape the context if it is available. Equivalent to performing
#         # multiple ctxt.unsqueeze(1) until the dim matches the input.
#         # Batch dimension is kept the same.
#         if ctxt is not None:
#             dim_diff = inputs.dim() - ctxt.dim()
#             if dim_diff > 0:
#                 ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
#                 ctxt = ctxt.expand(*inputs.shape[:-1], -1)
#             inputs = T.concat([inputs, ctxt], 1)

#         # Pass through the input block
#         # inputs = self.input_block(inputs, ctxt)
#         output = self.network(inputs)

#         return output

def load_path_info(model_path, yaml_bool=False):

    model_paths = sorted(glob(model_path + "/models/*"), key=os.path.getctime)
    print(f"Best path {model_paths[-1].split('/')[-1]}")
    # if yaml_bool:
    try:
        data_settings = load_yaml(model_path + "/data_settings.yaml")
        flow_args = load_yaml(model_path + "/flow_args.yaml")
    except FileNotFoundError:
        data_settings = load_json(model_path + "/data_settings.json")
        flow_args = load_json(model_path + "/flow_args.json")

    return data_settings, flow_args, model_paths[-1]


def load_flows(flow, path, device):
    "Load the pretrain sig and bkg flows"
    _, _, model_path = load_path_info(path, yaml_bool=True)
    # load flow
    flow = flow.to(device)
    model_state = T.load(model_path, map_location=device)
    flow.load_state_dict(model_state["flow"])
    return flow


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

def stacked_norm_flow(
    xz_dim: int,
    ctxt_dim: int = 0,
    nstacks: int = 3,
    param_func: str = "made",
    invrt_func: str = "rqs",
    net_kwargs: dict = None,
    rqs_kwargs: dict = None,
    device:str="cuda",
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
        raise NotImplementedError("Coupling layers not yet implemented. Use MADE")
        # # raise TypeError("not working with denseNetwork from mattstools")
        # def net_mkr(inpt, outp):
        #     return DenseNetwork(input_dim=inpt,
        #                         output_dim=outp,
        #                         **net_kwargs)

    ## Start the list of transforms out as an empty list
    trans_list = []

    for _ in range(nstacks):

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
        else:
            raise NotImplementedError("Only MADE implemented")

        trans_list.append(ReversePermutation(xz_dim))
        # ## For coupling layers
        # elif param_func == "cplng":

        #     ## Alternate between masking first half and second half (rounded up)
        #     mask = T.abs(T.round(T.arange(xz_dim) / (xz_dim - 1)).int() - i % 2)

        #     if invrt_func == "aff":
        #         trans_list.append(AffineCouplingTransform(mask, net_mkr))

        #     elif param_func == "cplng" and invrt_func == "rqs":
        #         trans_list.append(
        #             PiecewiseRationalQuadraticCouplingTransform(
        #                 mask, net_mkr, **rqs_kwargs
        #             )
        #         )

    base_input_dim=xz_dim+kwargs.get("drop_dim", 0)

    if "uniform" in kwargs.get("base_dist", "").lower():

        tail_bound = rqs_kwargs.get("tail_bound", [0.0,1.0])

        if tail_bound==1:
            tail_bound = [0.0,1.0]
        elif isinstance(tail_bound, (float, int)):
            tail_bound = [-tail_bound, tail_bound]
        base_dist = BoxUniform([tail_bound[0]]*base_input_dim,
                               [tail_bound[1]]*base_input_dim,
                               device=device)
    else:
        base_dist = StandardNormal(shape=[base_input_dim])

    return nFlow(CompositeTransform(trans_list), base_dist)



if __name__ == "__main__":
    flow_test = stacked_norm_flow(1, 1)
    print(flow_test)
