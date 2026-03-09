"Onnx setup for optimal transport framework"
import os
import sys
from glob import glob

import numpy as np
import torch

# sys.path.insert(0, "/home/users/a/algren/.local/lib/python3.7/site-packages")

import io
import json

import onnx
from onnxruntime.capi._pybind_state import GradientGraphBuilder
import onnxruntime as ort
from tools import misc
from torch.onnx import TrainingMode

from otcalib.torch.torch_utils as ot_utils
from otcalib.torch import PICNN


# import torch

# #notice how .sum() is applied on y at the end
# class F(torch.nn.Module):
#     def forward(self, x):
#         y = x.square().sum()
#         return torch.autograd.grad(y, x, create_graph=True, retain_graph=True)

# f = F()
# x = torch.randn([3,3]).to(torch.float32)
# x.requires_grad_(True)
# test_out = f(x)
# print(test_out)

# x2 = torch.randn([3,3]).to(torch.float32)
# x2.requires_grad_(True)
# torch.onnx.export(f, x2,'dummy.onnx', input_names=['x'],
#             output_names=['out'])
# ort_session = load_onnx('dummy.onnx')


def export_onnx_model(model, attr):
    torch.onnx.export(model, verbose=True, **attr)
    return attr["f"]


def create_onnx(
    model_path: str,
    model_index: int,
    model_input: tuple,
    input_names: list,
    output_names: list,
    save_path: str = None,
    dynamic_axes=None,
    device=None,
    attr={},
    g_func:PICNN=None
):
    print(f"Best epoch:  {model_index}")

    if g_func is None:
        _, g_func = ot_utils.load_model_w_hydra(model_path, device=device)
        g_func = g_func.to(device)
        g_func.device=device
    
    # g_func.correction_trainable = False
    attr["args"] = model_input
    attr["input_names"] = input_names
    attr["output_names"] = output_names
    attr["dynamic_axes"] = dynamic_axes
    # g_func.onnx_export = True
    if "opset_version" not in attr.keys():
        attr["opset_version"] = 14
    if save_path is not None:
        attr["f"] = save_path
    else:
        attr["f"] = io.BytesIO()
    # attr["custom_opsets"] = {'ai.onnx.ml': 2}
    # attr["custom_opsets"] = {'custom_domain': 2}

    onnx_func = export_onnx_model(g_func, attr)
    print(f"ONNX-RunTime version {ort.__version__}")
    return g_func, onnx_func


def build_gradient_onnx(
    f: io.BytesIO, input_name: set, output_name: str, file_path: str
):
    exported_model = f.getvalue()
    builder = GradientGraphBuilder(
        exported_model, {output_name}, input_name, output_name
    )
    # builder.save(
    #     file_path
    # )  # this is require otherwise segmentation fault (core dumped)
    builder.build()
    builder.save(file_path)
    
    return builder


def load_onnx(path: str, verbose: bool = True):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    if verbose:
        print(onnx.helper.printable_graph(onnx_model.graph))
    return onnx_model


def onnx_runtime_load(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(
        path, providers=["CPUExecutionProvider"], sess_options=options
    )
    return ort_session


def predict_using_onnx(path, data, output: list):
    ort_session = onnx_runtime_load(path)
    outputs = ort_session.run(output, data)
    return outputs, ort_session


if __name__ == "__main__":
    "export OT model as onnx"
    import torch as T
    from tools import misc

    from otcalib.otcalib.utils import onnx_pipeline

    global_path = "/home/users/a/algren/scratch/trained_networks/ftag_calib/paper_runs/calibrations/"
    PATH = f"{global_path}/ttbar/2024_06_03_22_49_18_190896_lights_z_plus_jet"

    config = misc.load_yaml(f"{PATH}/.hydra/config.yaml")

    model_inputs = (
        T.randn(16, config.model.cvx_dim, requires_grad=True),
        T.abs(T.randn(16, config.model.noncvx_dim, requires_grad=True)),
    )

    #     # onnx_pipeline.create_and_test_onnx_model(model_inputs=model_inputs,
    #     #                                         model_path=PATH,
    #     #                                         input_names=config["columns_to_use"])
    model_path = PATH
    input_names = ["prob", "pT"]
    output_path: str = None
    output_names: list = ["grad"]
    device = "cpu"
    best_index = -1
    # sys.exit()

    # BATCHSIZE = 16
    # input_names = ["pT", "prob"]
    # output_names = ["cvx_potential"]

    # model_input = (torch.abs(torch.randn(BATCHSIZE, 1, requires_grad=True).to(device)),
    #                 torch.randn(BATCHSIZE, 3, requires_grad=True).to(device))
    if output_path is None:
        os.makedirs(f"{model_path}/onnx", exist_ok=True)
        output_path = f"{model_path}/onnx/cvx_potential.onnx"
    best_index = 0
    g_func, onnx_model = create_onnx(
        model_path=model_path,
        model_index=best_index,
        model_input=model_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={i: {0: "batch_size"} for i in input_names + output_names},
        device=device,
        # save_path = output_path
    )
    # torch.onnx.export(g_func.transport, x2,'dummy.onnx', input_names=input_names,
    #             output_names=['probs_grad'])
    
    ort_session = load_onnx('dummy.onnx')

    sys.exit()
    output_path = output_path.replace(".onnx", "_gradient.onnx")
    build_gradient_onnx(
        onnx_model,
        input_name=set(input_names[1:]),
        output_name=output_names[0],
        file_path=output_path,
    )

    if True:  # testing output
        data = {
            i: torch.randn_like(inputs).cpu().detach().numpy()
            for i, inputs in zip(input_names, model_inputs)
        }
        output_names = ["cvx_potential"]  # , input_names[-1] + "_grad"]
        outputs, ort_session = predict_using_onnx(
            output_path, data, output=output_names
        )
        output = g_func(pT.to(device), dl1r.to(device)).cpu().detach().numpy()
        transport = (
            g_func.transport(pT.to(device), dl1r.to(device)).cpu().detach().numpy()
        )
        nr_ = 0
        print(outputs[1][-20:])
        print(outputs[0][-20:])
        # sys.exit()
        if len(output_names) > 1:
            print("true output ", output)
            print("inference output", outputs[nr_])
            nr_ += 1
            print()
        print("True transport", transport)
        print("interference transport", outputs[nr_])
        print(transport - outputs[nr_])
        athena_difference = output_data[-len(outputs[0]) :] - np.concatenate(outputs, 1)
        print(
            "Mean difference:"
            f" {np.mean(athena_difference[~np.any(np.isnan(athena_difference),1)])}"
        )
        print(
            "max difference:"
            f" {np.max(athena_difference[~np.any(np.isnan(athena_difference),1)])}"
        )

    ####
    str_correct = "_testing_for_onnx_meta.onnx"

    from onnx.defs import onnx_opset_version

    onnx_model = load_onnx(output_path)
    print(f"Old opset_import: {onnx_model.opset_import}")
    opsets = {"": 14, "ai.onnx.ml": 2}
    onnx_model.domain = "ai.onnx"

    # opsets
    del onnx_model.opset_import[:]  # pylint: disable=E1101
    for dom, value in opsets.items():
        op_set = onnx_model.opset_import.add()
        op_set.domain = dom
        op_set.version = value
    print(f"New opset_import: {onnx_model.opset_import}")

    with open(output_path.replace(".onnx", str_correct), "wb") as f:
        f.write(onnx_model.SerializeToString())
