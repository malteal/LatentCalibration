"export OT model as onnx"
import pyrootutils


root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
import os
from glob import glob
import logging
import datetime
import onnx
from onnxruntime.capi._pybind_state import GradientGraphBuilder

import torch as T

from otcalib.otcalib.utils import onnx_pipeline
from otcalib.otcalib.torch import PICNN


from tools import misc, hydra_utils

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    device= "cpu"

    logging.info(f"Testing onnx model")
    
    noncvx_dim = 1
    cvx_dim = 3
    
    model_inputs = (
        T.tensor([[1.0]*cvx_dim], requires_grad=True, device=device),
        T.tensor([[1.0]*noncvx_dim], requires_grad=False, device=device),
        )
    
    input_names=["probs", "pT"]
    # input_names=["probs"]

    output_path:str=None
    output_names:list=["cvx_potential"]
    layersize=128
    n_layers=3
    best_index=-1
    g_func = PICNN.PICNN(cvx_dim=cvx_dim, noncvx_dim=noncvx_dim, cvx_layersize=layersize,
                         cvx_act='softplus', noncvx_act='relu',
                         device=device, n_layers=n_layers, noncvx_layersize=layersize,
                         correction_trainable=True)

    output_path = f"./onnx_model_test_gradient.onnx"
    

    g_func, onnx_model = onnx_pipeline.create_onnx(
        model_path=None,
        model_index=best_index,
        model_input=model_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={i: {0:"batch_size"} for i in input_names + output_names},
        device=device,
        g_func=g_func,
        # save_path = output_path
    )

    if True:
        gradient_build = onnx_pipeline.build_gradient_onnx(
            onnx_model,
            input_name=set(input_names[:1]),
            output_name=output_names[0],
            file_path=output_path,
            )

    if True: # testing output
        data = {i:inputs.cpu().detach().numpy() for i, inputs in zip(input_names, model_inputs)}
        output_names = ["cvx_potential", input_names[0] + "_grad"]
        outputs, ort_session = onnx_pipeline.predict_using_onnx(
            output_path, data, output=output_names
        )
        input1 = T.tensor(data[input_names[0]], device=device, requires_grad=True)
        input2 = T.tensor(data[input_names[1]], device=device, requires_grad=True)
        output_trans = g_func.transport(input1,input2)

        print(output_trans[::-1])
        print(outputs)

    #### dont seem to change anything atm
    if False:
        str_correct = "_testing_for_onnx_meta.onnx"

        from onnx.defs import onnx_opset_version

        onnx_model = onnx_pipeline.load_onnx(output_path)
        print(f"Old opset_import: {onnx_model.opset_import}")
        opsets = {"": 14, "ai.onnx.ml": 2}
        onnx_model.domain = "ai.onnx"

        opsets
        del onnx_model.opset_import[:]  # pylint: disable=E1101
        for dom, value in opsets.items():
            op_set = onnx_model.opset_import.add()
            op_set.domain = dom
            op_set.version = value
        print(f"New opset_import: {onnx_model.opset_import}")

        with open(output_path.replace(".onnx", str_correct), "wb") as f:
            f.write(onnx_model.SerializeToString())
