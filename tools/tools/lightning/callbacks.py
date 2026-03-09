# Description: Callback to save ONNX models of the networks in the model.
import os
import torch as T
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

class ONNXModelCheckpoint(Callback):
    def __init__(self, networks, dirpath,
                 filename:str = 'best_{epoch:03d}_{val_loss:.4f}',
                 monitor='val_loss', mode='min', save_top_k=1, save_last:bool=True, verbose=False, device:str='cpu') -> None:
        """
        example of networks:
        network_name:
            input_dict:
                inpt:
                func:
                    _target_: torch.randn
                    _partial_: true
                args:
                    size: [8, 128, 3]
                mask:
                func:
                    _target_: torch.randint
                    _partial_: true
                args:
                    low: 0
                    high: 1
                    size: [8, 128]
        """
        super().__init__()
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)
        
        self.networks = OmegaConf.to_container(networks, resolve=True, enum_to_str=True)
        
        self.filename = filename
        self.device=device
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.verbose = verbose
        self.best = T.inf if self.mode == 'min' else -T.inf
        self.kth_best = None
        self.kth_value = None
        self.save_last=save_last
        self.saved_models = []

    def on_train_epoch_end(self, trainer, pl_module):
        if self.save_top_k == 0:
            return

        logs = trainer.callback_metrics
        # check if model should be saved
        if self.monitor is not None:
            monitor_val = logs.get(self.monitor)
            if monitor_val is None:
                return

        if self.mode == 'min':
            is_best = monitor_val < self.best
        else:
            is_best = monitor_val > self.best

        if self.save_top_k > 0:
            if is_best:
                self.kth_best = monitor_val
                self.kth_value = monitor_val
            elif len(self.saved_models) == self.save_top_k:
                if self.mode == 'min':
                    if monitor_val > self.kth_value:
                        return
                else:
                    if monitor_val < self.kth_value:
                        return
            else:
                self.saved_models.append(monitor_val)
                self.saved_models = sorted(self.saved_models, reverse=self.mode == 'max')
                self.kth_value = self.saved_models[-1]

        if self.save_top_k == -1 or is_best:
            self.export_model(pl_module)
            
    def export_model(self, pl_module, add_str='') -> None:
        for network_name,input_dict in self.networks.items():
            
            # create inputs for the network
            inputs = {i: j['func'](**j['args'], device=self.device) for i,j in input_dict.items()}

            for i in input_dict:
                if 'mask' in i:
                    inputs[i] = inputs[i].bool()
            
            # tell which axes are dynamic
            dynamic_axes={i: {0: "batch_size"} for i in inputs}
            dynamic_axes["embedding"] = {0: "batch_size"}

            filepath = f"{self.dirpath}/{network_name}_{self.filename}{add_str}.onnx"

            filepath = filepath.format(epoch=pl_module.current_epoch, val_loss=self.kth_best)

            # export model as onnx
            T.onnx.export(model=getattr(pl_module, network_name),
                        args=inputs,
                        f = filepath,
                        input_names=list(inputs.keys()),
                        output_names=["output"],
                        export_params=True,
                        opset_version=14,
                        verbose=self.verbose,
                        dynamic_axes=dynamic_axes
                        )

            if self.verbose:
                print(f"Saved ONNX model to {filepath}")

    def on_train_end(self, trainer, pl_module):
        if self.save_last:
            self.export_model(pl_module, '_last')
