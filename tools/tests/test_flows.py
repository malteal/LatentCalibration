import shutil
from tools.flows import Flow
import numpy as np
class TestFlow():
    n_epochs = 5
    flow_config = {"xz_dim":2, "ctxt_dim": 2,
                   "rqs_kwargs":{"tail_bound": 5,
                                 "tails": "linear",
                                 "num_bins": 10}}
    train_config = {"lr":1e-3, "n_epochs": n_epochs, "epoch_length": 100}
    data = np.c_[np.random.uniform(-1,1, size=(10_000,2)), 
                 np.random.normal(0,1, size=(10_000,2))]
    
    def test_train(self):
        flow = Flow(self.flow_config, self.train_config,
                    save_path="./flow_test", device="cpu",
                    add_time_to_dir=False)
        
        flow.create_loaders(np.float32(self.data[2000:]),
                            np.float32(self.data[:2000]))
        flow.train()
        assert len(flow.log_file.log["loss"])==self.n_epochs, "Flow training not working"
        shutil.rmtree(flow.save_path)

