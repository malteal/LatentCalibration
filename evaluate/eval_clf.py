'evaluate the classifier performance'
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
from pathlib import Path
import hydra
import numpy as np
import torch as T
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from src.datamodule import load_model_and_config


if __name__ == "__main__":
    glob_path = Path("/srv/beegfs/scratch/groups/rodem/latn_calib/latn_calib_clf")
    
    model_path = glob_path / "latn_calib_clf_jetclass_2025_01_30_14_25_58_435673"
    
    model, data_config = load_model_and_config(model_path)
    
    datamodule = hydra.utils.instantiate(data_config)
    
    val_dataloader = datamodule.val_dataloader()
    
    dict_values = {'labels': [], 'logits': []}

    with T.no_grad():
        for nr, batch in tqdm(enumerate(val_dataloader)):
            dict_values['labels'].append(
                batch.pop("labels").cpu().numpy()
                )
            batch['ctxt'] = batch.pop("scalars")
            batch = {k: v.to(model.device) for k, v in batch.items()}
            latn = model.pc_classifier(**batch)
            dict_values['logits'].append(
                model.classifier(latn).cpu().numpy()
                )

            if nr==100:
                break
    
    dict_values = {k: np.concatenate(v, 0) for k, v in dict_values.items()}
    
    print(f'Shape: {dict_values["logits"].shape}')
    
    print('Accuracy:', np.mean(np.argmax(dict_values['logits'], 1) == dict_values['labels']))
