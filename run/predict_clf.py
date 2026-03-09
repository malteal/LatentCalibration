
"pipeline"
import chunk
from pyexpat import model
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
from tqdm import tqdm
import os
import hydra
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO)

@hydra.main(config_path=str(root / "configs/"), config_name="predict_clf", version_base=None)
def main(config):

    data_len = 0

    for chunk_nr in tqdm(range(config.chunks_numbers)):
        # get latn space
        data = hydra.utils.instantiate(config.get_latn,
                                       chunk_nr=data_len)

        model_path = config.get_latn.model_path
        
        save_path = f'{model_path}/predictions/'
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save as h5 file
        for nr, (dataset_name, dataset_data) in enumerate(data.items()):
            
            # calculate the size
            size = len(dataset_data['labels'])

            if nr==0:
                data_len+=size

            file_name = f'{dataset_name}'

            os.makedirs(f'{save_path}/{file_name}', exist_ok=True)
            
            logging.info("Saving in %s", file_name)

            # latn is a dict of the different latn spaces
            # unpack it with the correct names
            latn = dataset_data.pop('latn')

            for i,j in latn.items():
                dataset_data[f'latn_{i}'] = j
            
            del latn
            
            train_idx, valid_idx = train_test_split(np.arange(size), **config.split_cfg)

            # save indexing
            np.savetxt(f'{save_path}/{file_name}/train_idx_{chunk_nr}.txt', train_idx)
            np.savetxt(f'{save_path}/{file_name}/valid_idx_{chunk_nr}.txt', valid_idx)

            # indxing for train and valid
            for idx, name in zip([train_idx, valid_idx], ['train', 'valid']):

                # save in file
                with h5py.File(f'{save_path}/{file_name}/{name}_{chunk_nr}.h5', 'w') as h5file:
                    
                    # loop over all keys and save
                    for key, value in dataset_data.items():
                        h5file.create_dataset(key, data=value[idx])
    
        logging.info("Data saved to %s", save_path)

    
if __name__ == '__main__':
    # after this run merge_h5_files_and_calculate_scaler in notebooks to merge
    main()