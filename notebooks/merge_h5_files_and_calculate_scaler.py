import h5py
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import joblib

def merge_h5_files(output_file, input_files):
    with h5py.File(output_file, 'w') as h5_out:
        for input_file in tqdm(input_files):
            with h5py.File(input_file, 'r') as h5_in:
                for dataset_name, data in h5_in.items():
                    data = data[:]

                    if dataset_name not in h5_out:
                        # Create dataset in output file if it doesn't exist
                        maxshape = (None,) + data.shape[1:]  # Ensure maxshape has the same rank as data shape
                        h5_out.create_dataset(dataset_name, data=data, maxshape=maxshape, chunks=True)
                    else:
                        # Append data to existing dataset
                        h5_out[dataset_name].resize((h5_out[dataset_name].shape[0] + data.shape[0]), axis=0)
                        h5_out[dataset_name][-data.shape[0]:] = data

if __name__ == "__main__":
    # Define path to directory containing .h5 files

    # baseline model
    path = Path('/srv/beegfs/scratch/groups/rodem/latn_calib/latn_calib_clf/latn_calib_clf_jetclass_2025_02_03_17_22_09_461014/predictions/PSmurr2_smeared')
    # path = Path('/srv/beegfs/scratch/groups/rodem/latn_calib/latn_calib_clf/latn_calib_clf_jetclass_2025_02_03_17_22_09_461014/predictions/BigBelloNominal')
    
    # only 128 model
    # path = Path('/srv/beegfs/scratch/groups/rodem/latn_calib/latn_calib_clf/latn_calib_clf_jetclass_2025_01_31_07_52_34_378267/predictions/BigBelloNominal')
    path = Path('/srv/beegfs/scratch/groups/rodem/latn_calib/latn_calib_clf/latn_calib_clf_jetclass_2025_01_31_07_52_34_378267/predictions/PSmurr2_smeared')
    
    # defining output files
    train_output_file = path / 'train_combined.h5'  # Output .h5 file
    valid_output_file = path / 'valid_combined.h5'  # Output .h5 file

    # Filter out the combined files from the list of files to be merged
    train_files = [f for f in path.rglob('train*.h5') if f.name != 'train_combined.h5']
    valid_files = [f for f in path.rglob('valid*.h5') if f.name != 'valid_combined.h5']

    if True:
        # merge the files
        merge_h5_files(train_output_file, train_files)
        merge_h5_files(valid_output_file, valid_files)
    
    if True:
        # calculate scaler for the latn space
        
        # Create scaler 
        combined_data_file = h5py.File(train_output_file, 'r')
        
        # get only latn keys
        latn_keys = [i for i in combined_data_file if i.startswith('latn_')]

        for key in tqdm(latn_keys):
            # Create and fit the scaler
            if True:
                scaler = StandardScaler()
                scaler_name = 'scaler_'
            else:
                scaler = QuantileTransformer(output_distribution='normal',
                                             subsample=None if '16' in key else 100_000)
                scaler_name = 'scaler_quantile_'

            scaler.fit(combined_data_file[key][:None if '16' in key else 4_000_000]) # can't fit on all data, too large

            # Dump the scaler
            scaler_file = path / f'{scaler_name}{key}.pkl'
            joblib.dump(scaler, scaler_file)
