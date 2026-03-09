import argparse

#import pyrootutils

#root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(
        description="Convert JetClass files to a more usable format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/data/",
        help="The path to the JetClass files",
    )

    parser.add_argument(
        "--pc_max",
        type=int,
        default=190,
        help="Maximum length of a point cloud",
    )
    return parser.parse_args()

def extend_dims(data, max_len, idx_to_extend):
    """
    Extends the dimensions of the data at the specified index to match max_len.

    Parameters:
    data (np.ndarray): The input data array.
    max_len (int): The maximum length to extend to.
    idx_to_extend (int): The index at which to extend the dimensions.

    Returns:
    np.ndarray: The data array with extended dimensions.
    """
    # Check if the length of data at idx_to_extend is less than max_len
    if data.shape[idx_to_extend] < max_len:
        # Calculate the difference
        diff = max_len - data.shape[idx_to_extend]
        
        # Create a padding array with the same shape as data but with the extended dimension
        pad_shape = list(data.shape)
        pad_shape[idx_to_extend] = diff
        padding = np.zeros(pad_shape, dtype=data.dtype)
        
        # Concatenate the original data with the padding array along the specified axis
        data = np.concatenate((data, padding), axis=idx_to_extend)
        
    return data

def extend_buffer(buffer:dict, keys_to_extend:dict, max_len:int):
    """
    Extends the dimensions of the data in the buffer at the specified index to match max_len."""
    
    for k, idx in keys_to_extend.items():

        for nr, val in enumerate(buffer[k]):

            buffer[k][nr] = extend_dims(val, max_len, idx)
    
    return buffer

def main() -> None:
    """Combine all jetclass files into a single HDF5 file."""
    # Get the arguments
    args = get_args()
    pc_max = args.pc_max
    sub_chunk_size = 1
    multiple_files=False
    
    # Get the top level folders (train, val, test)
    subsets = [x for x in Path(args.data_path).iterdir() if x.is_dir()]

    # Cycle through each subset
    for subset in subsets:

        # Skip the train set
        if "ttbar" not in subset.name:
            print("Skipping", subset.name)
            continue

        print(f"Processing {subset.name}")

        # Create the target file
        target_file = Path(args.data_path) / f"{subset.name}_combined.h5"
        if not multiple_files:
            h5fw = h5py.File(target_file, mode="w")
        row = 0  # Counter for current location

        # Get a list of all files in the subset and sort
        files = list(subset.glob("*.h5"))

        # Get the name of the keys from the first file
        with h5py.File(files[0], "r") as h5fr:
            buffer = {k: [] for k in h5fr}

        # Get a list of common numbers in the file names
        # This way we ensure each buffer has one file of each type
        common_nums = np.unique([int(x.stem.split("_")[-1]) for x in files])
        for num in tqdm(common_nums, desc="Total progress"):
            # Reset the buffer
            for k in buffer:
                buffer[k] = []

            if multiple_files:
                target_file = Path(args.data_path) / f"{subset.name}_combined_{num}.h5"
                h5fw = h5py.File(target_file, mode="w")

            # Cycle through each file
            sublist = [x for x in files if int(x.stem.split("_")[-1]) == num]
            for h5name in tqdm(sublist, leave=False, desc=f"Loading {num} h5 file"):
                with h5py.File(h5name, "r") as h5fr:
                    for k in buffer:
                        buffer[k].append(h5fr[k][:])

            # Shuffle each list in the buffer
            len_buff = sum(len(v) for v in buffer[k])
            order = np.random.default_rng().permutation(len_buff)
            
            # in-place operation!
            extend_buffer(buffer, {"csts": 1, 'csts_id':1, 'mask': 1}, 
                          max([i.shape[1] for i in buffer['csts']]) if multiple_files else pc_max)

            for k in buffer:
                buffer[k] = np.concatenate(buffer[k], axis=0)[order]

            # Write the buffer to the target file
            for k, v in tqdm(buffer.items(), leave=False, desc=f"Writing to {num} file"):
                # Create the dataset if it doesn't exist
                if row == 0:
                    h5fw.create_dataset(
                        k,
                        dtype=v.dtype,
                        shape=v.shape,
                        chunks=(1000, *v.shape[1:]),
                        maxshape=(None, *v.shape[1:]),
                    )

                # Resize the target table if it is too small
                if row + len_buff > len(h5fw[k]):
                    h5fw[k].resize((row + len_buff, *v.shape[1:]))

                
                if sub_chunk_size==1:
                    # Save the data
                    h5fw[k][-v.shape[0]:] = v
                else:
                    # Define sub-chunk size
                    
                    # Split the new chunk into smaller sub-chunks
                    sub_chunks = [v[i:i+sub_chunk_size] for i in range(0, v.shape[0], sub_chunk_size)]

                    # Write each sub-chunk
                    start_idx = h5fw[k].shape[0] - v.shape[0]
                    for sub_chunk in tqdm(sub_chunks, leave=False, desc=f"Writing sub-chunks"):
                        h5fw[k][start_idx : start_idx+len(sub_chunk)] = sub_chunk
                        
                        start_idx += len(sub_chunk)
            
            if multiple_files:
                h5fw.close()
            else:
                row += len_buff

        # Close the file
        h5fw.close()


if __name__ == "__main__":
    main()
