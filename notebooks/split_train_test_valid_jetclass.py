import argparse
from pathlib import Path
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from numpy.random import RandomState

def get_args():
    parser = argparse.ArgumentParser(
        description="Convert JetClass files to a more usable format"
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="/srv/fast/share/rodem/JetClassH5/BB_nominal_ttbar_combined.h5",
        help="The path to the JetClass files",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="/home/users/a/algren/scratch/latn_calib/",
        help="The path to the target files",
    )
    parser.add_argument(
        "--pc_max",
        type=int,
        default=190,
        help="Maximum length of a point cloud",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()

def main() -> None:
    """Combine all jetclass files into a single HDF5 file."""
    # Get the arguments
    args = get_args()
    h5_path = Path(args.h5_path)
    pc_max = args.pc_max
    seed = args.seed
    target_path = Path(args.target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    random_state = RandomState(seed)
    
    # Get the top level folders (train, val, test)
    with h5py.File(h5_path, mode="r") as h5fr:
        keys = list(h5fr.keys())
        lengths = np.unique([len(j) for _, j in h5fr.items()])
        if not lengths.size == 1:
            raise ValueError("The files are not the same length!")
        
    length = lengths[0]
    # Create indices for the whole dataset
    all_indices = np.arange(length)
    
    # Shuffle all indices
    random_state.shuffle(all_indices)
    
    # Split into train, val, test
    train_idx, valid_idx = train_test_split(all_indices, test_size=0.4, random_state=seed)
    valid_idx, test_idx = train_test_split(valid_idx, test_size=0.3, random_state=seed)
    print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")

    for name, idxs in zip(["train", "val", "test"], [train_idx, valid_idx, test_idx]):
        target_file = target_path / f"{name}_{h5_path.stem}_shuffled.h5"
        
        # No need to sort indices, as we want to maintain the shuffled order
        # But we will process in batches for memory efficiency
        
        # Create the target file
        with h5py.File(target_file, "w") as h5fw:
            with h5py.File(h5_path, "r") as h5fr:
                # Create datasets in the target file with the right size
                for k in keys:
                    h5fw.create_dataset(
                        k,
                        shape=(len(idxs), *h5fr[k].shape[1:]),
                        dtype=h5fr[k].dtype,
                        chunks=True, #(1000, *h5fr[k].shape[1:]),
                    )
                
                if False: # bacthed not hsuffling tho
                    # Process data in batches
                    batch_size = 50  # Adjust based on available memory
                    for batch_start in tqdm(range(0, len(idxs), batch_size), 
                                        desc=f"Processing {name} set"):
                        # Get the batch indices
                        batch_end = min(batch_start + batch_size, len(idxs))
                        batch_idxs = idxs[batch_start:batch_end]
                        
                        # Sort batch indices for efficient reading (just this batch)
                        sorted_indices = np.sort(batch_idxs)
                        sort_idx_map = {idx: i for i, idx in enumerate(sorted_indices)}
                        
                        # Determine the range of indices to read from h5fr
                        min_idx = sorted_indices[0]
                        max_idx = sorted_indices[-1] + 1
                        
                        # Load only the required range from the source file
                        for k in keys:
                            # Load the range from source
                            data_range = h5fr[k][min_idx:max_idx]
                            
                            # Map the indices to the correct positions while maintaining shuffle order
                            for i, orig_idx in enumerate(batch_idxs):
                                sorted_pos = sort_idx_map[orig_idx]
                                idx_in_range = sorted_indices[sorted_pos] - min_idx
                                h5fw[k][batch_start + i] = data_range[idx_in_range]
                else:
                    # Load only the required range from the source file
                    for k in tqdm(keys):
                        # Load the range from source
                        # Map the indices to the correct positions while maintaining shuffle order
                        h5fw[k][:len(idxs)] = h5fr[k][:][idxs]

if __name__ == "__main__":
    main()