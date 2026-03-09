import argparse

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.root_utils import (
    common_particle_class,
    lifetime_signing,
    read_jetclass_file,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert JetClass files to a more usable format"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/home/users/a/algren/scratch/latn_calib/",
        # default="/srv/beegfs/scratch/groups/rodem/datasets/JetClass/Pythia/test_20M/TT*",
        help="The path to the JetClass files",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default="/home/users/a/algren/scratch/latn_calib/",

        # default="/srv/fast/share/rodem/JetClassH5_full_csts/",
        help="The path to save the converted files",
    )
    parser.add_argument(
        "--pid_bool",
        type=bool,
        default=False,
        # default="/srv/fast/share/rodem/JetClassH5_full_csts/",
        help="Use PID information for track classification",
    )
    return parser.parse_args()


def main() -> None:
    """Convert the JetClass root files to a more usable HDF format.

    The output features are the following:
    Independant continuous (7 dimensional vector)
    - 0: pt
    - 1: deta
    - 2: dphi
    - 3: d0val
    - 4: d0err
    - 5: dzval
    - 6: dzerr
    Independant categorical (single int representing following classes)
    - 0: isPhoton
    - 1: isHadron_Neg
    - 2: isHadron_Neutral
    - 3: isHadron_Pos
    - 4: isElectron_Neg
    - 5: isElectron_Pos
    - 6: isMuon_Neg
    - 7: isMuon_Pos
    """
    # Get the arguments
    args = get_args()

    # Make sure the destination path exists
    dest_path = Path(args.dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Get all of the root files in the source path
    source_path = Path(args.source_path)
    subfolders = [x for x in source_path.iterdir() if x.is_dir()]

    # Loop over the subfolders
    for subfolder in subfolders:
        
        # Skip the subfolder if it starts with an underscore
        if subfolder.name[0] == '_' or 'PSmurr2_smeared' not in subfolder.name:
            continue

        print(f"Processing {subfolder.name}")

        # Copy the subfolder to the destination path
        dest_folder = dest_path / subfolder.name

        # Make the folder
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        files = list(subfolder.glob("*.root"))

        # Sort the files based the number in the name
        files = sorted(files, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

        # Loop through the files in the subfolder and load the information
        for file in tqdm(files):
            # Define the destination file
            dest_file = dest_folder / file.name.replace(".root", ".h5")

            # Skip if the file already exists
            # if Path(dest_file).exists():
            # continue

            # Load the data from the file
            jets, csts, labels = read_jetclass_file(file, num_particles=None,
                                                    pid_bool=args.pid_bool)
            
            # Get the pt from the px and py
            pt = np.linalg.norm(csts[..., :2], axis=-1, keepdims=True)

            # Split the csts into the different groups of information
            sel_csts = np.concatenate([pt, csts[..., 2:8]], axis=-1)

            # Switch to lifetime signing convention for the impact parameters
            if False:
                d0, z0 = lifetime_signing(
                    d0=sel_csts[..., 3],
                    z0=sel_csts[..., 5],
                    tracks=sel_csts[..., :3],
                    jets=jets[..., :3],
                    is_centered=True,
                )
                sel_csts[..., 3] = d0
                sel_csts[..., 5] = z0

            # Clip eta and phi to the actual jet radius
            sel_csts[..., 1:3] = np.clip(sel_csts[..., 1:3], -0.8, 0.8)

            # Convert the particle class information to the common format
            if args.pid_bool:
                csts_id = common_particle_class(
                    charge=csts[..., -2],
                    pdgid=csts[..., -1] 
                )
            else:
                csts_id = common_particle_class(
                    charge=csts[..., -6],
                    isPhoton=csts[..., -5].astype(bool),
                    isHadron=csts[..., -4].astype(bool) | csts[..., -3].astype(bool),
                    isElectron=csts[..., -2].astype(bool),
                    isMuon=csts[..., -1].astype(bool),
                )

            labels = labels.astype(int)
            # The jet features need the number of constituents
            mask = sel_csts[..., 0] > 0
            num_csts = np.sum(mask, axis=-1, keepdims=True)
            if not all(np.ravel(num_csts)==jets[:, -1]):
                raise ValueError("Number of constituents does not match the jet value")
            # jets = np.concatenate([jets, num_csts], axis=-1)

            # Save the data to an HDF file
            with h5py.File(dest_file, "w") as f:
                f.create_dataset("csts", data=sel_csts)
                f.create_dataset("csts_id", data=csts_id)
                f.create_dataset("jets", data=jets)
                f.create_dataset("labels", data=labels)
                f.create_dataset("mask", data=mask)


if __name__ == "__main__":
    main()
