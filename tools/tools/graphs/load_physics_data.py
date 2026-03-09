"load physics data"
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch_geometric.data import Data
import h5py

def csv_to_graph(paths, columns, n_cnsts=None, device = "cpu"):
    "Used to load the csv format from Fast calo sim"
    graph_list = []
    data = pd.DataFrame()
    for nr, path in enumerate(paths):
        single_file = pd.read_csv(path)
        if nr == 0:
            data = single_file
        else:
            single_file.entry = single_file.entry+data.entry.max()
            data = pd.concat([data, single_file])
    entries = data.entry.drop_duplicates()
    data["full_energy"]=np.log(data["full_energy"])

    data[columns] = (data[columns]-data[columns].mean())/data[columns].std()
    for i in tqdm(entries, total=len(entries), disable=True):
        mask = data.entry == i

        edge_data = torch.tensor(data[mask][columns].values)[:n_cnsts]

        mask = torch.any(edge_data != np.float64(-999),1)
        edges = torch.arange(torch.sum(mask))
        edge_index = torch.combinations(edges).t()

        graph_list.append(Data(
                            edge_index=edge_index.clone().detach().requires_grad_(False).to(device), 
                            x=edge_data.clone().detach().requires_grad_(True).to(device).float()
                            , requires_grad=True)
                            )
    return graph_list

def load_g4_fcs_data(g4_paths, fcs_paths, columns, n_cnsts=None):
    "loading geant4 and fast calo sim"
    data = {"g4":[], "fcs":[]}
    for i, name in zip([g4_paths, fcs_paths], list(data.keys())):
        single_file = csv_to_graph(i, columns, n_cnsts=n_cnsts)
        data[name].extend(single_file)
    return data

def load_lhco(n_jets=3, n_cnsts=None, 
              path = "/home/users/a/algren/work/cp-flows/data/events_anomalydetection_v2.h5"):
    "load LHCo samples"
    with h5py.File(path, "r") as file: 
        data = file["df"]["block0_values"][:n_jets]
    data = np.reshape(data[:,:-1], (n_jets, 700, 3))
    data = np.take_along_axis(data, np.repeat(np.argsort(data[:,:,0], 1)[:,::-1][...,None],3,axis=2), axis=1) # fucking stupid!
    return data[:,:n_cnsts, :]