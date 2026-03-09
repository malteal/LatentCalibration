"utils"
import sys
sys.path.append("/home/users/a/algren/.local/lib/python3.7/site-packages")
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from glob import glob
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from itertools import cycle

from tools.graphs.toy_data import toy_data, get_toy_shape, circle_points
from tools.graphs.load_physics_data import load_g4_fcs_data
from tools.graphs.visualization import visualize_points

def load_graph_data(datatype, n_jets, source_shape=3, target_shape=4, n_nodes = 4,
                    r:list=None, n:list=None,x_shift=0,
                    y_shift=0, node_dst=False,
                    source_params = {"mean": (3,3), "cov": [[1,0],[0,1]], "graph_size":1},
                    n_files= None, device="cpu", max_nodes:int=0):
    "construct data from graph testing"
    if datatype=="toy":
        #Getting single point
        dst = toy_data("gauss", source_params, n_jets, node_dst= node_dst)
        #Creating shape from that single point
        source_shape = get_toy_shape(1,n_nodes)
        shift = np.zeros_like(dst)+source_shape

        # creating random changes in shap
        random = np.random.randint(0,2, n_jets)*2
        random = np.c_[np.zeros((n_jets, shift.shape[1]-1)),random]
        shift[:, ...,1] = shift[:, ...,1]+random
    
        source = dst+shift

        # selecting target shape
        target_shape = get_toy_shape(1,n_nodes, y_shift=y_shift, x_shift=x_shift)
        target_shift = np.zeros_like(dst)+target_shape

        # creating random changes in shap
        # random = np.random.randint(0,2, n_jets)*2
        # random = np.c_[np.zeros_like(random), np.zeros_like(random), random]
        target_shift[:, ...,1] = target_shift[:, ...,1]+random
        target = dst+target_shift

        target_shape = torch.tensor(target_shape, device=device)

        n_cnsts = len(source_shape)
        num_node = len(source_params["mean"])
    elif datatype == "varying_size_toy":

        #SHAPE CREATION
        source_shape = get_toy_shape(3,n_nodes)
        dst = toy_data("gauss", source_params, n_jets, node_dst= node_dst)
        shift = np.zeros_like(dst)+source_shape
        random = np.random.randint(0,2, n_jets)*2
        random = np.c_[[np.zeros_like(random) if i < n_nodes-1 else random for i in range(n_nodes)]]
        shift[:, ...,1] = shift[:, ...,1]#+random
        source = dst+shift
        optimal_transport = torch.tensor(get_toy_shape(4,n_nodes, y_shift=y_shift, 
                                                            x_shift=x_shift),
                                         device=device)

        target = dst+optimal_transport.cpu().detach().numpy()
        num_node = len(source_params["mean"])
    elif datatype=="circles":
        assert isinstance(r, list), "Using circles r as a list has to be set in the function"
        assert isinstance(n, list), "Using circles n as a list has to be set in the function"
        optimal_transport = np.array([0,-y_shift])
        source = []
        target = []
        n_cnsts = 10
        # n_jets = 3
        drop_number=2
        target_optimal=False
        nr=0
        if len(n) ==2:
            n_source, n_target = n[0], n[1]
        else:
            n_source = n
        for x,y, n_nr in tqdm(zip(np.random.uniform(-8, 8, n_jets),
                                np.random.uniform(-1, 1, n_jets),
                                np.random.randint(0, len(n_target), n_jets)),
                                total=n_jets):
            circles = circle_points(r, [int(n_source[n_nr])], drop=True, shift=np.array([x,y]),
                                    drop_number=drop_number, max_nodes=max_nodes)
            source.append(circles)

            circles = circle_points(r, [int(n_target[n_nr])],drop="neighbors", 
                                    shift=np.array([x+x_shift,y+y_shift]),
                                    drop_number=drop_number, max_nodes=max_nodes)
            target.append(circles)
            nr+=1
        num_node = source[0].shape[1]
        target = np.array(target)
        source = np.array(source)
    elif datatype=="fcs": # fcs correction
        path = "/srv/beegfs/scratch/groups/dpnc/atlas/QT-Mariia-Guillaume/clusters_ML/full_tables_features"
        g4_files = glob(path+"/*_g4_*")[:n_files]
        fcs_files = glob(path+"/*_fcs_*")[:n_files]
        columns = ["eta", "full_energy"]
        graph_data = load_g4_fcs_data(g4_files, fcs_files, columns, None)
        source = graph_data["fcs"]
        target = graph_data["g4"]
        num_node = len(columns)
    else:
        raise ValueError("Unknown datatype in *load_graph_data*")

    return source, target, num_node

def construct_torch_geo_graphs(node_data, shuffle_row=False, flip_sign=False,
                     pad_to=None, device="cpu", label=None):
    "zero padding should be zero"
    graph_list = []
    for i in tqdm(range(len(node_data))):
        if isinstance(node_data[i], Data):
            node = node_data[i].x
        else:
            node = node_data[i]
        if not isinstance(node, torch.Tensor):
            node = torch.tensor(node)
        mask = torch.any(node != np.float64(-999),1)
        edges = torch.arange(torch.sum(mask))
        edge_index = torch.combinations(edges).t()
        if shuffle_row:
            if flip_sign:
                 node[mask] = node[mask]+np.random.randint(0,2)*np.array([[0,0], [0,0], [0,-2]])
            node_attr = node[mask][torch.randperm(torch.sum(mask))]
        else:
            node_attr = node[mask]
        if pad_to is not None:
            node_attr = torch.nn.functional.pad(node_attr, (0,0,0,pad_to-len(node_attr)), value=-999)
        graph_list.append(Data(
                               x=node_attr.clone().detach().requires_grad_(True).to(device).float(),
                               edge_index=edge_index.clone().detach().requires_grad_(False).to(device),
                               label = label, requires_grad=True)
                               )
    return graph_list

def list_to_dataloader(data, attr={}):
    return cycle(DataLoader(data, **attr))

