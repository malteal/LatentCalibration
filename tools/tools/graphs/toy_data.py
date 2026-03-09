"Construct toy data"
from tools import misc
import numpy as np
import random
from tqdm import tqdm
import torch
from torch_geometric.data import Data

def create_toy_graph(name, params):
    if name == "gauss":
        dist = misc.gaussian2d(**params)
    return dist

def toy_data(name, params, sample_size, node_dst=False):
    source_list = []
    for _ in tqdm(range(sample_size)):
        if node_dst:
            assert isinstance(node_dst, list), "node_dst should be a list of [min,max] nodes"
            set_to_zero = np.random.randint(node_dst[0], node_dst[1]+1)
        source_nodes = create_toy_graph(name, params)

        if node_dst:
            source_nodes[set_to_zero:] = -999
        source_list.append(source_nodes)

    source_list = np.array(source_list)
    return source_list

def get_toy_shape(type_nr, size, x_shift=0, y_shift=0):
    "Possible shape where we know the optimal transport"
    if size == 3:
        # graphs with 3 nodes
        if type_nr==0:
            shape = np.array([[1,0], [-1,0], [0,1]])
        elif type_nr==1:
            shape = np.array([[1,0], [-1,0], [-1.0,-1.0]])
        elif type_nr==2:
            shape = np.array([[1,0], [-1,0], [1.0,1.0]])
    elif size == 4:
        # graphs with 4 nodes
        if type_nr==0:
            shape = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        elif type_nr==1:
            shape = np.array([[1,0], [-1,0], [-1.0,-1.0], [1.0,1.0]])
        elif type_nr==2:
            shape = np.array([[1,0], [-1,0], [1.0,1.0], [1.0,-1.0]])
        elif type_nr==3:
            shape = np.array([[2,0], [-2,0], [0.0,1.0], [0.0,-1.0]])
        elif type_nr==4:
            shape = np.array([[1,0], [-1,0], [0.0,2.0], [0.0,-2.0]])
    shape += np.zeros_like(shape)+np.array([x_shift,y_shift])
    return shape

def circle_points(r, n, drop=False, shift=0, drop_number=1, max_nodes:int=0):
    "Creating circles and possible dropping some"
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    circles = circles[0]+shift
    lst = list(np.arange(0,n))
    
    if drop and not isinstance(drop, str):
        # drop non neighbouring
        remove_values = []
        for i in range(drop_number):
            remove_values.append(random.sample(lst,1))
            for k in np.arange(-1,2):
                # try:
                if (remove_values[i][0] == n-1) and (0 in lst):
                    lst.remove([0])
                if (remove_values[i][0] == 0) and (n-1 in lst):
                    lst.remove([n-1])
                if remove_values[i]+k in lst:
                    lst.remove([remove_values[i]+k])
        if (np.abs(remove_values[0][0]-remove_values[1][0]) == 1 or
            np.abs(remove_values[0][0]-remove_values[1][0]) == n-1):
            print(remove_values)
            print("neighbouring - not correct!")
        circles = np.delete(circles, remove_values, axis=0)
    elif drop=="neighbors":
        # drop neighbouring circles
        random_value = np.random.randint(1, len(circles)-1)
        other_random_value = random_value+(1 if np.random.randint(0,2) else -1)
        circles = np.delete(circles, [random_value, other_random_value], axis=0)
    if (max_nodes != 0):
        circles = np.r_[circles, np.ones((max_nodes-circles.shape[0], circles.shape[1]))*-999]
    return circles

def main_test():
    params = {"mean": (3,3), "cov": [[1,0],[0,1]], "graph_size":1}
    output = create_toy_graph("gauss", params)
    assert (not isinstance(output, list)) & len(output[0]) != 2, "gaussian2d not outputting the correct values"

if __name__ == "__main__":
    main_test()