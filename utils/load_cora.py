import numpy as np
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd
import os
from torch_geometric.utils import is_undirected, to_undirected, coalesce, add_self_loops, remove_self_loops


def get_cora_casestudy(root, train_ratio=0.6, val_ratio=0.2, seed=0):
    path = os.path.join(root, 'cora/cora_orig/cora')
    data_X, data_Y, data_citeid, data_edges = parse_cora(path)
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid(f"{root}/{data_name}", data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    num_nodes = data.num_nodes

    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)

    train_idx = indices[:num_train]
    val_idx = indices[num_train:num_train+num_val]
    test_idx = indices[num_train+num_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data, data_citeid

def parse_cora(path):
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_data_cora(root, train_ratio=0.6, val_ratio=0.2, seed=0):
    data, data_citeid = get_cora_casestudy(root, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    paper_path = os.path.join(root, 'cora/cora_orig/mccallum/cora/papers')
    with open(paper_path)as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    extraction_path = os.path.join(root, 'cora/cora_orig/mccallum/cora/extractions/')
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(extraction_path+fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti+'\n'+ab)
    
    num_nodes = data.y.shape[0]
    print(f"Num nodes: {num_nodes}")
    check = is_undirected(data.edge_index)
    if check:
        print("Graph is undirected")
        data.edge_index = coalesce(data.edge_index)
        pass
    else:
        print("Making graph undirected...")
        data.edge_index = to_undirected(data.edge_index)
        data.edge_index = coalesce(data.edge_index)
        if is_undirected(data.edge_index):
            pass
        else:
            raise ValueError("Graph is still directed after making it undirected")
    data.edge_index = remove_self_loops(data.edge_index)[0]
    data.edge_index = add_self_loops(data.edge_index, num_nodes=num_nodes)[0]
    
        
    return data, text