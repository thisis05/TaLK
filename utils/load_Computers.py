from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import numpy as np
import random
import os
from torch_geometric.utils import is_undirected, to_undirected, coalesce, add_self_loops, remove_self_loops

def get_raw_data_Computers(root, train_ratio=0.6, val_ratio=0.2,  seed=0):

    csv_path = os.path.join(root, 'Computers/Computers.csv')
    df = pd.read_csv(csv_path)
    text = list(df['text'])
    data = torch.load(os.path.join(root, 'Computers/Computers_pyg.pt'), weights_only=False)
    num_nodes = data['label'].shape[0]

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
    data.y = data['label']

    check = is_undirected(data.edge_index)
    if check:
        print("Graph is undirected")
        data.edge_index = coalesce(data.edge_index)
        pass
    else:
        print("Making graph undirected...")
        edge_index = to_undirected(data.edge_index)
        data.edge_index = coalesce(edge_index)
        if is_undirected(data.edge_index):
            pass
        else:
            raise ValueError("Graph is still directed after making it undirected")
    data.edge_index = remove_self_loops(data.edge_index)[0]
    data.edge_index = add_self_loops(data.edge_index, num_nodes=num_nodes)[0]
    return data, text