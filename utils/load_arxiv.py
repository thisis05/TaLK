from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import numpy as np
import random
import os
from torch_geometric.utils import is_undirected, to_undirected, coalesce, add_self_loops, remove_self_loops

def get_raw_data_arxiv(root, train_ratio=0.6, val_ratio=0.2, seed=0):

    download_path = os.path.join(root, 'ogbn_arxiv')
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', transform=T.ToSparseTensor(), root=download_path)
    data = dataset[0]

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

    adj_t = data.adj_t.to_symmetric()
    row, col, _ = adj_t.coo()
    data.edge_index = torch.stack([row, col], dim=0)

    id_path = os.path.join(root, 'ogbn_arxiv/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz')
    nodeidx2paperid = pd.read_csv(
        id_path, compression='gzip')

    text_path = os.path.join(root, 'ogbn_arxiv/titleabs.tsv.gz')
    raw_text = pd.read_csv(text_path,
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])
    nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(str)
    raw_text['paper id'] = raw_text['paper id'].astype(str)
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    
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