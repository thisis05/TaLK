import torch
from torch.nn import functional as F

def load_data(root, dataset, train_ratio=0.6, val_ratio=0.2, seed=0):
    if dataset == 'cora':
        from utils.load_cora import get_raw_data_cora as get_data
    elif dataset == 'arxiv':
        from utils.load_arxiv import get_raw_data_arxiv as get_data
    elif dataset == 'Computers':
        from utils.load_Computers import get_raw_data_Computers as get_data
    elif dataset == 'Photo':
        from utils.load_Photo import get_raw_data_Photo as get_data
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    data, text = get_data(root, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    return data, text


def load_full_data(root, name, train_ratio=0.6, val_ratio=0.2, sparse = False, seed=0):
    data, text = load_data(root=root, dataset=name, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    train_mask = data.train_mask
    val_mask   = data.val_mask
    test_mask  = data.test_mask
    num_nodes  = data.y.shape[0]
    edge_index = data.edge_index
    if sparse:
        E = torch.sparse_coo_tensor(
            indices=edge_index,  # (2, num_edges)
            values=torch.ones(edge_index.shape[1]),  
            size=(num_nodes, num_nodes)
        ).coalesce()
    else:
        E = edge_index
        
    y  = data.y
    if len(y.shape) > 1:
        y = y.squeeze(-1)
    num_classes = len(torch.unique(y))

    idx_train = torch.where(train_mask)[0]
    idx_val   = torch.where(val_mask)[0]
    idx_test  = torch.where(test_mask)[0]
    y_one_hot = F.one_hot(y, num_classes).float()

    return data, text, y, y_one_hot, E, idx_train, idx_val, idx_test
