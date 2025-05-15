from methods.bind_train import train_bind
from methods.bind_influence_compute import compute_influence_bind
from methods.bind_remove import remove_bind
import numpy as np
from torch_geometric.data import Data
import torch
from utils.dataset import index_to_mask,sys_normalized_adjacency,sparse_mx_to_torch_sparse_tensor
from torch_geometric.utils import from_scipy_sparse_matrix

def run_bind(data, args, trial=1):


    train_mask = data.train_mask[trial - 1]
    train_mask = train_mask.cpu().numpy()
    val_mask = data.val_mask[trial - 1]
    val_mask = val_mask.cpu().numpy()
    test_mask = data.test_mask[trial - 1]
    test_mask = test_mask.cpu().numpy()

    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    train_bind(trial-1, args, data.dataset, data.adj, data.x, data.y, idx_train, idx_val,
               idx_test, data.sens, need_norm_features=False)

    # (BIND) bind_influence_computate.py
    compute_influence_bind(trial-1, args, data.dataset, data.adj, data.x, data.y, idx_train, idx_val,
               idx_test, data.sens, need_norm_features=False)

    # (BIND) bind_remove.py
    newadj, newfeatures, newlabels, newidx_train, newidx_val, newidx_test, newsens \
        = remove_bind(trial-1, args, data.dataset, data.adj, data.x, data.y, idx_train, idx_val,
               idx_test, data.sens, need_norm_features=False, bind_del_rate=args.bind_del_rate)


    newtrain_mask = index_to_mask(len(newlabels), torch.LongTensor(newidx_train))
    newval_mask = index_to_mask(len(newlabels), torch.LongTensor(newidx_val))
    newtest_mask = index_to_mask(len(newlabels), torch.LongTensor(newidx_test))

    newadj_norm = sys_normalized_adjacency(newadj)
    newadj_norm_sp = sparse_mx_to_torch_sparse_tensor(newadj_norm)
    newedge_index, _ = from_scipy_sparse_matrix(newadj)


    data.adj = newadj
    data.x = newfeatures
    data.edge_index = newedge_index
    data.adj_norm_sp = newadj_norm_sp
    data.y = newlabels
    data.train_mask[trial - 1]=newtrain_mask
    data.val_mask[trial - 1]=newval_mask
    data.test_mask[trial - 1]=newtest_mask
    data.sens=newsens
