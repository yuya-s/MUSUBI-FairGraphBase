from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
import random
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import pandas as pd
import subprocess


DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']

    return [{k: v for k, v in zip(keys, line.split(', '))} for line in lines]

def params_count(encoder):
    return sum([p.numel() for p in encoder.parameters()])


def propagate(x, edge_index, edge_weight=None):
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    if(edge_weight == None):
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')

def propagate2(x, edge_index):
    edge_index, _ = add_remaining_self_loops(
        edge_index, num_nodes=x.size(0))

    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.allow_tf32 = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def random_drop_edges(adj, drop_prob):
    mask = torch.rand(adj.size()) > drop_prob
    adj = adj * mask
    adj = adj + adj.t() - adj * adj.t()
    return adj


def balance_samples(sens, labels):

    unique_sens_aa = np.unique(sens)
    unique_labels_aa = np.unique(labels)
    #print(f"sens len--------->>> {len(sens)}")
    #print(f"labels len--------->>> {len(labels)}")

    #print(f"sens--------->>> {unique_sens_aa}")
    #print(f"labels--------->>> {unique_labels_aa}")

    labels01 = labels[np.isin(labels, [0, 1])]
    sens01 = sens[np.isin(sens, [0, 1])]

    unique_sens = np.unique(sens01)
    unique_labels = np.unique(labels01)
    unique_pairs = [(s, l) for s in unique_sens for l in unique_labels]

    indices_by_pair = {}
    for pair in unique_pairs:
        s, l = pair
        indices = np.where((sens == s) & (labels == l))[0]
        indices_by_pair[pair] = indices

    min_samples = float('inf')
    for indices in indices_by_pair.values():
        num_samples = len(indices)
        if num_samples < min_samples:
            min_samples = num_samples

    balanced_indices = []
    for pair, indices in indices_by_pair.items():
        num_samples = len(indices)
        if num_samples > min_samples:
            selected_indices = np.random.choice(indices, min_samples, replace=False)
        else:
            selected_indices = np.random.choice(indices, min_samples, replace=True)
        #print(f"sample pair ---> {pair} : sample num ---> {num_samples}")
        balanced_indices.extend(selected_indices)

    #print(f"min sample num ---> {min_samples}")

    #print(f"min sample num × 4) ---> {len(balanced_indices)}")
    return np.array(balanced_indices)

