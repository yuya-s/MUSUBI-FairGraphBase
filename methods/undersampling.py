import numpy as np
from utils.dataset import index_to_mask,sys_normalized_adjacency,sparse_mx_to_torch_sparse_tensor
import torch

def run_undersampling(data, args, trial=1):
    train_mask = data.train_mask[trial - 1]
    train_mask = train_mask.cpu().numpy()

    train_indices = np.where(train_mask)[0]

    train_labels = data.y[train_indices]
    train_sens = data.sens[train_indices]
    train_labels =train_labels.cpu().numpy()
    train_sens = train_sens.cpu().numpy()

    unique_pairs = [(s, l) for s in np.unique(train_sens) for l in np.unique(train_labels)]

    indices_by_pair = {
        pair: train_indices[(train_sens == pair[0]) & (train_labels == pair[1])]
        for pair in unique_pairs
    }

    min_samples = min(len(idx) for idx in indices_by_pair.values())

    balanced_indices = []
    for pair, indices in indices_by_pair.items():
        selected = indices[:min_samples]
        balanced_indices.extend(selected)

    new_train_mask = index_to_mask(len(data.y), torch.LongTensor(balanced_indices))

    data.train_mask[trial - 1]=new_train_mask
