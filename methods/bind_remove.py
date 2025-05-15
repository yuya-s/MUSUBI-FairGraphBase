from __future__ import division
from __future__ import print_function

from numpy import *
import scipy.sparse as sp
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance_matrix
import os
import networkx as nx
from torch_geometric.utils import convert
import time
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')
import ctypes
from utils.utils import balance_samples


def remove_bind(trial, args, dataset_name, adj, features, labels, idx_train, idx_val, idx_test, sens,
               need_norm_features=False, bind_del_rate=1):

    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.bind_seed)
    torch.manual_seed(args.bind_seed)
    torch.cuda.manual_seed(args.bind_seed)
    open_factor = args.bind_helpfulness_collection
    print(open_factor)

    def accuracy_new(output, labels):
        correct = output.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def build_relationship(x, thresh=0.25):
        df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
        df_euclid = df_euclid.to_numpy()
        idx_map = []
        for ind in range(df_euclid.shape[0]):
            max_sim = np.sort(df_euclid[ind, :])[-2]
            neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
            import random
            random.seed(912)
            random.shuffle(neig_id)
            for neig in neig_id:
                if neig != ind:
                    idx_map.append([ind, neig])
        idx_map = np.array(idx_map)

        return idx_map

    def normalize(mx):
        rowsum = np.array(mx.sum(1))

        r_inv = np.power(rowsum, -1).flatten()

        r_inv[np.isinf(r_inv)] = 0.

        r_mat_inv = sp.diags(r_inv)

        mx = r_mat_inv.dot(mx)

        return mx

    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)

        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))

        values = torch.from_numpy(sparse_mx.data)

        shape = torch.Size(sparse_mx.shape)

        return torch.sparse.FloatTensor(indices, values, shape)

    def del_adj(harmful, adj):

        mask = np.ones(adj.shape[0], dtype=bool)
        mask[harmful] = False
        adj = sp.coo_matrix(adj.tocsr()[mask, :][:, mask])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        return adj

    def find123Nei(G, node):
        nodes = list(nx.nodes(G))
        nei1_li = []
        nei2_li = []
        nei3_li = []
        for FNs in list(nx.neighbors(G, node)):
            nei1_li .append(FNs)

        for n1 in nei1_li:
            for SNs in list(nx.neighbors(G, n1)):
                nei2_li.append(SNs)
        nei2_li = list(set(nei2_li) - set(nei1_li))
        if node in nei2_li:
            nei2_li.remove(node)

        for n2 in nei2_li:
            for TNs in nx.neighbors(G, n2):
                nei3_li.append(TNs)
        nei3_li = list(set(nei3_li) - set(nei2_li) - set(nei1_li))
        if node in nei3_li:
            nei3_li.remove(node)

        return [nei1_li, nei2_li, nei3_li]

    def feature_norm(features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2*(features - min_values).div(max_values-min_values) - 1

    if need_norm_features:
        norm_features = feature_norm(features)
        norm_features[:, 0] = features[:, 0]
        features = feature_norm(features)

    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    computation_graph_involving = []
    the_adj = adj
    hop = 1
    print("Finding neighbors ... ")

    G = nx.Graph(the_adj)

    for i in tqdm(range(idx_train.shape[0])):
        neighbors = find123Nei(G, idx_train[i].item())
        mid = []
        for j in range(hop):
            mid += neighbors[j]
        mid = list(set(mid).intersection(set(idx_train.numpy().tolist()))) + [idx_train[i].item()]
        computation_graph_involving.append(mid)

    final_influence = np.load(f"data/{dataset_name}/bind_final_influence_{trial}.npy", allow_pickle=True)

    helpful = idx_train[np.argsort(final_influence).copy()].tolist()
    helpful_idx = np.argsort(final_influence).copy().tolist()
    harmful_idx = helpful_idx[::-1]
    harmful = idx_train[harmful_idx].tolist()

    if open_factor:
        harmful = helpful
        harmful_idx = helpful_idx

    total_neighbors = []
    masker = np.ones(len(harmful), dtype=bool)

    for i in range(len(harmful) - 1):
        if masker[i] == True:
            total_neighbors += computation_graph_involving[harmful_idx[i]]
        if list(set(total_neighbors).intersection(set(computation_graph_involving[harmful_idx[i + 1]]))) != []:
            masker[i+1] = False

    harmful_idx = np.array(harmful_idx)[masker].tolist()
    harmful = idx_train[harmful_idx].tolist()

    max_num = 0
    for i in range(len(final_influence[harmful_idx]) - 1):
        if final_influence[harmful_idx][i] * final_influence[harmful_idx][i+1] <= 0:
            print("At most effective number:")
            print(i + 1)
            max_num = i + 1
            break

    def fair_metric(pred, labels, sens):
        idx_s0 = sens==0
        idx_s1 = sens==1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
        parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))

        return parity.item(), equality.item()


    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index)
        loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        preds = (output.squeeze() > 0).type_as(labels)
        acc_train = accuracy_new(preds[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.bind_fastmode:
            model.eval()
            output = model(features, edge_index)

        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
        acc_val = accuracy_new(preds[idx_val], labels[idx_val])

        return loss_val.item()

    def tst():
        model.eval()
        output = model(features, edge_index)
        preds = (output.squeeze() > 0).type_as(labels)
        loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
        acc_test = accuracy_new(preds[idx_test], labels[idx_test])

        idx_sens_test = sens[idx_test]
        idx_output_test = output[idx_test]
        fair_cost_records.append(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))

        auc_roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
        f1_test = f1_score(labels[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
        parity, equality = fair_metric(preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                       sens[idx_test].numpy())

        sp_records.append(parity)
        eo_records.append(equality)
        acc_records.append(acc_test.item())
        auc_records.append(auc_roc_test)
        f1_records.append(f1_test)

    influence_approximation = []
    fair_cost_records = []
    sp_records = []
    eo_records = []
    acc_records = []
    auc_records = []
    f1_records = []

    batch_size = 1
    percetage_budget = float(bind_del_rate) / 100

    adj, features, labels, idx_train, idx_val, idx_test, sens = adj, features.clone(), labels.clone(), idx_train.clone(), idx_val.clone(), idx_test.clone(), sens.clone()

    k = int(percetage_budget * max_num)

    harmful_flags = harmful[:k]

    influence_approximation.append(sum(final_influence[harmful_idx[:k]]))

    harmful_idx_flags = harmful_idx[:k]
    mask = np.ones(idx_train.shape[0], dtype=bool)
    mask[harmful_idx_flags] = False
    idx_train = idx_train[mask]
    idx_val = idx_val.clone()
    idx_test = idx_test.clone()

    reference = list(range(adj.shape[0]))
    for i in range(len(harmful_flags)):
        for j in range(len(reference) - harmful_flags[i]):
            reference[j + harmful_flags[i]] -= 1

    idx_train = torch.LongTensor(np.array(reference)[idx_train.numpy()])
    idx_val = torch.LongTensor(np.array(reference)[idx_val.numpy()])
    idx_test = torch.LongTensor(np.array(reference)[idx_test.numpy()])

    mask = np.ones(labels.shape[0], dtype=bool)
    mask[harmful_flags] = False
    features = features[mask, :]
    labels = labels[mask]
    sens = sens[mask]

    adj = del_adj(harmful_flags, adj)

    edge_index = convert.from_scipy_sparse_matrix(adj)[0]


    return adj, features, labels, idx_train, idx_val, idx_test, sens
