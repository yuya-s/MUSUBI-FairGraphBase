from numpy import *
from methods.bind_approximator import grad_z_graph, cal_influence_graph, s_test_graph_cost, cal_influence_graph_nodal
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance_matrix
import os
import networkx as nx
import time
import argparse
from torch_geometric.utils import convert
import warnings
warnings.filterwarnings('ignore')
import ctypes
torch.backends.cudnn.benchmark = True


def compute_influence_bind(trial, args, dataset_name, adj, features, labels, idx_train, idx_val, idx_test, sens, need_norm_features=False):

    adj_vanilla = adj
    features_vanilla = features
    labels_vanilla = labels
    idx_train_vanilla = idx_train
    idx_val_vanilla = idx_val
    idx_test_vanilla = idx_test
    sens_vanilla = sens

    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.bind_seed)
    torch.manual_seed(args.bind_seed)
    torch.cuda.manual_seed(args.bind_seed)

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

        return nei1_li, nei2_li, nei3_li

    def feature_norm(features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2*(features - min_values).div(max_values-min_values) - 1

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

    def del_adj(harmful):
        adj = adj_vanilla

        mask = np.ones(adj.shape[0], dtype=bool)

        mask[harmful] = False

        adj = sp.coo_matrix(adj.tocsr()[mask,:][:,mask])

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])

        return adj

    if need_norm_features:
        norm_features = feature_norm(features_vanilla)
        norm_features[:, 8] = features_vanilla[:, 8]
        features_vanilla = norm_features

    model = torch.load(f"data/{dataset_name}/bind_gcn_{trial}.pth")

    edge_index = convert.from_scipy_sparse_matrix(adj_vanilla)[0]
    #print("Pre-processing data...")
    computation_graph_involving = []
    the_adj = adj
    hop = 1
    G = nx.Graph(the_adj)
    for i in tqdm(range(idx_train_vanilla.shape[0])):
        neighbors = find123Nei(G, idx_train_vanilla[i].item())
        mid = []
        for j in range(hop):
            mid += neighbors[j]
        mid = list(set(mid).intersection(set(idx_train_vanilla.numpy().tolist())))
        computation_graph_involving.append(mid)
    #print("Pre-processing completed.")

    time1 = time.time()
    h_estimate_cost = s_test_graph_cost(args.device, edge_index, features_vanilla, idx_train_vanilla, idx_test_vanilla, labels_vanilla, sens_vanilla, model)

    gradients_list = grad_z_graph(args.device, edge_index, features_vanilla, idx_train_vanilla, labels_vanilla, model)

    influence, harmful, helpful, harmful_idx, helpful_idx = cal_influence_graph(idx_train_vanilla, h_estimate_cost, gradients_list)

    non_iid_influence = []

    for i in tqdm(range(idx_train_vanilla.shape[0])):

        if len(computation_graph_involving[i]) == 0:
            non_iid_influence.append(0)
            continue

        reference = list(range(adj_vanilla.shape[0]))
        for j in range(len(reference) - idx_train_vanilla[i]):
            reference[j + idx_train_vanilla[i]] -= 1

        mask = np.ones(idx_train_vanilla.shape[0], dtype=bool)
        mask[i] = False
        idx_train = idx_train_vanilla[mask]
        idx_val = idx_val_vanilla.clone()
        idx_test = idx_test_vanilla.clone()

        idx_train = torch.LongTensor(np.array(reference)[idx_train.numpy()])
        idx_val = torch.LongTensor(np.array(reference)[idx_val.numpy()])
        idx_test = torch.LongTensor(np.array(reference)[idx_test.numpy()])

        computation_graph_involving_copy = computation_graph_involving.copy()
        for j in range(len(computation_graph_involving_copy)):
            computation_graph_involving_copy[j] = np.array(reference)[computation_graph_involving_copy[j]]

        mask = np.ones(labels_vanilla.shape[0], dtype=bool)
        mask[idx_train_vanilla[i]] = False

        features = features_vanilla[mask, :]
        labels = labels_vanilla[mask]
        sens = sens_vanilla[mask]

        adj = del_adj(idx_train_vanilla[i])
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        h_estimate_cost_nodal = h_estimate_cost.copy()
        gradients_list_nodal = grad_z_graph(args.device, edge_index, features, torch.LongTensor(computation_graph_involving_copy[i]), labels, model)
        influence_nodal, _, _, _, _ = cal_influence_graph_nodal(idx_train, torch.LongTensor(computation_graph_involving_copy[i]), h_estimate_cost_nodal, gradients_list_nodal)

        non_iid_influence.append(sum(influence_nodal))

    final_influence = []
    for i in range(len(non_iid_influence)):
        ref = [idx_train_vanilla.numpy().tolist().index(item) for item in (computation_graph_involving[i] + [idx_train_vanilla[i]])]
        final_influence.append(non_iid_influence[i] - np.array(influence)[ref].sum())

    time4 = time.time()
    print("Average time per training node:", (time4 - time1)/1000, "s")
    np.save(f"data/{dataset_name}/bind_final_influence_{trial}.npy", np.array(final_influence))
