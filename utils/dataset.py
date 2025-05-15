import csv
import zipfile

import gdown
import pandas as pd
import os
import numpy as np
import random
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
import torch
import scipy.sparse as sp
import requests


def download(url: str, filename: str):
    r = requests.get(url)
    assert r.status_code == 200
    open(filename, "wb").write(r.content)


def make_train_val_test_indexes(args, runs, data_dir, sens, labels_np):
    train_masks = []
    val_masks = []
    test_masks = []
    idx_trains = []
    idx_vals = []
    idx_tests = []
    for trial in range(runs):
        #fname_idx = f'{data_dir}/label_idx_{args.preprocessing}_{trial}.pickle' "delete"
        fname_idx = f'{data_dir}/label_idx_{trial}.pickle'

        import pickle
        if os.path.exists(fname_idx):
            with open(fname_idx, "rb") as handle:
                label_idx = pickle.load(handle)
        else:
            #if args.preprocessing == "undersampling":
            #    label_idx = balance_samples(sens, labels_np)
            #else:
            label_idx_0 = np.where(labels_np == 0)[0]
            label_idx_1 = np.where(labels_np == 1)[0]
            label_idx = np.append(label_idx_0, label_idx_1)

            random.shuffle(label_idx)
            with open(fname_idx, "wb") as handle:
                pickle.dump(label_idx, handle)

        print("LENGTH OF label_idx: ", len(label_idx))

        split = [args.trainsize, args.trainsize+args.valsize]

        idx_train = label_idx[:int(split[0] * len(label_idx))]
        idx_val = label_idx[int(split[0] * len(label_idx)):int(split[1] * len(label_idx))]
        idx_test = label_idx[int(split[1] * len(label_idx)):]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        train_mask = index_to_mask(len(labels_np), torch.LongTensor(idx_train))
        val_mask = index_to_mask(len(labels_np), torch.LongTensor(idx_val))
        test_mask = index_to_mask(len(labels_np), torch.LongTensor(idx_test))

        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

        idx_trains.append(idx_train)
        idx_vals.append(idx_val)
        idx_tests.append(idx_test)

    return train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests

def mx_to_torch_sparse_tensor(sparse_mx, is_sparse=False, return_tensor_sparse=True):
    if not is_sparse:
        sparse_mx = sp.coo_matrix(sparse_mx)
    else:
        sparse_mx = sparse_mx.tocoo()
    if not return_tensor_sparse:
        return sparse_mx

    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1

    return mask


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    norm = 2 * (features - min_values).div(max_values - min_values) - 1
    norm_filled = torch.where(torch.isnan(norm), torch.tensor(0.0), norm)
    return norm_filled


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
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


def load_credit(args, dataset, runs):
    sens_attr = "Age"
    predict_attr = "NoDefaultNextMonth"
    path = "data/credit/"
    label_number = 1000

    credit_file = os.path.join(path, "credit.csv")
    if not os.path.exists(credit_file):
        url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/credit/credit.csv"
        download(url, credit_file)

    idx_features_labels = pd.read_csv(credit_file)
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels_np = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels_np.shape[0], labels_np.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels_np)

    sens = idx_features_labels[sens_attr].values.astype(int)

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests = make_train_val_test_indexes(args, runs, path, sens, labels_np)

    from collections import Counter
    print('predict_attr:',Counter(idx_features_labels[predict_attr]))
    print('sens_attr:',Counter(idx_features_labels[sens_attr]))

    sens = torch.FloatTensor(sens)

    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests


def load_bail(args, dataset, runs):
    sens_attr = "WHITE"
    predict_attr = "RECID"
    path = "data/bail/"
    label_number = 1000

    bail_file = os.path.join(path, "bail.csv")
    if not os.path.exists(bail_file):
        url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail.csv"
        download(url, bail_file)

    idx_features_labels = pd.read_csv(bail_file)
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels_np = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels_np.shape[0], labels_np.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels_np)
    sens = idx_features_labels[sens_attr].values.astype(int)

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests = make_train_val_test_indexes(args, runs, path, sens, labels_np)

    sens = torch.FloatTensor(sens)

    from collections import Counter
    print('predict_attr:', Counter(idx_features_labels[predict_attr]))
    print('sens_attr:', Counter(idx_features_labels[sens_attr]))

    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests


def load_german(args, dataset, runs):
    sens_attr = "Gender"
    predict_attr = "GoodCustomer"
    path = "data/german/"
    label_number = 1000

    german_file = os.path.join(path, "german.csv")
    if not os.path.exists(german_file):
        url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/german/german.csv"
        download(url, german_file)

    idx_features_labels = pd.read_csv(german_file)
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels_np = idx_features_labels[predict_attr].values
    labels_np[labels_np == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels_np.shape[0], labels_np.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels_np)

    sens = idx_features_labels[sens_attr].values.astype(int)

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests = make_train_val_test_indexes(args, runs, path, sens, labels_np)

    sens = torch.FloatTensor(sens)

    from collections import Counter
    print('predict_attr:', Counter(idx_features_labels[predict_attr]))
    print('sens_attr:', Counter(idx_features_labels[sens_attr]))
    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests

def load_pokec(args, dataset, runs=5):

    if dataset == 'pokec_z':
        filename = 'region_job_1'
    elif dataset == 'pokec_n':
        filename = 'region_job_2_2'

    data_dir = f"data/{dataset}/"
    features = np.load(f"{data_dir}/{filename}_features.npy")
    labels_np = np.load(f"{data_dir}/{filename}_labels.npy")
    edges = np.load(f"{data_dir}/{filename}_edges.npy")
    sens = np.load(f"{data_dir}/{filename}_sens.npy")

    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0,:], edges[1,:])),
                        shape=(labels_np.shape[0], labels_np.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels_np)

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests = make_train_val_test_indexes(args, runs, data_dir, sens, labels_np)

    sens_idx = set(np.where(sens >= 0)[0])
    sens = torch.FloatTensor(sens)

    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests



def load_pokec_large(args, dataset, runs=5):

    sens_attr = "region"
    predict_attr = "I_am_working_in_field"

    data_path = f"data/{dataset}"

    if dataset == 'pokec_n_large':
        url = "https://drive.google.com/u/0/uc?id=1wWm6hyCUjwnr0pWlC6OxZIj0H0ZSnGWs&export=download"
        destination = os.path.join(data_path, "pokec_n.zip")
        filename = 'region_job_2'
    else:
        url = "https://drive.google.com/u/0/uc?id=1FOYOIdFp6lI9LH5FJAzLhjFCMAxT6wb4&export=download"
        destination = os.path.join(data_path, "pokec_z.zip")
        filename = 'region_job'

    feature_file = os.path.join(data_path, f"{filename}.csv")
    edge_file = os.path.join(data_path, f"{filename}_relationship.txt")

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(feature_file):
        gdown.download(url, destination, quiet=False)
        with zipfile.ZipFile(destination, "r") as zip_ref:
            zip_ref.extractall(data_path)

    idx_features_labels = pd.read_csv(feature_file)
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels_np = idx_features_labels[predict_attr].values

    idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        edge_file,
        dtype=np.int64,
    )

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
    ).reshape(edges_unordered.shape)

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels_np.shape[0], labels_np.shape[0]),
        dtype=np.float32,
    )

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels_np)

    unique_labels = np.unique(labels)

    sens = idx_features_labels[sens_attr].values

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests = make_train_val_test_indexes(args, runs, data_path, sens, labels_np)

    sens = torch.FloatTensor(sens)

    features = torch.cat([features, sens.unsqueeze(-1)], -1)

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests




def load_google(args, dataset, runs=5):

    data_path = f"data/{dataset}"
    id = "111058843129764709244"
    feature_file = os.path.join(data_path, f"{id}.feat")
    featnames_file = os.path.join(data_path, f"{id}.featnames")
    edge_file = os.path.join(data_path, f"{id}.edges")

    sym=True

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(edge_file):
        url = f"https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/gplus/{id}.edges"
        download(url, edge_file)

    if not os.path.exists(feature_file):
        url = f"https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/gplus/{id}.feat"
        download(url, feature_file)

    if not os.path.exists(featnames_file):
        url = f"https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/gplus/{id}.featnames"
        download(url, featnames_file)

    edges_file = open(edge_file)
    edges = []
    for line in edges_file:
        edges.append([int(one) for one in line.strip("\n").split(" ")])

    feat_file = open(feature_file)
    feats = []
    for line in feat_file:
        feats.append([int(one) for one in line.strip("\n").split(" ")])

    feat_name_file = open(featnames_file)
    feat_name = []
    for line in feat_name_file:
        feat_name.append(line.strip("\n").split(" "))
    names = {}
    for name in feat_name:
        if name[1] not in names:
            names[name[1]] = name[1]

    feats = np.array(feats)

    node_mapping = {}
    for j in range(feats.shape[0]):
        node_mapping[feats[j][0]] = j

    feats = feats[:, 1:]

    feats = np.array(feats, dtype=float)

    sens = feats[:, 0]
    labels_np = feats[:, 164]

    feats = np.concatenate([feats[:, :164], feats[:, 165:]], -1)
    feats = feats[:, 1:]

    edges = np.array(edges)

    node_num = feats.shape[0]

    edges_adj = np.zeros([edges.shape[0], edges.shape[1]])

    for j in range(edges.shape[0]):
        edges_adj[j, 0] = node_mapping[edges[j][0]]
        edges_adj[j, 1] = node_mapping[edges[j][1]]

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges_adj[:, 0], edges_adj[:, 1])),
        shape=(labels_np.shape[0], labels_np.shape[0]),
        dtype=np.float32,
    )

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    idx_train = np.random.choice(
        list(range(node_num)), int(0.8 * node_num), replace=False
    )
    idx_val = list(set(list(range(node_num))) - set(idx_train))
    idx_test = np.random.choice(idx_val, len(idx_val) // 2, replace=False)
    idx_val = list(set(idx_val) - set(idx_test))

    features = torch.FloatTensor(feats)
    sens = torch.FloatTensor(sens)

    adj_norm = sys_normalized_adjacency(adj)

    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    labels = torch.LongTensor(labels_np)

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests = make_train_val_test_indexes(args, runs, data_path, sens, labels_np)

    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests


def load_aminer(args, dataset, runs=5):

    data_path = f"data/{dataset}"

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if dataset == 'aminer_l':
        feature_file = os.path.join(data_path, "raw_LCC/X_LCC.npz")
        labels_file = os.path.join(data_path, "raw_LCC/labels_LCC.txt")
        sens_file = os.path.join(data_path, "raw_LCC/sens_LCC.txt")
        edge_file = os.path.join(data_path, "raw_LCC/edgelist_LCC.txt")
        url = "https://drive.google.com/u/0/uc?id=1wYb0wP8XgWsAhGPt_o3fpMZDM-yIATFQ&export=download"
        destination = os.path.join(data_path, "raw_LCC.zip")
        if not os.path.exists(edge_file):
            gdown.download(url, destination, quiet=False)
            with zipfile.ZipFile(destination, "r") as zip_ref:
                zip_ref.extractall(data_path)
    elif dataset == 'aminer_s':
        feature_file = os.path.join(data_path, "X_Small.npz")
        labels_file = os.path.join(data_path, "labels_Small.txt")
        sens_file = os.path.join(data_path, "sens_Small.txt")
        edge_file = os.path.join(data_path, "edgelist_Small.txt")
        if not os.path.exists(edge_file):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/raw_Small/edgelist_Small.txt"
            download(url, edge_file)
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/raw_Small/labels_Small.txt"
            download(url, labels_file)
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/raw_Small/sens_Small.txt"
            download(url, sens_file)
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/raw_Small/X_Small.npz"
            download(url, feature_file)

    edgelist = csv.reader(open(edge_file))
    edges = []
    for line in edgelist:
        edge = line[0].split("\t")
        edges.append([int(one) for one in edge])

    edges = np.array(edges)

    labels_file = csv.reader(open(labels_file))
    labels = []
    for line in labels_file:
        labels.append(float(line[0].split("\t")[1]))
    labels = np.array(labels)

    sens_file = csv.reader(open(sens_file))
    sens = []
    for line in sens_file:
        sens.append([float(line[0].split("\t")[1])])
    sens = np.array(sens)


    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    unique_labels, counts = np.unique(labels, return_counts=True)
    most_common_label = unique_labels[np.argmax(counts)]
    labels = (labels == most_common_label).astype(int)
    labels_np = labels
    labels = torch.LongTensor(labels)

    sens = torch.FloatTensor(sens)

    features = np.load(feature_file)
    features = sp.coo_matrix(
            (features["data"], (features["row"], features["col"])),
            shape=(labels.shape[0], np.max(features["col"]) + 1),
            dtype=np.float32,
        ).todense()

    features = torch.FloatTensor(features)
    features = torch.cat([features, sens], -1)
    sens = sens.squeeze()

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests  = make_train_val_test_indexes(args, runs, data_path, sens, labels_np)

    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests


def load_wikidata(args, dataset, runs=5):

    data_dir = f"data/{dataset}/"

    features = np.load(f"{data_dir}/features.npy")
    features = sp.csr_matrix(features, dtype=np.float32)

    labels_np = np.load(f"{data_dir}/labels.npy")

    edges = np.load(f"{data_dir}/edges.npy", allow_pickle=True)

    sens = np.load(f"{data_dir}/sens.npy")

    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0,:], edges[1,:])),
                        shape=(labels_np.shape[0], labels_np.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))

    labels = torch.LongTensor(labels_np)

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests = make_train_val_test_indexes(args, runs, data_dir, sens, labels_np)

    sens = torch.FloatTensor(sens)

    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests


def load_dbpedia(args, dataset, runs=5):

    data_dir = f"data/{dataset}/"

    features = np.load(f"{data_dir}/features.npy")
    features = sp.csr_matrix(features, dtype=np.float32)

    labels_np = np.load(f"{data_dir}/labels.npy")

    edges = np.load(f"{data_dir}/edges.npy", allow_pickle=True)

    sens = np.load(f"{data_dir}/sens.npy")

    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0,:], edges[1,:])),
                        shape=(labels_np.shape[0], labels_np.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))

    labels = torch.LongTensor(labels_np)

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests = make_train_val_test_indexes(args, runs, data_dir, sens, labels_np)

    sens = torch.FloatTensor(sens)

    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests

def load_yago(args, dataset, runs=5):

    data_dir = f"data/{dataset}/"

    features = np.load(f"{data_dir}/features.npy")
    features = sp.csr_matrix(features, dtype=np.float32)

    labels_np = np.load(f"{data_dir}/labels.npy")

    edges = np.load(f"{data_dir}/edges.npy", allow_pickle=True)

    sens = np.load(f"{data_dir}/sens.npy")

    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0,:], edges[1,:])),
                        shape=(labels_np.shape[0], labels_np.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))

    labels = torch.LongTensor(labels_np)

    train_masks, val_masks, test_masks, idx_trains, idx_vals, idx_tests = make_train_val_test_indexes(args, runs, data_dir, sens, labels_np)

    sens = torch.FloatTensor(sens)

    return adj_norm_sp, edge_index, features, labels, train_masks, val_masks, test_masks, sens, adj, idx_trains, idx_vals, idx_tests



def get_dataset(dataname, runs, args):
    if(dataname == 'credit'):
        load = load_credit
    elif(dataname == 'bail'):
        load = load_bail
    elif(dataname == 'german'):
        load = load_german
    elif(dataname == 'pokec_z'):
        load = load_pokec
    elif(dataname == 'pokec_n'):
        load = load_pokec
    elif(dataname == 'pokec_z_large'):
        load = load_pokec_large
    elif(dataname == 'pokec_n_large'):
        load = load_pokec_large
    elif(dataname == 'google'):
        load = load_google
    elif (dataname == 'aminer_l'):
        load = load_aminer
    elif (dataname == 'aminer_s'):
        load = load_aminer
    elif(dataname.startswith('wikidata')):
        load = load_wikidata
    elif(dataname.startswith('dbpedia')):
        load = load_dbpedia
    elif (dataname.startswith('yago')):
        load = load_yago

    adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens, adj, idx_trains, idx_vals, idx_tests = load(
        args, dataset=dataname, runs=runs)

    if(dataname == 'credit'):
        sens_idx = 1
    elif(dataname == 'bail' or dataname == 'german'):
        sens_idx = 0
    elif(dataname == 'region_job' or dataname == 'region_job_2'):
        sens_idx = 3
    else:
        sens_idx = None

    x_max, x_min = torch.max(features, dim=0)[
        0], torch.min(features, dim=0)[0]

    if(dataname != 'german'):
        norm_features = feature_norm(features)
        features = norm_features

    return Data(adj=adj, x=features, edge_index=edge_index, adj_norm_sp=adj_norm_sp, y=labels.float(),
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, sens=sens, dataset=dataname
                    ), sens_idx, x_min, x_max

