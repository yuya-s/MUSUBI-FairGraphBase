from __future__ import division
from __future__ import print_function
from torch_geometric.utils import convert
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import ctypes


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x


def train_bind(trial, args, dataset_name, adj, features, labels, idx_train, idx_val, idx_test, sens, need_norm_features=False):

    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.bind_seed)
    torch.manual_seed(args.bind_seed)
    if args.device:
        torch.cuda.manual_seed(args.bind_seed)

    def feature_norm(features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2*(features - min_values).div(max_values-min_values) - 1

    if need_norm_features:
        norm_features = feature_norm(features)
        norm_features[:, 0] = features[:, 0]
        features = feature_norm(features)

    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    model = GCN(nfeat=features.shape[1], nhid=args.bind_hidden, nclass=1, dropout=args.bind_dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.bind_lr, weight_decay=args.bind_weight_decay)

    if args.device:
        model.to(args.device)
        features = features.to(args.device)
        edge_index = edge_index.to(args.device)
        labels = labels.to(args.device)
        idx_train = idx_train.to(args.device)
        idx_val = idx_val.to(args.device)
        idx_test = idx_test.to(args.device)
        sens = sens.to(args.device)

    def accuracy_new(output, labels):
        correct = output.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

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

        if not args.bind_fastmode: # val mode
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

        print("*****************  Cost  ********************")
        print("SP cost:")
        idx_sens_test = sens[idx_test]
        idx_output_test = output[idx_test]
        print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))

        print("EO cost:")
        idx_sens_test = sens[idx_test][labels[idx_test]==1]
        idx_output_test = output[idx_test][labels[idx_test]==1]
        print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))
        print("**********************************************")

        parity, equality = fair_metric(preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                       sens[idx_test].numpy())

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        print("Statistical Parity:  " + str(parity))
        print("Equality:  " + str(equality))


    t_total = time.time()
    final_epochs = 0
    loss_val_global = 1e10

    starting = time.time()
    for epoch in tqdm(range(args.bind_epochs)):
        loss_mid = train(epoch)
        if loss_mid < loss_val_global:
            loss_val_global = loss_mid
            torch.save(model, f"data/{dataset_name}/bind_gcn_{trial}.pth")
            final_epochs = epoch


    ending = time.time()
    print("Time:", ending - starting, "s")

