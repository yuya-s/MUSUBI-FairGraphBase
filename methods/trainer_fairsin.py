import csv
import math
import platform
import time
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
import scipy.sparse as sp

from models.model import Encoder, Classifier
from models.model_fairsin import FairSIN_MLP, MLP_discriminator
from methods.trainer_utils import Early_stopper
from utils.utils import get_gpu_info, seed_everything, params_count
import torch.nn.functional as F

def run_trial_fairsin(data, args, trial=1):
    seed_everything(trial + args.seed)

    train_mask=data.train_mask[trial-1]
    val_mask=data.val_mask[trial-1]
    test_mask=data.test_mask[trial-1]

    data.adj = data.adj - sp.eye(data.adj.shape[0])
    print('sample begin.')
    size = data.adj.shape[0]
    rows = []
    cols = []
    values = []

    for i in tqdm(range(data.adj.shape[0])):
        neighbor = torch.tensor(data.adj[i].nonzero()).to(args.device)
        mask = (data.sens[neighbor[1]] != data.sens[i])
        h_nei_idx = neighbor[1][mask]
        idxes = h_nei_idx.to('cpu').detach().numpy().copy()
        for idx in idxes:
            rows.append(i)
            cols.append(idx)
            values.append(1)
    indices = torch.tensor([rows, cols])
    values = torch.tensor(values)
    new_adj = torch.sparse_coo_tensor(indices, values, size=[size, size])
    new_adj = new_adj.coalesce()
    c_X = data.x
    new_adj = new_adj.cpu()
    row_sums = torch.zeros(new_adj.shape[0], dtype=torch.long)
    indices = new_adj.indices()
    for i, v in zip(new_adj.indices()[0], new_adj.values()):
        row_sums[i] += v
    deg = torch.tensor(row_sums, dtype=torch.float).cpu()

    mat = new_adj.to(torch.float)
    hh = torch.spmm(mat, (data.x).cpu())
    h_X = hh / deg.unsqueeze(-1)

    mask = torch.any(torch.isnan(h_X), dim=1)
    h_X = h_X[~mask].to(args.device)
    c_X = c_X[~mask].to(args.device)
    print('node avg degree:', data.edge_index.shape[1] / data.adj.shape[0], ' heteroneighbor degree mean:',
          deg.float().mean(), ' node without heteroneghbor:', (deg == 0).sum())

    from sklearn.model_selection import train_test_split

    indices = np.arange(c_X.shape[0])
    [indices_train, indices_test, y_train, y_test] = train_test_split(indices, indices, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = c_X[indices_train], c_X[indices_test], h_X[indices_train], h_X[indices_test]


#---------------
    criterion = nn.BCELoss()
    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

    classifier = Classifier(args.hidden, 1, args.cls_layer_size, False).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    encoder = Encoder(args.encoder, args.num_features, args.hidden,
                      hidden_size=args.gnn_hidden, num_layers=args.gnn_layer_size, data=data).to(args.device)
    optimizer_e = torch.optim.Adam([
        dict(params=encoder.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)

    model = FairSIN_MLP(len(data.x[0]),args.hidden,len(data.x[0])).to(args.device)
    optimizer = torch.optim.Adam([
            dict(params=model.parameters(), weight_decay=0.001)], lr=args.m_lr)


    discriminator.reset_parameters()
    classifier.reset_parameters()
    encoder.reset_parameters()
    model.reset_parameters()

    best_val_tradeoff = -1
    best_val_loss = math.inf
    best_eval_labels = None
    best_eval_pred = None
    best_eval_sens = None

    params_num_encoder = params_count(encoder)
    params_num_classifier = params_count(classifier)

    early_stopper = Early_stopper(20, args.metrics, args.alpha, trial, params_num_encoder + params_num_classifier)


    with open(f'{args.output_dir}/train_time/train_time_FairSIN_{args.dataset}_{args.inprocessing}_{trial}.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train time"])

        for epoch in tqdm(range(0, args.epochs)):
            start_time = time.time()
            for m_epoch in range(0, args.m_epoch):
                model.train()
                optimizer.zero_grad()
                output = model(X_train)
                train_loss = torch.nn.functional.mse_loss(output, y_train)
                train_loss.backward()
                optimizer.step()

                model.eval()
                output = model(X_test)
                valid_loss = torch.nn.functional.mse_loss(output, y_test)

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_mlp_state = model.state_dict()
            model.load_state_dict(best_mlp_state)
            model.eval()

            classifier.train()
            encoder.train()
            for epoch_c in range(0, args.c_epochs):
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()

                h = encoder(data.x + args.delta * model(data.x), data.edge_index)
                output = classifier(h)

                loss_c = F.binary_cross_entropy_with_logits(
                    output[train_mask], data.y[train_mask].unsqueeze(1).to(args.device).float())

                loss_c.backward()

                optimizer_e.step()
                optimizer_c.step()

            if args.d == 'yes':
                discriminator.train()
                encoder.train()
                for epoch_d in range(0, args.d_epochs):
                    optimizer_d.zero_grad()
                    optimizer_e.zero_grad()
                    optimizer.zero_grad()

                    h = encoder(data.x + args.delta * model(data.x), data.edge_index)
                    output = discriminator(h)

                    loss_d = criterion(output.view(-1),
                                       data.sens)

                    loss_d.backward()
                    optimizer_d.step()
                    optimizer_e.step()
                    optimizer.step()

            classifier.eval()
            encoder.eval()
            with torch.no_grad():
                h = encoder(data.x, data.edge_index)
                output = classifier(h)

            if early_stopper.check_stop(output, data):
                break

            end_time = time.time()
            train_time = end_time - start_time
            writer.writerow([epoch, train_time])

    all_metrics = early_stopper.get_all_metrics(data)
    return all_metrics, early_stopper.best_output
