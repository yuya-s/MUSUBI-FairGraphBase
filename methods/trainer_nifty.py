import csv
import platform
import time

import torch
from torch import optim
from torch_geometric.utils import dropout_edge
from tqdm import tqdm
import torch.nn.functional as F
#from triton.ops.matmul_perf_model import early_config_prune

from models.model_nifty import NIFTY_GAT
from methods.trainer_utils import Early_stopper
from utils.utils import get_gpu_info, params_count


def run_trial_nifty(data, args, trial=1):

    device = args.device
    epochs = args.epochs

    t_total = time.time()
    best_loss = 100
    best_acc = 0
    early_stopping_count = 0
    best_val_tradeoff = -1

    edge_index = data.edge_index
    features = data.x
    labels = data.y

    train_mask=data.train_mask[trial-1]
    val_mask = data.val_mask[trial-1]
    test_mask = data.val_mask[trial - 1]

    sim_coeff = args.sim_coeff

    drop_edge_rate_1 = drop_edge_rate_2 = 0
    drop_feature_rate_1 = drop_feature_rate_2 = 0

    val_edge_index_1 = dropout_edge(edge_index.to(device), p=drop_edge_rate_1)[0]
    val_edge_index_2 = dropout_edge(edge_index.to(device), p=drop_edge_rate_2)[0]
    sens_idx = -1
    val_x_1 = drop_feature(features.to(device), drop_feature_rate_1, sens_idx, sens_flag=False)
    val_x_2 = drop_feature(features.to(device), drop_feature_rate_2, sens_idx)

    model = NIFTY_GAT(
                    seed = int(args.seed + trial),
                    num_features=features.shape[1],
                    num_hidden = args.hidden,
                    num_proj_hidden = args.num_proj_hidden,
                    sim_coeff = args.sim_coeff,
                    encoder=args.encoder,
                    gnn_layer_size=args.gnn_layer_size,
                    gnn_hidden=args.gnn_hidden,
                    cls_layer_size=args.cls_layer_size,
                    device=device,
                    data=data
                    )

    par_1 = model.parameters_1()
    par_2 = model.parameters_2()
    optimizer_1 = optim.Adam(par_1, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_2 = optim.Adam(par_2, lr=args.lr, weight_decay=args.weight_decay)

    params_num = params_count(model)
    early_stopper = Early_stopper(20, args.metrics, args.alpha, trial, params_num)

    with open(f'{args.output_dir}/train_time/train_time_NIFTY_{args.inprocessing}_{trial}.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train time"])

        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            t = time.time()

            sim_loss = 0
            cl_loss = 0
            rep = 1
            for _ in range(rep):
                model.train()
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate_1)[0]
                edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate_2)[0]
                x_1 = drop_feature(features, drop_feature_rate_1, sens_idx, sens_flag=False)
                x_2 = drop_feature(features, drop_feature_rate_2, sens_idx)
                z1 = model.forward(x_1, edge_index_1)
                z2 = model.forward(x_2, edge_index_2)

                p1 = model.projection(z1)
                p2 = model.projection(z2)

                h1 = model.prediction(p1)
                h2 = model.prediction(p2)

                l1 = model.D(h1[train_mask], p2[train_mask]) / 2
                l2 = model.D(h2[train_mask], p1[train_mask]) / 2
                sim_loss += sim_coeff * (l1 + l2)

            (sim_loss / rep).backward()
            optimizer_1.step()

            z1 = model.forward(x_1, edge_index_1)
            z2 = model.forward(x_2, edge_index_2)
            c1 = model.classifier(z1)
            c2 = model.classifier(z2)

            l3 = F.binary_cross_entropy_with_logits(c1[train_mask],
                                                    labels[train_mask].unsqueeze(1).float().to(device)) / 2
            l4 = F.binary_cross_entropy_with_logits(c2[train_mask],
                                                    labels[train_mask].unsqueeze(1).float().to(device)) / 2

            cl_loss = (1 - sim_coeff) * (l3 + l4)
            cl_loss.backward()
            optimizer_2.step()
            loss = (sim_loss / rep + cl_loss)

            model.eval()
            emb = model.forward(val_x_1, val_edge_index_1)
            output = model.classifier(emb)

            if early_stopper.check_stop(output, data):
                break

            end_time = time.time()
            train_time = end_time - start_time
            writer.writerow([epoch, train_time])

    all_metrics = early_stopper.get_all_metrics(data)
    return all_metrics, early_stopper.best_output


def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    if sens_flag:
        x[:, sens_idx] = 1-x[:, sens_idx]

    return x

