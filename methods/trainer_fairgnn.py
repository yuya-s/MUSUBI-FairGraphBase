import csv
import platform
import time

import torch
from torch import nn
from tqdm import tqdm

from models.model_fairgnn import FairGNN_ALL
from methods.trainer_utils import Early_stopper
from utils.utils import get_gpu_info, params_count


def run_trial_fairgnn(data, args, trial=1):
    features = data.x
    labels = data.y
    edge_index = data.edge_index
    train_mask = data.train_mask[trial-1]
    val_mask = data.val_mask[trial-1]
    test_mask = data.test_mask[trial-1]
    sens = data.sens
    idx_sens_train = data.train_mask[trial-1]

    device = args.device

    if idx_sens_train is None:
        idx_sens_train = train_mask
    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    sens = sens.to(device)
    idx_sens_train = idx_sens_train.to(device)

    model = FairGNN_ALL(
                    encoder=args.encoder,
                    nfeat=features.shape[1],
                    num_hidden = args.hidden,
                    gnn_layer_size=args.gnn_layer_size,
                    gnn_hidden=args.gnn_hidden,
                    cls_layer_size=args.cls_layer_size,
                    device=device,
                    data=data
                    )
    adv = nn.Linear(args.hidden, 1)

    model.to(device)
    adv.to(device)

    t_total = time.time()
    best_fair = 1000
    best_acc = 0

    x = features

    val_loss = 0
    early_stopping_count = 0

    best_val_tradeoff = -100

    params_num = params_count(model)
    early_stopper = Early_stopper(20, args.metrics, args.alpha, trial, params_num)

    with open(
            f'{args.output_dir}/train_time/train_time_FairGNN_{args.dataset}_{args.inprocessing}_{trial}.csv',
            'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train time"])

        G_params = model.get_gparams()

        optimizer_G = torch.optim.Adam(
            G_params, lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer_A = torch.optim.Adam(
            adv.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        for epoch in tqdm(range(args.epochs)):
            start_time = time.time()
            t = time.time()
            model.train()
            adv.train()
            optimize(
                args, model, adv,
                features, labels, train_mask, sens, idx_sens_train, edge_index,
                optimizer_G, optimizer_A, criterion
            )
            model.eval()

            _, _,output = model(edge_index, features)


            if early_stopper.check_stop(output, data):
                break

            end_time = time.time()
            train_time = end_time - start_time
            writer.writerow([epoch, train_time])


    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    all_metrics = early_stopper.get_all_metrics(data)
    return all_metrics, early_stopper.best_output

def optimize(args, model, adv, x, labels, idx_train, sens, idx_sens_train, edge_index,
             optimizer_G, optimizer_A, criterion):
    model.train()
    adv.train()

    adv.requires_grad_(False)
    optimizer_G.zero_grad()

    s, h, y = model(edge_index, x)
    s_g = adv(h)

    s_score = torch.sigmoid(s.detach())
    s_score[idx_sens_train] = sens[idx_sens_train].unsqueeze(1).float()
    y_score = torch.sigmoid(y)

    cov = torch.abs(
        torch.mean(
            (s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))
        )
    )

    cls_loss = criterion(
        y[idx_train], labels[idx_train].unsqueeze(1).float()
    )
    adv_loss = criterion(s_g, s_score)

    G_loss = (
            cls_loss + args.g_alpha * cov - args.g_beta * adv_loss
    )
    G_loss.backward()
    optimizer_G.step()

    adv.requires_grad_(True)
    optimizer_A.zero_grad()
    s_g = adv(h.detach())
    A_loss = criterion(s_g, s_score)
    A_loss.backward()
    optimizer_A.step()
