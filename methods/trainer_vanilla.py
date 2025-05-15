import csv
import platform
import time

from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from models.model_vanilla import Vanilla_GNN
from methods.trainer_utils import Early_stopper
from utils.utils import get_gpu_info, params_count


def run_trial_vanilla(data, args, trial=1):

    epochs = args.epochs
    device = args.device
    best_loss = 100
    early_stopping_count = 0
    best_val_tradeoff = -1

    labels = data.y
    edge_index  = data.edge_index
    features = data.x
    train_mask=data.train_mask[trial-1]
    val_mask = data.val_mask[trial-1]
    test_mask = data.test_mask[trial-1]
    sens = data.sens

    model = Vanilla_GNN(
        encoder=args.encoder,
        num_feature=features.shape[1],
        num_hidden=args.hidden,
        gnn_layer_size=args.gnn_layer_size,
        gnn_hidden=args.gnn_hidden,
        cls_layer_size=args.cls_layer_size,
        device=device,
        data=data
    )

    optimizer_2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_name = args.inprocessing

    params_num = params_count(model)
    early_stopper = Early_stopper(20, args.metrics, args.alpha, trial, params_num)

    with open(
            f'{args.output_dir}/train_time/train_time_VGCN_{model_name}_{trial}.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train time"])

        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            sim_loss = 0
            model.train()
            optimizer_2.zero_grad()
            edge_index_1 = edge_index
            x_1 = features

            c1 = model(x_1, edge_index_1)

            cl_loss = F.binary_cross_entropy_with_logits(c1[train_mask],
                                                         labels[train_mask].unsqueeze(1).float().to(
                                                             device))

            cl_loss.backward()
            optimizer_2.step()

            model.eval()
            c_val = model(features, edge_index)
            val_loss = F.binary_cross_entropy_with_logits(c_val[val_mask],
                                                          labels[val_mask].unsqueeze(1).float().to(
                                                              device))

            if epoch % 100 == 0:
                print(f"[Train] Epoch {epoch}: train_c_loss: {cl_loss:.4f} | val_c_loss: {val_loss:.4f}")

            if early_stopper.check_stop(c_val, data):
                break

            end_time = time.time()
            train_time = end_time - start_time
            writer.writerow([epoch, train_time])

    all_metrics = early_stopper.get_all_metrics(data)
    return all_metrics, early_stopper.best_output
