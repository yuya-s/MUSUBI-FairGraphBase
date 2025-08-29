import csv
import platform
import time

import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_scatter import scatter_add
from tqdm import tqdm

from models.model_vanilla import Vanilla_GNN
from methods.trainer_utils import Early_stopper
from utils.utils import get_gpu_info, params_count

def run_trial_fairgb(data, args, trial=1):    
    neighbor_dist_list = get_ins_neighbor_dist(data.y.size(0), data.edge_index, args.device)

    data = data.to(args.device)
    orig_n = data.x.size(0)
    device = args.device
    features = data.x
    n_cls = data.y.max().int().item() + 1
    n_sen = data.sens.max().int().item() + 1
    index_list = torch.arange(len(data.y)).to(args.device)
    group_num_list, idx_info = [], []
    train_mask = data.train_mask[trial-1]
    val_mask   = data.val_mask[trial-1]
    for i in range(n_cls):
        for j in range(n_sen):
            mask = ((data.y == i) & (data.sens == j) & train_mask)
            data_num = mask.sum()
            group_num_list.append(int(data_num.item()))
            idx_info.append(index_list[mask])

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

    optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': args.e_lr, 'weight_decay': args.wd},
    {'params': model.c1.parameters(), 'lr': args.c_lr, 'weight_decay': args.wd},
])
    model_name = args.inprocessing

    params_num = params_count(model)
    early_stopper = Early_stopper(50, args.metrics, args.alpha, trial, params_num)

    with open(
            f'{args.output_dir}/train_time/train_time_FairGB_{model_name}_{trial}.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train time"])

        for epoch in range(0, args.epochs):
            start_time = time.time()
            model.train()
            optimizer.zero_grad()
            
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(group_num_list, idx_info, args.eta)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            lam = lam.to(args.device)
        
            if epoch >= args.warmup:
                new_edge_index = neighbor_sampling(data.x.size(0), data.edge_index, sampling_src_idx, neighbor_dist_list)
                new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)
                
                output = model(new_x, new_edge_index)

                add_num = output.shape[0] - train_mask.shape[0]
                new_train_mask = torch.ones(add_num, dtype=torch.bool, device=args.device)
                new_train_mask = torch.cat((torch.zeros(train_mask.shape[0], dtype=torch.bool, device=args.device), new_train_mask), dim=0)

                loss_src = F.binary_cross_entropy_with_logits(
                    output[new_train_mask], data.y[sampling_src_idx].unsqueeze(1).to(args.device), reduction='none')
                loss_dst = F.binary_cross_entropy_with_logits(
                    output[new_train_mask], data.y[sampling_dst_idx].unsqueeze(1).to(args.device), reduction='none')
                
                pos_grad_src = (1. - torch.exp(-loss_src).detach()) * lam
                pos_grad_dst = (1. - torch.exp(-loss_dst).detach()) * (1-lam)
                grad_count = []
                for i in range(n_cls):
                    for j in range(n_sen):
                        mask_src = (data.y[sampling_src_idx] == i) & (data.sens[sampling_src_idx] == j)
                        mask_dst = (data.y[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        grad_count.append(pos_grad_src[mask_src].sum().item() + pos_grad_dst[mask_dst].sum().item())

                min_grad = np.min(grad_count)
                group_weight_list = [float(min_grad)/(float(num) + 1e-6) for num in grad_count]

                for i in range(n_cls):
                    for j in range(n_sen):
                        mask_src = (data.y[sampling_src_idx] == i) & (data.sens[sampling_src_idx] == j)
                        mask_dst = (data.y[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        loss_src[mask_src] *= group_weight_list[i*2+j]
                        loss_dst[mask_dst] *= group_weight_list[i*2+j]

                loss = lam * loss_src + (1-lam) * loss_dst
                train_loss = loss.mean()
                train_loss.backward()
            else:
                output = model(data.x, data.edge_index)

                train_loss = F.binary_cross_entropy_with_logits(
                    output[train_mask], data.y[train_mask].unsqueeze(1).to(args.device))
                train_loss.backward()
            optimizer.step()

            model.eval()
            c_val = model(features, data.edge_index)
            val_loss = F.binary_cross_entropy_with_logits(c_val[val_mask], data.y[val_mask].unsqueeze(1).to(device))

            if epoch % 100 == 0:
                print(f"[Train] Epoch {epoch}: train_c_loss: {train_loss:.4f} | val_c_loss: {val_loss:.4f}")

            with torch.no_grad():
                h = model.encoder(data.x[:orig_n], data.edge_index)   
                clean_val_out = model.c1(h)
            if early_stopper.check_stop(clean_val_out, data):
                break
    all_metrics = early_stopper.get_all_metrics(data)

    import gc
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_metrics, early_stopper.best_output

@torch.no_grad()
def sampling_idx_individual_dst(group_num_list, idx_info, eta=0.5):
    n_cls, n_grp = 2, 2
    sampling_src_idx = torch.cat(idx_info)
    if np.random.rand() < eta:
        inter = True
    else:
        inter = False
    sampling_dst_idx = []
    for i in range(n_cls):
        for j in range(n_grp):
            if inter:
                target_group_id = 2 * (1 - i) + j
            else:
                target_group_id = 2 * i + (1 - j)
            prob = torch.ones(group_num_list[target_group_id]) / group_num_list[target_group_id]
            sampled_idx = torch.multinomial(prob, group_num_list[i * 2 + j], replacement=True)
            sampled_idx = idx_info[target_group_id][sampled_idx]
            sampling_dst_idx.append(sampled_idx)
    
    sampling_dst_idx = torch.cat(sampling_dst_idx)
    
    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]

    return sampling_src_idx, sampling_dst_idx


def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
    new_src = x[sampling_src_idx.to(x.device), :].clone()
    new_dst = x[sampling_dst_idx.to(x.device), :].clone()
    lam = lam.to(x.device)

    mixed_node = lam * new_src + (1-lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim =0)
    return new_x


@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx, neighbor_dist_list):
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index

@torch.no_grad()
def get_ins_neighbor_dist(num_nodes, edge_index, device):
        # edge_index: [2, E] (row=src, col=dst)
    row = edge_index[0].to(device)  # src(i)
    col = edge_index[1].to(device)  # dst(j)

    # j行 i列に1（= i→j のエッジ数）
    indices = torch.stack([col, row], dim=0)              # (2, E)
    values  = torch.ones(col.numel(), dtype=torch.float32, device=device)

    M = torch.sparse_coo_tensor(indices, values,
                                size=(num_nodes, num_nodes),
                                device=device).coalesce()

    # 行ごとにL1正規化（sumが0の行はそのまま0）
    r = M.indices()[0]                                    # 行インデックス（j）
    row_sum = torch.zeros(num_nodes, device=device).index_add_(0, r, M.values())
    norm_vals = M.values() / row_sum.clamp_min(1e-12)[r]

    M_norm = torch.sparse_coo_tensor(M.indices(), norm_vals,
                                     size=(num_nodes, num_nodes),
                                     device=device)
    return M_norm.to_dense()

    edge_index = edge_index.clone().to(device)
    row, col = edge_index[0], edge_index[1]

    # Compute neighbor distribution
    
    neighbor_dist_list = []
    for j in tqdm(range(num_nodes)):
        neighbor_dist = torch.zeros(num_nodes, dtype=torch.float32)
        idx = row[(col == j)]
        neighbor_dist[idx] += 1
        neighbor_dist_list.append(neighbor_dist)

    mat = torch.stack(neighbor_dist_list, dim=0)  # CPU
    mat = F.normalize(mat, dim=1, p=1)
    neighbor_dist_list = mat.to(device, non_blocking=True)
    return neighbor_dist_list

    #neighbor_dist_list = []
    #for j in tqdm(range(num_nodes)):
    #    neighbor_dist = torch.zeros(num_nodes, dtype=torch.float32).to(device)

    #    idx = row[(col==j)]
    #    neighbor_dist[idx] = neighbor_dist[idx] + 1
    #    neighbor_dist_list.append(neighbor_dist)

    #neighbor_dist_list = torch.stack(neighbor_dist_list,dim=0).to(device)
    #neighbor_dist_list = F.normalize(neighbor_dist_list,dim=1,p=1)

    #return neighbor_dist_list

