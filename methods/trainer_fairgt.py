import csv
import platform
import time

from torch import optim
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from models.model_fairgt import FairGT
from methods.trainer_utils import Early_stopper
from utils.utils import params_count


def run_trial_fairgt(data, args, trial=1):
        
    epochs = args.epochs
    device = args.device

    labels = data.y
    features = data.x
    features = features.to(device)
    # counter_features = args.counter_features
    # counter_features = counter_features.to(device)     
    adj_no_self_loop = data.adj - sp.eye(data.adj.shape[0])    
    sens = data.sens          
    train_mask=data.train_mask[trial-1]
    val_mask = data.val_mask[trial-1]    
    _, eignvector = compute_G_laplacian_eigen(adj_no_self_loop, args.pe_dim)
    eignvector = eignvector.to(device) 

    #if((args.dataset == 'pokec_n') or (args.dataset == 'pokec_z')): #元のコード通りの分岐. 恐らくグラフサイズによって分けている.
    new_adj = create_adj_with_same_sens_small_cliques(sens, args.cliques, device)     
    #else:
    #    new_adj = create_adj_with_same_sens_cliques(sens, device)         

    
    

    feature_and_eign = torch.cat((features, eignvector), dim=1) 
    processed_input = create_fair_input(new_adj, feature_and_eign, args.hops)
    processed_input = processed_input.to(device)
    args.in_dim = processed_input.shape[-1]
    
    # counter_features_and_eign = torch.cat((counter_features, eignvector), dim=1)
    # processed_counter_input = create_fair_input(new_adj, counter_features_and_eign, args.hops)
    # processed_counter_input = processed_counter_input.to(device)            
    model = FairGT(vars(args)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)   
    params_num = params_count(model)
    early_stopper = Early_stopper(20, args.metrics, args.alpha, trial, params_num)  
      
    model_name = args.inprocessing    
         
    with open(
            f'{args.output_dir}/train_time/train_time_{model_name}_{trial}.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train time"])   
   
        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            model.train()
            optimizer.zero_grad()
            cl = model(processed_input)        
            cl_loss = F.binary_cross_entropy_with_logits(cl[train_mask], labels[train_mask].unsqueeze(1).float().to(device))
            cl_loss.backward()
            optimizer.step()   

            model.eval()
            c_val = model(processed_input)                                 
            cl_loss_val = F.binary_cross_entropy_with_logits(c_val[val_mask], labels[val_mask].unsqueeze(1).float().to(device))                                      
            # counter_c_val = model(processed_counter_input)                           

            # if early_stopper.check_stop(c_val, counter_c_val, data, cl_loss_val):
            #     break

            if early_stopper.check_stop(c_val, data):
                break

            end_time = time.time()
            train_time = end_time - start_time
            writer.writerow([epoch, train_time])


    all_metrics = early_stopper.get_all_metrics(data)
    return all_metrics, early_stopper.best_output    
    
    
        
def create_adj_with_same_sens_cliques(sens, device):

    node_number = sens.shape[0]
    srcs, dsts = [], []

    for key in torch.unique(sens):
        node_indices = (sens == key).nonzero().squeeze()
        repeat_num = len(node_indices)
    
        src = node_indices.repeat_interleave(repeat_num)
        dst = node_indices.repeat(repeat_num)
        srcs.append(src)
        dsts.append(dst)

    src_ = torch.cat(srcs).long()
    dst_ = torch.cat(dsts).long()
    num_nodes = node_number
    edge_index = torch.stack([src_, dst_], dim=0).to(device)
    values = torch.ones(edge_index.shape[1], dtype=torch.float32).to(device)

    adj_sparse = torch.sparse_coo_tensor(
        indices=edge_index,
        values=values,
        size=(num_nodes, num_nodes),
        dtype=torch.float32
    )

    new_adj_dense = adj_sparse.to_dense()

    diag_elements = torch.diag(new_adj_dense)
    diag_matrix = torch.diag_embed(diag_elements)
    new_adj_dense -= diag_matrix

    return new_adj_dense

def create_random_cliques(id_list, subnum):
    shuffled_ids = id_list[torch.randperm(len(id_list))]
    sub_sequences = torch.split(shuffled_ids, subnum)
    srcs, dsts=[],[]
    
    for idx, sub_seq in enumerate(sub_sequences):
        repeat_num = len(sub_seq)
        src = sub_seq.repeat_interleave(repeat_num)
        dst = sub_seq.repeat(repeat_num)

        srcs.append(src)
        dsts.append(dst)
    src_ = torch.cat(srcs)
    dst_ = torch.cat(dsts)
    return src_,dst_

def create_adj_with_same_sens_small_cliques(sens, subnum, device):
    node_number = sens.shape[0]
    srcs, dsts = [],[] 
    
    for key in torch.unique(sens):
        node_indices = (sens==key).nonzero().squeeze()
        src, dst = create_random_cliques(node_indices, subnum)
        srcs.append(src)
        dsts.append(dst)

    src_ = torch.cat(srcs).long()
    dst_ = torch.cat(dsts).long()
    num_nodes = node_number
    edge_index = torch.stack([src_, dst_], dim=0).to(device)
    values = torch.ones(edge_index.shape[1], dtype=torch.float32).to(device)

    adj_sparse = torch.sparse_coo_tensor(
        indices=edge_index,
        values=values,
        size=(num_nodes, num_nodes),
        dtype=torch.float32
    )

    new_adj_dense = adj_sparse.to_dense()

    diag_elements = torch.diag(new_adj_dense)
    diag_matrix = torch.diag_embed(diag_elements)

    new_adj_dense -= diag_matrix

    return new_adj_dense

def compute_G_laplacian_eigen(adj, pe_dim):

    A = adj.tocsr()

    D_vec = A.sum(axis=1).A.squeeze()
    D_vec[D_vec == 0] = 1e-12  
    D_inv_sqrt = np.power(D_vec, -0.5)
    D_inv_sqrt_sparse = sp.diags(D_inv_sqrt)

    I = sp.eye(adj.shape[0], dtype=np.float32)
    L = (I - D_inv_sqrt_sparse @ A @ D_inv_sqrt_sparse)

    eigenvalues_np, eigenvectors_np = sla.eigsh(
        L.asfptype(), 
        k=min(pe_dim, adj.shape[0]), 
        which='LM', 
        sigma=1e-12, 
        maxiter=1000
    )
    
    eigenvalues = torch.from_numpy(eigenvalues_np).float()
    eigenvectors = torch.from_numpy(eigenvectors_np).float()
    
    return eigenvalues, eigenvectors

def create_fair_input(adj, features, K): 
    if K==0:
        return features.unsqueeze(1)
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1]) # (N, 1, K+1, d )
    
    for i in range(features.shape[0]): # node id

        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)
    x = x

    for i in range(K): # 0 -> K-1

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]        

    nodes_features = nodes_features.squeeze()


    return nodes_features

