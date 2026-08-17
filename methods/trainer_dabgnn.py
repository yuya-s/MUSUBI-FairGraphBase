import csv
import platform
import time
import torch

from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, autograd
from models.model_dabgnn import StructualModel, AttributeModel, InteractionModel, WDapproximator
from models.model import Classifier
from methods.trainer_utils import Early_stopper
from utils.utils import get_gpu_info, params_count
from utils.evaluation import calc_metrics


def run_trial_dabgnn(data, args, trial=1):
    epochs = args.epochs
    device = args.device
    
    labels = data.y
    features = data.x
    features = features.to(device)       
    counter_features = data.counter_x
    counter_features = counter_features.to(device)
    edge_index  = data.edge_index
    knn_index = make_knn_edge_index(features.cpu().numpy(), args.k) 
    knn_index = knn_index.to(device)
    sens = data.sens
    sens= sens.to(device)              
    train_mask=data.train_mask[trial-1]
    val_mask = data.val_mask[trial-1]

    
    model_str = StructualModel(encoder=args.encoder, nfeat=features.shape[1], num_hidden=args.hidden, gnn_layer_size=args.gnn_layer_size , gnn_hidden=args.gnn_hidden, device=device, data=data)
    model_atr = AttributeModel(encoder=args.encoder, nfeat=features.shape[1], num_hidden=args.hidden, gnn_layer_size=args.gnn_layer_size , gnn_hidden=args.gnn_hidden, device=device, data=data)
    model_pot = InteractionModel(num_hidden=args.hidden*2, device=device)
    classifier_final = Classifier(in_features=args.hidden*3, out_features=1, num_layers=args.cls_layer_size, use_spectral_norm=True)
    classifier_final = classifier_final.to(device)
    optimizer_str = optim.Adam(model_str.parameters(), lr=args.s_lr, weight_decay=args.weight_decay)
    optimizer_atr = optim.Adam(model_atr.parameters(), lr=args.a_lr, weight_decay=args.weight_decay)
    optimizer_pot = optim.Adam(model_pot.parameters(), lr=args.l_lr, weight_decay=args.weight_decay)
    optimizer_cl = optim.Adam(classifier_final.parameters(), lr=args.c_lr, weight_decay=args.weight_decay)
    
    wd_str = WDapproximator(nfeat=args.hidden, device=device)
    wd_atr = WDapproximator(nfeat=args.hidden, device=device)
    wd_pot = WDapproximator(nfeat=args.hidden, device=device)
    optimizer_wd_str = optim.Adam(wd_str.parameters(), lr=args.w_lr, weight_decay=args.weight_decay)
    optimizer_wd_atr = optim.Adam(wd_atr.parameters(), lr=args.w_lr, weight_decay=args.weight_decay)
    optimizer_wd_pot = optim.Adam(wd_pot.parameters(), lr=args.w_lr, weight_decay=args.weight_decay)


    params_model_str = params_count(model_str)
    params_model_atr = params_count(model_atr)
    params_model_pot = params_count(model_pot)
    params_classifier = params_count(classifier_final)   
    params_num = params_model_str + params_model_atr + params_model_pot + params_classifier
       
    record_emb_str = Store_embedding(20, args.metrics, args.alpha, trial, params_model_str) 
    record_emb_atr = Store_embedding(20, args.metrics, args.alpha, trial, params_model_atr) 
    record_emb_pot = Store_embedding(20, args.metrics, args.alpha, trial, params_model_pot) 
    early_stopper_final = Early_stopper(20, args.metrics, args.alpha, trial, params_num) 

    model_name = args.inprocessing 
    with open(
            f'{args.output_dir}/train_time/train_time_DAB-GCN_{model_name}_{trial}.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train time"])
        
        
        for epoch1 in tqdm(range(epochs)): #forループの回数は調整が必要
            start_time = time.time()
            model_str.train()
            model_atr.train()
            optimizer_str.zero_grad()
            optimizer_atr.zero_grad()

            cl_str, emb_str = model_str(edge_index) 
            cl_atr, emb_atr = model_atr(features, knn_index) 
                               
            cl_loss_str = F.binary_cross_entropy_with_logits(cl_str[train_mask],
                                                         labels[train_mask].unsqueeze(1).float().to(
                                                              device))        

            cl_loss_atr = F.binary_cross_entropy_with_logits(cl_atr[train_mask],
                                                         labels[train_mask].unsqueeze(1).float().to(
                                                              device))         
            
            wd_loss_str = calculate_wd_loss(wd_str.to(device), emb_str[train_mask], sens[train_mask], args.s_alpha)
            wd_loss_atr = calculate_wd_loss(wd_atr.to(device), emb_atr[train_mask], sens[train_mask], args.a_alpha)
            
            disentanglement_loss = calculate_dis_loss(structural_embedding=emb_str[train_mask], attribute_embedding=emb_atr[train_mask], 
                                                      num_hidden=args.hidden, dis=args.dis, div_by_hidden=True)    
            
            total_loss = (cl_loss_str + wd_loss_str) + (cl_loss_atr + wd_loss_atr) + disentanglement_loss
            
            total_loss.backward()         
            optimizer_str.step()
            optimizer_atr.step()
            
            optimize_wd_approximator(wd_str, optimizer_wd_str, emb_str[train_mask].detach(), sens[train_mask], args.lambda_gp, device)
            optimize_wd_approximator(wd_atr, optimizer_wd_atr, emb_atr[train_mask].detach(), sens[train_mask], args.lambda_gp, device)
            
            model_str.eval()
            model_atr.eval()
            with torch.no_grad():
                
                cl_str, emb_str = model_str(edge_index)
                counter_cl_str, counter_emb_str = model_str(edge_index)
                 
                cl_atr, emb_atr = model_atr(features, knn_index)    
                counter_cl_atr, counter_emb_atr = model_atr(counter_features, knn_index)

                record_emb_str.store_best_emb(cl_str, counter_cl_str, data, emb_str, counter_emb_str)                                                       
                record_emb_atr.store_best_emb(cl_atr, counter_cl_atr, data, emb_atr, counter_emb_atr)   

            end_time = time.time()
            train_time = end_time - start_time
            writer.writerow([epoch1, train_time])

    
        best_emb_str = record_emb_str.best_embedding.detach()
        best_emb_atr = record_emb_atr.best_embedding.detach()
        concat_emb = torch.cat((best_emb_str, best_emb_atr), dim=1) 
        counter_emb_str = record_emb_str.counter_embedding.detach()
        counter_emb_atr = record_emb_atr.counter_embedding.detach()
        concat_counter_emb = torch.cat((counter_emb_str, counter_emb_atr), dim=1)
        
        for epoch2 in tqdm(range(epochs)): #forループの回数は調整が必要
            start_time = time.time()
            model_pot.train()
            optimizer_pot.zero_grad()

            cl_pot, emb_pot = model_pot(concat_emb) 
                               
            cl_loss_pot = F.binary_cross_entropy_with_logits(cl_pot[train_mask],
                                                         labels[train_mask].unsqueeze(1).float().to(
                                                              device))        

            wd_loss_pot = calculate_wd_loss(wd_pot, emb_pot[train_mask], sens[train_mask], args.l_alpha)
            
            disentanglement_loss = calculate_dis_loss_pot(best_emb_str[train_mask], best_emb_atr[train_mask], emb_pot[train_mask], args.l_dis)    
            
            total_loss = (cl_loss_pot + wd_loss_pot) + disentanglement_loss
            
            total_loss.backward()         
            optimizer_pot.step()
            
            optimize_wd_approximator(wd_pot, optimizer_wd_pot, emb_pot[train_mask].detach(), sens[train_mask], args.lambda_gp, device)
            
            model_pot.eval()
            with torch.no_grad():
                
                cl_pot, emb_pot = model_pot(concat_emb)
                counter_cl_pot, counter_emb_pot = model_pot(concat_counter_emb)
                 
                record_emb_pot.store_best_emb(cl_pot, counter_cl_pot, data, emb_pot, counter_emb_pot)                                          
 
            end_time = time.time()
            train_time = end_time - start_time
            writer.writerow([epoch2, train_time])



        best_emb_pot = record_emb_pot.best_embedding.detach()
        concat_emb_final = torch.cat((best_emb_str, best_emb_atr, best_emb_pot), dim=1)
        counter_emb_pot = record_emb_pot.counter_embedding.detach()
        concat_counter_emb_final = torch.cat((counter_emb_str, counter_emb_atr, counter_emb_pot), dim=1)
 
        for epoch3 in tqdm(range(epochs)):
            start_time = time.time()
            classifier_final.train()
            optimizer_cl.zero_grad()

            cl_final = classifier_final(concat_emb_final) 
                               
            cl_loss_final = F.binary_cross_entropy_with_logits(cl_final[train_mask],
                                                         labels[train_mask].unsqueeze(1).float().to(
                                                              device))        
            
            cl_loss_final.backward()         
            optimizer_cl.step()
            
            classifier_final.eval()
            with torch.no_grad():
                cl_final = classifier_final(concat_emb_final)
                counter_cl_final = classifier_final(concat_counter_emb_final)
                 
            if early_stopper_final.check_stop(cl_final, counter_cl_final, data):
                break                                    
 
            end_time = time.time()
            train_time = end_time - start_time
            writer.writerow([epoch3, train_time])
    
    all_metrics = early_stopper_final.get_all_metrics(data)
    return all_metrics, early_stopper_final.best_output






def make_knn_edge_index(feature, k):

    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    nearest_neighbors.fit(feature)

    adjacency_matrix_sparse = nearest_neighbors.kneighbors_graph(feature, mode='connectivity')
    adjacency_matrix_transposed = adjacency_matrix_sparse.transpose()
    adjacency_matrix_symmetric = adjacency_matrix_sparse.maximum(adjacency_matrix_transposed)

    knn_adj = sp.coo_matrix(adjacency_matrix_symmetric, dtype=np.float32)

    knn_edge_index, _ = from_scipy_sparse_matrix(knn_adj)

    return knn_edge_index


def optimize_wd_approximator(wd_approximator, optimizer_wd, embedding, sens, lambda_gp, device):
    wd_approximator.requires_grad_(True)
    wd_approximator.train()
    for _ in range(8):
        optimizer_wd.zero_grad()
        wasserstein_distances = wd_approximator(embedding)

        positive_eles = torch.masked_select(wasserstein_distances.squeeze(),
                                            sens == 1)
        negative_eles = torch.masked_select(wasserstein_distances.squeeze(),
                                            sens == 0)

        positive_embedding = embedding[sens == 1]
        negative_embedding = embedding[sens == 0]

        gp = compute_gradient_penalty(wd_approximator, positive_embedding, negative_embedding, device)

        wd_loss_train = (torch.mean(positive_eles) - torch.mean(negative_eles)) + lambda_gp * gp
        wd_loss_train.backward()
        optimizer_wd.step()
        

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if real_samples.size(0) < fake_samples.size(0):
        size = real_samples.size(0)
        fake_samples = fake_samples[:size]
    else:
        size = fake_samples.size(0)
        real_samples = real_samples[:size]
    alpha = Tensor(np.random.random((size, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Tensor(size, 1).fill_(1.0).requires_grad_(False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calculate_wd_loss(wd_approximator, embedding, sens, alpha):
    wd_approximator.requires_grad_(False)
    wasserstein_distances = wd_approximator(embedding)

    positive_eles = torch.masked_select(wasserstein_distances.squeeze(),
                                        sens == 1)
    negative_eles = torch.masked_select(wasserstein_distances.squeeze(),
                                        sens == 0)

    wd_loss = - (torch.mean(positive_eles) - torch.mean(negative_eles)) * alpha
    return wd_loss


def calculate_dis_loss(structural_embedding, attribute_embedding, num_hidden, dis, div_by_hidden=True):
    DISLoss = torch.nn.MSELoss()
    a = (1/num_hidden) if div_by_hidden else 1
    dis_loss = dis * a * (DISLoss(structural_embedding,attribute_embedding))

    return dis_loss


def calculate_dis_loss_pot(best_structural_embedding, best_attribute_embedding, potential_embedding, l_dis):
    DISLoss = torch.nn.MSELoss()
    dis_loss = 0.5 * l_dis * (
        DISLoss(best_structural_embedding, potential_embedding) +       
        DISLoss(best_attribute_embedding, potential_embedding)
    )
    return dis_loss


class Store_embedding(Early_stopper):
    def __init__(self, stop_count, metrics, alpha, trial, model_param_cnt):
        super().__init__(stop_count, metrics, alpha, trial, model_param_cnt)
    
    def store_best_emb(self, output, counter_output, data, embedding, counter_embedding): 
        self.check_stop(output, counter_output, data)
        if self.check_val >= self.best_val_tradeoff:
            self.best_embedding = embedding
            self.counter_embedding = counter_embedding
     

