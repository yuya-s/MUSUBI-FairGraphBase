import csv
import platform
import time
import torch
import os

from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from models.model_sgc import SGC
from methods.trainer_utils import Early_stopper
from methods.trainer_utils import ALM
from utils.utils import params_count
from utils.evaluation import calc_metrics

def run_trial_constraint_sgc(data, args, trial=1):

    epochs = args.epochs
    device = args.device

    labels = data.y
    sens = data.sens
    
    adj_norm_sp = data.adj_norm_sp
    adj_norm_sp = adj_norm_sp.to(device)
    
    features = data.x
    features = features.to(device)
    processed_features = sgc_precompute(features, adj_norm_sp, args.degree)
    processed_features = processed_features.to(device) 
         
         
    counter_features = args.counter_features
    counter_features = counter_features.to(device)
    processed_counter_features = sgc_precompute(counter_features, adj_norm_sp, args.degree)
    processed_counter_features = processed_counter_features.to(device)  
    
    train_mask=data.train_mask[trial-1]
    val_mask = data.val_mask[trial-1]
    
    model = SGC(num_features = processed_features.shape[1],
                nclass = 1,
                device = device                
                )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
    
    model_name = args.inprocessing

    params_num = params_count(model)
    early_stopper = Early_stopper(20, args.metrics, args.alpha, trial, params_num)

    os.makedirs(f'{args.output_dir}/val_shift/', exist_ok=True)
    with open(f'{args.output_dir}/train_time/train_time_VGCN_{model_name}_{trial}.csv', 'w', newline="") as f, \
         open(f'{args.output_dir}/val_shift/val_shift_VGCN_{model_name}_{trial}.csv', 'w', newline="") as f2:

        writer = csv.writer(f)
        writer.writerow(["epoch", "train time"])
        
        writer2 = csv.writer(f2) 
        writer2.writerow(["epoch", "train_loss", "validation_loss", "accuracy", "F1", "SP", "EO", "CF", "ALM_update"])


        GRAD_THRESHOLD = 1e-1
        best_val_loss = float('inf')
        converge_count = 0
        
        augmented_lagrange = ALM(args)



        for epoch in tqdm(range(epochs)):
            start_time = time.time()        
            model.train()
            optimizer.zero_grad()

            output = model(features)
            counter_output = model(counter_features)

            train_loss, max_term_vector, current_violations = augmented_lagrange.augmented_lagrange_function(output[train_mask], counter_output[train_mask], labels[train_mask], sens[train_mask]) 
            train_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)               
            optimizer.step()           
            

            model.eval()
            val_loss, _, _ = augmented_lagrange.augmented_lagrange_function(output[val_mask], counter_output[val_mask], labels[val_mask], sens[val_mask])                    
             
            
            converge_flag = 0            
            #暫定解の収束判定
            if(grad_norm < GRAD_THRESHOLD):
                converge_flag = 1            
                print('ALM_converge')
                augmented_lagrange.function_update(max_term_vector, current_violations)
                        

            # early_stopper
            if(val_loss < best_val_loss):
                converge_count == 0
                best_val_loss = val_loss
                early_stopper.check_stop(output, counter_output, data)                  
            else:
                converge_count += 1                      
                        

            end_time = time.time()
            train_time = end_time - start_time
            accs, auc_rocs, F1s, paritys, equalitys, counterfactual_fairness = calc_metrics(output, counter_output, data, trial-1)  
            writer.writerow([epoch, train_time])
            writer2.writerow([epoch, train_loss.item(), val_loss.item(), accs['val'], F1s['val'], paritys['val'], equalitys['val'], counterfactual_fairness['val'], converge_flag])

            if(converge_count >= 100):
                break
                           
    if(converge_count < 0):
        early_stopper.check_stop(output, counter_output, data)     
           
    all_metrics = early_stopper.get_all_metrics(data)
    return all_metrics, early_stopper.best_output
    
    
    
    
    
    
    
# sgcの畳み込み
def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


