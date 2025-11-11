from utils.evaluation import calc_metrics, calc_test_cm, calc_sens_test_metrics
import torch
import torch.nn.functional as F

class All_metrics:
    def __init__(self, acc, auc, f1, parity, equality, counterfactual_fairness,
                 recall, precision, cm, cm_sens0, cm_sens1,
                 acc_sens0, auc_sens0, f1_sens0, acc_sens1, auc_sens1, f1_sens1, model_param_cnt):
        self.acc = acc
        self.auc = auc
        self.f1 = f1
        self.parity = parity
        self.equality = equality
        self.counterfactual_fairness = counterfactual_fairness
        self.recall = recall
        self.precision = precision
        self.cm = cm
        self.cm_sens0 = cm_sens0
        self.cm_sens1 = cm_sens1
        self.acc_sens0 = acc_sens0
        self.auc_sens0 = auc_sens0
        self.f1_sens0 = f1_sens0
        self.acc_sens1 = acc_sens1
        self.auc_sens1 = auc_sens1
        self.f1_sens1 = f1_sens1
        self.model_param_cnt = model_param_cnt


class Early_stopper:
    def __init__(self, stop_count, metrics, alpha, trial, model_param_cnt):
        self.stop_count = stop_count
        self.metrics = metrics
        self.alpha = alpha
        self.trial = trial
        self.model_param_cnt = model_param_cnt

        self.best_val_tradeoff = -1
        self.early_stop_count = 0
        self.epoch = 0

    def check_stop(self, output, counter_output, data):
        accs, auc_rocs, F1s, paritys, equalitys, counterfactual_fairness = calc_metrics(output, counter_output, data, self.trial-1)
        if self.metrics == 'acc':
            check_val = accs['val']
        elif self.metrics == 'f1':
            check_val = F1s['val']
        elif self.metrics == 'alpha':
            check_val = F1s['val'] + accs['val'] - self.alpha * (paritys['val'] + equalitys['val'] + counterfactual_fairness['val']) # counterfactual_fairness追加

        if check_val >= self.best_val_tradeoff:
            self.test_acc = accs['test']
            self.test_auc_roc = auc_rocs['test']
            self.test_f1 = F1s['test']
            self.parity = paritys['test']
            self.equality = equalitys['test']
            self.counterfactual_fairness = counterfactual_fairness['test']            
            self.best_val_tradeoff = check_val
            self.early_stop_count = 0
            self.best_epoc = self.epoch
            self.best_output = output
        else:
            #print(self.early_stop_count)
            #print(check_val)
            #print(self.best_val_tradeoff)

            self.early_stop_count += 1
            if self.early_stop_count >= self.stop_count:
                return True
        self.epoch = self.epoch + 1
        return False

    def get_all_metrics(self, data):
        val_metrics = self.get_all_metrics_sub(data, True)
        test_metrics = self.get_all_metrics_sub(data, False)
        return [val_metrics, test_metrics]

    def get_all_metrics_sub(self, data, is_val):
        recall, precision, cm, cm_sens0, cm_sens1 = calc_test_cm(self.best_output, data, self.trial-1, is_val)
        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = \
            calc_sens_test_metrics(self.best_output, data, self.trial-1, is_val)

        return All_metrics(self.test_acc, self.test_auc_roc, self.test_f1, self.parity, self.equality, self.counterfactual_fairness,
        recall, precision, cm, cm_sens0, cm_sens1,
        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1,
        self.model_param_cnt)
        
        
class ALM:
    def __init__(self, args):
        self.sp_tolerance = args.sp_tolerance
        self.eo_tolerance = args.eo_tolerance
        self.cf_tolerance = args.cf_tolerance
        self.penalty_factor = args.penalty_factor
        self.update_weight = args.update_weight 
        self.previous_violations = float('inf')
        self.multiplier = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(args.device)             
        self.device = args.device
        
        
    # SP, EO, CFの各制約違反度を計算, ベクトルとしてまとめる
    def concat_violations(self, output, counter_output, labels, sens):

        idx_s0 = sens == 0
        idx_s1 = sens == 1
        sens_mask = (sens == 0) | (sens == 1)           
        idx_s0_y1 = torch.bitwise_and(idx_s0.unsqueeze(1), labels == 1)
        idx_s1_y1 = torch.bitwise_and(idx_s1.unsqueeze(1), labels == 1)

        sigmoid_prob = torch.sigmoid(output)    
        cf_sigmoid_prob = torch.sigmoid(counter_output)

        sp_violation1 = (sigmoid_prob[idx_s0].sum() / idx_s0.sum() - sigmoid_prob[idx_s1].sum() / idx_s1.sum()) - self.sp_tolerance
        sp_violation2 = (sigmoid_prob[idx_s1].sum() / idx_s1.sum() - sigmoid_prob[idx_s0].sum() / idx_s0.sum()) - self.sp_tolerance
  
        eo_violation1 = (sigmoid_prob[idx_s0_y1].sum() / idx_s0_y1.sum() - sigmoid_prob[idx_s1_y1].sum() / idx_s1_y1.sum()) - self.eo_tolerance 
        eo_violation2 = sigmoid_prob[idx_s1_y1].sum() / idx_s1_y1.sum() - sigmoid_prob[idx_s0_y1].sum() / idx_s0_y1.sum() - self.eo_tolerance
   
        cf_violation1 =  ((sigmoid_prob - cf_sigmoid_prob).sum() / sens_mask.sum()) - self.cf_tolerance
        cf_violation2 = (cf_sigmoid_prob - sigmoid_prob).sum() / sens_mask.sum() - self.cf_tolerance   

        violation_vector =  torch.cat((sp_violation1.unsqueeze(0), sp_violation2.unsqueeze(0), eo_violation1.unsqueeze(0), eo_violation2.unsqueeze(0), cf_violation1.unsqueeze(0), cf_violation2.unsqueeze(0)), dim=0) 

        return violation_vector
    
    def augmented_lagrange_function(self, output, counter_output, labels, sens):
        BCE_loss = F.binary_cross_entropy_with_logits(output, labels.unsqueeze(1).float().to(self.device))

        violation_vector = self.concat_violations(output, counter_output, labels.unsqueeze(1).float().to(self.device), sens)
        
        max_term_vector = self.multiplier + 2 * self.penalty_factor * violation_vector
        max_term_vector[max_term_vector <= 0] = 0

        augmented_lagrange_loss = BCE_loss + (1/ (4 * self.penalty_factor)) * (torch.dot(max_term_vector, max_term_vector) - torch.dot(self.multiplier, self.multiplier)) 
        
        return augmented_lagrange_loss, max_term_vector, violation_vector
    
    def function_update(self, max_term_vector, current_violations):
        with torch.no_grad():
            self.multiplier.copy_(max_term_vector) 
            
            # if(current_violations > self.previous_violations * 0.5): 
            #     self.penalty_factor *= self.update_weight
            #     print('penalty_update')   
            # else:                                                        
            #     self.previous_violations = current_violations         
