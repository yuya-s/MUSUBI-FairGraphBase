from utils.evaluation import calc_metrics, calc_test_cm, calc_sens_test_metrics

class All_metrics:
    def __init__(self, acc, auc, f1, parity, equality, equal_accuracy, counterfactual_fairness,
                 recall, precision, cm, cm_sens0, cm_sens1,
                 acc_sens0, auc_sens0, f1_sens0, acc_sens1, auc_sens1, f1_sens1, model_param_cnt):
        self.acc = acc
        self.auc = auc
        self.f1 = f1
        self.parity = parity
        self.equality = equality
        self.equal_accuracy = equal_accuracy
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

        self.check_val = 0
        self.best_val_tradeoff = float('-inf')
        self.early_stop_count = 0
        self.epoch = 0

    def check_stop(self, output, counter_output, data):
        accs, auc_rocs, F1s, paritys, equalitys, equal_accuracys, counterfactual_fairness = calc_metrics(output, counter_output, data, self.trial-1)
        if self.metrics == 'acc':
            self.check_val = accs['val']
        elif self.metrics == 'f1':
            self.check_val = F1s['val']
        elif self.metrics == 'alpha':
            self.check_val = F1s['val'] + accs['val'] - self.alpha * (paritys['val'] + equalitys['val'])

        if self.check_val >= self.best_val_tradeoff:
            self.best_val_tradeoff = self.check_val
            self.early_stop_count = 0
            self.best_epoc = self.epoch
            self.best_output = output
            #self.best_embedding = embedding
            self.test_acc = accs['test']
            self.test_auc_roc = auc_rocs['test']
            self.test_f1 = F1s['test']
            self.test_parity = paritys['test']
            self.test_equality = equalitys['test']
            self.test_counterfactual_fairness = counterfactual_fairness['test']
            self.test_equal_accuracy = equal_accuracys['test']

            self.val_acc = accs['val']   
            self.val_auc_roc = auc_rocs['val']
            self.val_f1 = F1s['val']
            self.val_parity = paritys['val']
            self.val_equality = equalitys['val']
            self.val_counterfactual_fairness = counterfactual_fairness['val']
            self.val_equal_accuracy = equal_accuracys['val']

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

        if(is_val):
            return All_metrics(self.val_acc, self.val_auc_roc, self.val_f1, self.val_parity, self.val_equality, 
            self.val_equal_accuracy, self.val_counterfactual_fairness,
            recall, precision, cm, cm_sens0, cm_sens1,
            ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, self.model_param_cnt)
        else:
            return All_metrics(self.test_acc, self.test_auc_roc, self.test_f1, self.test_parity, self.test_equality, 
            self.test_equal_accuracy, self.test_counterfactual_fairness,
            recall, precision, cm, cm_sens0, cm_sens1,
            ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, self.model_param_cnt)   
