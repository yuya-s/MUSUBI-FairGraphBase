from utils.evaluation import calc_metrics, calc_test_cm, calc_sens_test_metrics


class All_metrics:
    def __init__(self, acc, auc, f1, parity, equality,
                 recall, precision, cm, cm_sens0, cm_sens1,
                 acc_sens0, auc_sens0, f1_sens0, acc_sens1, auc_sens1, f1_sens1, model_param_cnt):
        self.acc = acc
        self.auc = auc
        self.f1 = f1
        self.parity = parity
        self.equality = equality
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

    def check_stop(self, output, data):
        accs, auc_rocs, F1s, paritys, equalitys = calc_metrics(output, data, self.trial-1)
        if self.metrics == 'acc':
            check_val = accs['val']
        elif self.metrics == 'f1':
            check_val = F1s['val']
        elif self.metrics == 'alpha':
            check_val = F1s['val'] + accs['val'] - self.alpha * (paritys['val'] + equalitys['val'])

        if check_val > self.best_val_tradeoff:
            self.test_acc = accs['test']
            self.test_auc_roc = auc_rocs['test']
            self.test_f1 = F1s['test']
            self.parity = paritys['test']
            self.equality = equalitys['test']
            self.best_val_tradeoff = check_val
            self.early_stop_count = 0
            self.best_epoc = self.epoch
            self.best_output = output
        else:
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

        return All_metrics(self.test_acc, self.test_auc_roc, self.test_f1, self.parity, self.equality,
        recall, precision, cm, cm_sens0, cm_sens1,
        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1,
        self.model_param_cnt)