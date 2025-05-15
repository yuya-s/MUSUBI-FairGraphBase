import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np

def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                   sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()

def calc_metrics(output, data, count):
    pred_val = (output[data.val_mask[count]].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask[count]].squeeze() > 0).type_as(data.y)
    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}
    accs['val'] = pred_val.eq(
        data.y[data.val_mask[count]]).sum().item() / data.val_mask[count].sum().item()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask[count]]).sum().item() / data.test_mask[count].sum().item()

    F1s['val'] = f1_score(data.y[data.val_mask[count]].cpu().numpy(), pred_val.cpu().numpy())
    F1s['test'] = f1_score(data.y[data.test_mask[count]].cpu().numpy(), pred_test.cpu().numpy())
    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask[count]].cpu().numpy(), output[data.val_mask[count]].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask[count]].cpu().numpy(), output[data.test_mask[count]].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask[count]].cpu(
    ).numpy(), data.sens[data.val_mask[count]].cpu().numpy())
    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask[count]].cpu(
    ).numpy(), data.sens[data.test_mask[count]].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys

def calc_test_cm(output, data, count, is_val=False):
    if is_val:
        sens = data.sens[data.val_mask[count]].cpu().numpy()
    else:
        sens = data.sens[data.test_mask[count]].cpu().numpy()

    idx_s0 = sens != 1
    idx_s1 = sens == 1

    if is_val:
        pred = (output[data.val_mask[count]].squeeze() > 0).type_as(data.y)
        label = data.y[data.val_mask[count]]
    else:
        pred = (output[data.test_mask[count]].squeeze() > 0).type_as(data.y)
        label = data.y[data.test_mask[count]]

    recall = recall_score(label.cpu().numpy(), pred.cpu().numpy())
    precision = precision_score(label.cpu().numpy(), pred.cpu().numpy(), zero_division=0)
    cm = confusion_matrix(label.cpu().numpy(), pred.cpu().numpy(), labels=[0,1])
    cm_sens0 = confusion_matrix(label[idx_s0].cpu().numpy(), pred[idx_s0].cpu().numpy(), labels=[0,1])
    cm_sens1 = confusion_matrix(label[idx_s1].cpu().numpy(), pred[idx_s1].cpu().numpy(), labels=[0,1])


    return recall, precision, cm, cm_sens0, cm_sens1


def calc_sens_test_metrics(output, data, count, is_val):
    labels = data.y.cpu().numpy()
    sens = data.sens
    if is_val:
        idx_test = data.val_mask[count].cpu().numpy()
    else:
        idx_test = data.test_mask[count].cpu().numpy()

    output = (output > 0).long().detach().cpu().numpy()
    pred = output[idx_test]
    result = []
    for sensval in [0, 1]:
        F1 = f1_score(
            labels[idx_test][
                sens[idx_test].detach().cpu().numpy() == sensval
            ],
            pred[sens[idx_test].detach().cpu().numpy() == sensval],
            average="micro",
        )
        ACC = accuracy_score(
            labels[idx_test][
                sens[idx_test].detach().cpu().numpy() == sensval
            ],
            pred[sens[idx_test].detach().cpu().numpy() == sensval],
        )
        if labels.max() > 1:
            AUCROC = 0
        else:
            y_true = labels[idx_test][
                    sens[idx_test].detach().cpu().numpy() == sensval
                ]
            if len(np.unique(y_true)) != 2:
                AUCROC = 0
            else:
                AUCROC = roc_auc_score(
                    y_true,
                    pred[sens[idx_test].detach().cpu().numpy() == sensval],
                )
        result.extend([ACC, AUCROC, F1])

    ACC_sens0 = result[0]
    AUCROC_sens0 = result[1]
    F1_sens0 = result[2]
    ACC_sens1 = result[3]
    AUCROC_sens1 = result[4]
    F1_sens1 = result[5]
    return ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1



def evaluate(x, classifier, hp, encoder, data, args, count):
    classifier.eval()
    encoder.eval()

    with torch.no_grad():
        h = encoder(data.x, data.edge_index)
        output = classifier(h)

    accs, auc_rocs, F1s, paritys, equalitys = calc_metrics(output, data, count)

    labels = data.y[data.test_mask[count]].cpu().numpy()
    pred = (output[data.test_mask[count]].squeeze() > 0).type_as(data.y).cpu().numpy()
    sens = data.sens[data.test_mask[count]].cpu().numpy()

    return accs, auc_rocs, F1s, paritys, equalitys, labels, pred, sens