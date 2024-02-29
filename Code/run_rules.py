#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os
import pandas as pd
import numpy as np

log_dir = './SimTSC/logs'
tmp_dir = 'tmp'
dataset = "AHU_Winter_2022_2023"
dataset_dir = './SimTSC/datasets/EBC'

def load_pred(dataset, log_dir):
    """ Load GNN prediction labels as list from pickle file
    """
    seed = 0
    test_out_dir = os.path.join(log_dir, 'TEST')
    preds_out_path = os.path.join(test_out_dir, dataset+'_'+str(seed)+'_preds')
    with open (preds_out_path, 'rb') as fp:
        labellist = pickle.load(fp)
    return labellist

def data_loader(dataset_dir, dataset):
    dataset_dir = os.path.join(dataset_dir, dataset)
    df_test = pd.read_csv(os.path.join(dataset_dir, dataset+'_TEST.tsv'), sep='\t', header=None)
    y_test = df_test.values[:, 0].astype(str)
    X_test = df_test.drop(columns=[0]).astype(np.float32)
    X_test = X_test.values
    return X_test, y_test

def logic_rule_ODATemp(X_vals, y_preds, y_truths):
    rule_min = -10.0
    rule_max = 30.0
    applicable_class = "Temperature"
    class_toapply = "ODA"
    result_idx = []

    y_truth_idx = [idx for idx, x in enumerate(y_truths) if class_toapply in x]

    idx = np.array([i for i in range(len(X_vals))])
    for i in idx:
        if y_preds[i] == applicable_class:
            minval = np.amin(X_vals[i,:])
            maxval = np.amax(X_vals[i,:])
            if minval>rule_min and maxval<rule_max:
                result_idx.append(i)
    rule_result = dict()
    rule_result[class_toapply] = result_idx
    return rule_result, y_truth_idx

def logic_rule_SUPTemp(X_vals, y_preds, y_truths):
    rule_min = 15.0
    rule_max = 40.0
    applicable_class = "Temperature"
    class_toapply = "SUP"
    result_idx = []

    idx = np.array([i for i in range(len(X_vals))])
    for i in idx:
        if y_preds[i] == applicable_class:
            minval = np.amin(X_vals[i,:])
            maxval = np.amax(X_vals[i,:])
            if minval>rule_min and maxval<rule_max:
                result_idx.append(i)
    rule_result = dict()
    rule_result[class_toapply] = result_idx
    y_truth_idx = [idx for idx, x in enumerate(y_truths) if class_toapply in x]
    return rule_result, y_truth_idx


def logic_rule_PHValve(X_vals, y_preds, y_truths):
    applicable_class = ["Temperature", "Valve"]
    class_toapply = ["PH", "ODA"]
    valve_result_idx = []
    temp_result_idx = []

    temp_idx = [idx for idx, x in enumerate(y_preds) if x == "Temperature"]
    valve_idx = [idx for idx, x in enumerate(y_preds) if x == "Valve"]
    for t_idx in temp_idx:
        tstamps = [idx for idx, x in enumerate(X_vals[t_idx,:]) if x < -3.0] # find timestamps where temperature value < -3°C
        if tstamps: # check if list is not empty
            for v_idx in valve_idx:
                matches = [True for x in X_vals[v_idx,tstamps] if x > 0.0] # check if valve > 0% 
                overlap = len(matches) / len(X_vals[v_idx,tstamps])
                if overlap > 0.9:
                    if v_idx not in valve_result_idx:
                        valve_result_idx.append(v_idx)
                    if t_idx not in temp_result_idx:
                        temp_result_idx.append(t_idx)
    PH_result = dict()
    ODA_result = dict()
    ODA_result['ODA'] = temp_result_idx
    PH_result['PH'] = valve_result_idx

    PH_truth_idx = [idx for idx, x in enumerate(y_truths) if class_toapply[0] in x]
    ODA_truth_idx = [idx for idx, x in enumerate(y_truths) if class_toapply[1] in x]
    return PH_result, ODA_result, PH_truth_idx, ODA_truth_idx


def compute_accuracy(rule_preds, truth_idxs):
    rule_pred_key = list(rule_preds.keys())[0]
    rule_pred_idxs = list(rule_preds.values())[0]
    intersect = list(set(rule_pred_idxs) & set(truth_idxs))
    print(rule_preds)
    acc = (len(intersect) / len(rule_pred_idxs))*100
    return acc


if __name__ == "__main__":
    labellist = load_pred(dataset, log_dir)
    X_test, y_test = data_loader(dataset_dir, dataset)

    # rule_preds_ODATemp, trueidx_ODA = logic_rule_ODATemp(X_test, labellist, y_test)
    rule_preds_SUPTemp, trueidx_SUP = logic_rule_SUPTemp(X_test, labellist, y_test)

    # oda_acc = compute_accuracy(rule_preds_ODATemp, trueidx_ODA)
    # print(oda_acc)
    sup_acc = compute_accuracy(rule_preds_SUPTemp, trueidx_SUP)
    print("SUP true idxs: ", trueidx_SUP)
    print("SUP accuracy: ",sup_acc)

    rule_preds_PH, rule_preds_ODA, trueidx_PH, trueidx_ODA= logic_rule_PHValve(X_test, labellist, y_test)
    ph_acc = compute_accuracy(rule_preds_PH, trueidx_PH)
    print("PH true idxs: ", trueidx_PH)
    print("PH accuracy: ",ph_acc)
    oda_acc = compute_accuracy(rule_preds_ODA, trueidx_ODA)
    print("ODA true idxs: ", trueidx_ODA)
    print("ODA accuracy: ",oda_acc)
