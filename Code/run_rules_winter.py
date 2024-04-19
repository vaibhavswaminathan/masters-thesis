#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import tsfresh.feature_extraction.feature_calculators as tscalc

log_dir = './SimTSC/logs'
tmp_dir = 'tmp'
dataset = "AHU_prin_winter_2023_stanscaler_RULES"
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
    df_test = pd.read_csv(os.path.join(dataset_dir, dataset+'_SEPTEST.tsv'), sep='\t', header=None)
    y_test = df_test.values[:, 0:2].astype(str)
    X_test = df_test.drop(columns=[0,1]).astype(np.float32)
    X_test = X_test.values
    return X_test, y_test

def compute_rule_acc(rule_preds, truth_idxs):
    rule_pred_key = list(rule_preds.keys())[0]
    rule_pred_idxs = list(rule_preds.values())[0]
    intersect = list(set(rule_pred_idxs) & set(truth_idxs))
    # print(rule_preds)
    precision = (len(intersect) / len(rule_pred_idxs))*100
    recall = (len(intersect) / len(truth_idxs))*100
    acc = dict()
    acc['Precision'] = precision
    acc['Recall'] = recall
    return acc

def compute_overall_acc(pred_idxs, truth_idxs):
    intersect = list(set(pred_idxs) & set(truth_idxs))
    precision = (len(intersect) / len(pred_idxs))*100
    recall = (len(intersect) / len(truth_idxs))*100
    acc = dict()
    acc['Precision'] = precision
    acc['Recall'] = recall
    return acc

def rule_confmat(rule_preds, y_truths, predlabel, rulename):
    rule_pred_idxs = list(rule_preds.values())[0]
    true_labels = y_truths[:,0][rule_pred_idxs]
    target_names = list(set(true_labels))
    pred_labels = [str(predlabel)] * len(rule_pred_idxs)
    # strip common text from labels
    true_labels = [s.replace('ADSInternalValuesMirror','') for s in true_labels]
    target_names = [s.replace('ADSInternalValuesMirror','') for s in target_names]
    pred_labels = [s.replace('ADSInternalValuesMirror','') for s in pred_labels]

    conf_mat = confusion_matrix(true_labels, pred_labels, labels=target_names)
    conf_df = pd.DataFrame(conf_mat, index=target_names, columns=target_names)
    print(conf_df)
    log_out_dir = os.path.join(log_dir, 'RULES')
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir)
    confmat_path = os.path.join(log_out_dir, rulename+'_confusion_mat.csv')
    conf_df.to_csv(confmat_path)

def SUPTemp_rulelogic(X_vals, y_preds, y_truths):
    """ Rule: 15°C < SUPTemp < 40°C
        Seasonality: All
    """
    print("\n Running RULE 1")
    print("Rule: 15°C < SUPTemp < 40°C")
    rule_min = 15.0
    rule_max = 40.0
    applicable_class = "Temperature"
    class_toapply = "TempSUPADS"
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
    y_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply in x]
    return rule_result, y_truth_idx

def SUPTempSetpoint_rulelogic(X_vals, y_preds, y_truths):
    """ Rule: Temp Setpoint has least number of unique values i.e most number of reoccuring values
        Seasonality: All
    """
    print("\n Running RULE 11")
    print("Rule: Temp Setpoint has least number of unique values i.e most number of reoccuring values")
    applicable_class = "Temperature"
    class_toapply = "TempSUPSet"
    result_idx = []

    temp_idx = [idx for idx, x in enumerate(y_preds) if x == "Temperature"]
    for idx in temp_idx:
        rounded_temps = list(np.around(np.array(X_vals[idx,:]),3))
        reocc_ratio = tscalc.percentage_of_reoccurring_datapoints_to_all_datapoints(rounded_temps)
        if reocc_ratio >= 0.9:
            result_idx.append(idx)
    rule_result = dict()
    rule_result[class_toapply] = result_idx
    y_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply in x]
    return rule_result, y_truth_idx

def ODATemp_rulelogic(X_vals, y_preds, y_truths):
    print("\n Running RULE 2")
    rule_min = -10.0
    rule_max = 20.0
    applicable_class = "Temperature"
    class_toapply = "ODA"
    result_idx = []

    idx = np.array([i for i in range(len(X_vals))])
    # temp_index = []
    for i in idx:
        if y_preds[i] == applicable_class:
            # temp_index.append(i)
            minval = np.amin(X_vals[i,:])
            maxval = np.amax(X_vals[i,:])
            if minval>rule_min and maxval<rule_max:
                result_idx.append(i)
    # print(temp_index)
    rule_result = dict()
    rule_result[class_toapply] = result_idx
    y_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply in x]
    return rule_result, y_truth_idx

def ValveMin_winter_rulelogic(X_vals, y_preds, y_truths):
    """ Rule: If min(Valve over 5-day window) < 10%, then PH Valve
        Seasonality: Winter
    """
    print("\n Running RULE 12")
    print("Rule: If min(Valve over 5-day window) < 10%, then PH Valve")
    applicable_class = ["Valve"]
    class_toapply = ["PH"]
    valve_result_idx = []
    rule_min = 10.0

    valve_idx = [idx for idx, x in enumerate(y_preds) if x == "Valve"]
    # valve_idx = [idx for idx, x in enumerate(y_truths[:,0]) if "Valve" in x]
    for v_idx in valve_idx:
        minval = np.amin(X_vals[v_idx,:])
        if minval<=rule_min:
            if v_idx not in valve_result_idx:
                valve_result_idx.append(v_idx)
    PH_result = dict()
    PH_result['PH'] = valve_result_idx

    PH_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply[0] in x]
    return PH_result, PH_truth_idx


def PHValve_winter_rulelogic(X_vals, y_preds, y_truths, oda_preds):
    """ Rule: ODATemp < 3°C implies PH ValveAct > 0%
        Seasonality: Winter
    """
    print("\n Running RULE 3")
    print("Rule: ODATemp < 3°C implies PH ValveAct > 0%")
    applicable_class = ["Temperature", "Valve"]
    class_toapply = ["PH", "ODA"]
    valve_result_idx = []
    temp_result_idx = []

    temp_idx = [idx for idx in oda_preds['ODA']]
    # temp_idx = [idx for idx, x in enumerate(y_preds) if x == "Temperature"]
    # valve_idx = [idx for idx, x in enumerate(y_preds) if x == "Valve"]
    valve_idx = [idx for idx, x in enumerate(y_truths[:,0]) if "Valve" in x]
    for t_idx in temp_idx:
        for v_idx in valve_idx:
            if y_truths[t_idx][1] == y_truths[v_idx][1]: # if the windows are same
                tstamps = [idx for idx, x in enumerate(X_vals[t_idx,:]) if x < 3.0] # find timestamps where temperature value < 3°C
                if tstamps: # check if list is not empty
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

    PH_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply[0] in x]
    ODA_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply[1] in x]
    return PH_result, ODA_result, PH_truth_idx, ODA_truth_idx

def PHValve_summer_rulelogic(X_vals, y_preds, y_truths, oda_preds):
    """ Rule: ODATemp > 10°C implies PH ValveAct = 0%
        Seasonality: Summer
    """
    print("\n Running RULE 5")
    print("Rule : ODATemp > 10°C implies PH ValveAct = 0%")
    applicable_class = ["Temperature", "Valve"]
    class_toapply = ["PH", "ODA"]
    valve_result_idx = []
    temp_result_idx = []

    # temp_idx = [idx for idx, x in enumerate(y_preds) if x == "Temperature"]
    temp_idx = [idx for idx in oda_preds['ODA']]
    valve_idx = [idx for idx, x in enumerate(y_preds) if x == "Valve"]
    # temp_idx = [idx for idx, x in enumerate(y_truths[:,0]) if "Temp" in x]
    # valve_idx = [idx for idx, x in enumerate(y_truths[:,0]) if "Valve" in x]
    for t_idx in temp_idx:
        for v_idx in valve_idx:
            if y_truths[t_idx][1] == y_truths[v_idx][1]: # if the windows are same
                tstamps = [idx for idx, x in enumerate(X_vals[t_idx,:]) if x > 10.0] # find timestamps where temperature value > 10°C
                if tstamps: # check if list is not empty
                    matches = [True for x in X_vals[v_idx,tstamps] if x < 0.5] # check if valve = 0% 
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

    PH_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply[0] in x]
    ODA_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply[1] in x]
    return PH_result, ODA_result, PH_truth_idx, ODA_truth_idx

def PumpOperating_rulelogic(X_vals, y_preds, y_truths, preds_phvalve):
    """ Rule: PH ValveAct >= 20% implies PH PumpOperating = 1
        Seasonality: All
    """
    print("\n Running RULE 4")
    print(" Rule: PH ValveAct >= 20% implies PH PumpOperating = 1")
    applicable_class = ["Operating", "Valve"]
    class_toapply = ["PH"]
    valve_result_idx = []
    oper_result_idx = []

    oper_idx = [idx for idx, x in enumerate(y_preds) if x == "Operating"]
    # valve_idx = [idx for idx, x in enumerate(y_preds) if x == "Valve"]
    valve_idx = [idx for idx in preds_phvalve['PH']]
    for o_idx in oper_idx:
        for v_idx in valve_idx:
            if y_truths[o_idx][1] == y_truths[v_idx][1]: # if the windows are same
                tstamps = [idx for idx, x in enumerate(X_vals[o_idx,:]) if x == 1.0] # find timestamps where pump operating = 1
                if tstamps: # check if list is not empty
                    matches = [True for x in X_vals[v_idx,tstamps] if x > 20.0] # check if valve > 20% 
                    overlap = len(matches) / len(X_vals[v_idx,tstamps])
                    if overlap > 0.8:
                        if v_idx not in valve_result_idx:
                            valve_result_idx.append(v_idx)
                        if o_idx not in oper_result_idx:
                            oper_result_idx.append(o_idx)
    PH_result = dict()
    PHPump_result = dict()
    PHPump_result['PH'] = oper_result_idx
    PH_result['PH'] = valve_result_idx

    PH_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply[0] in x]
    return PH_result, PHPump_result, PH_truth_idx

def delta_ODATemp_rulelogic(X_vals, y_preds, y_truths):
    """ Rule: If Max - Min > 5°C implies ODA Temp 
        Seasonality: All (except sometimes in summer)
    """
    print("\n Running RULE 6")
    print("Rule : Max temp - Min temp > 5°C implies ODA Temp")
    delta = 5.0
    applicable_class = "Temperature"
    class_toapply = "ODA"
    result_idx = []

    y_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply in x]

    idx = np.array([i for i in range(len(X_vals))])
    for i in idx:
        if y_preds[i] == applicable_class:
            minval = np.amin(X_vals[i,:])
            maxval = np.amax(X_vals[i,:])
            if np.abs(maxval - minval) > delta:
                result_idx.append(i)
    rule_result = dict()
    rule_result[class_toapply] = result_idx
    return rule_result, y_truth_idx

def ETATemp_rulelogic(X_vals, y_preds, y_truths, sup_preds, oda_preds):
    print("\n Running RULE 7")
    applicable_class = "Temperature"
    class_toapply = "ETA"
    result_idx = []

    y_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply in x]
    temp_idx = [idx for idx, x in enumerate(y_preds) if x == "Temperature"]
    idx = np.array([i for i in range(len(X_vals))])
    for i in temp_idx:
        if i not in sup_preds['TempSUP'] and i not in oda_preds['ODA']:    
            result_idx.append(i)
    rule_result = dict()
    rule_result[class_toapply] = result_idx
    return rule_result, y_truth_idx

def RemValve_rulelogic(X_vals, y_preds, y_truths, ph_preds):
    """ Rule:  All Valves - PH Valves = RH Valves
        Seasonality: All
    """
    print("\n Running RULE 8")
    print("Rule : All Valves - PH Valves = RH Valves")
    applicable_class = "Valve"
    class_toapply = "RH"
    result_idx = []

    valve_idx = [idx for idx, x in enumerate(y_preds) if x == "Valve"]
    y_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply in x]
    # idx = np.array([i for i in range(len(X_vals))])
    for i in valve_idx:
        if i not in ph_preds['PH']:
            result_idx.append(i)
    rule_result = dict()
    rule_result[class_toapply] = result_idx
    return rule_result, y_truth_idx

def RHPump_rulelogic(X_vals, y_preds, y_truths, ph_preds):
    """ Rule:  All Valves - PH Valves = RH Valves
        Seasonality: All
    """
    print("\n Running RULE 10")
    print("Rule : RH Pump = All Pump - PH Pump")
    applicable_class = "Operating"
    class_toapply = "RH"
    result_idx = []

    oper_idx = [idx for idx, x in enumerate(y_preds) if x == applicable_class]
    # CO_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if "COPump" in x]
    y_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply in x]
    for i in oper_idx:
        if i not in ph_preds['PH']:
            result_idx.append(i)
    rule_result = dict()
    rule_result[class_toapply] = result_idx
    return rule_result, y_truth_idx

def RHValve_rulelogic(X_vals, y_preds, y_truths):
    """ Rule: diff(ODA_Temp , SUP_Temp) > 5°C implies RH ValveAct > 20%
        Seasonality: All
    """
    print("\n Running RULE 9")
    applicable_class = ["Temperature", "Valve"]
    class_toapply = "RHValve"
    valve_result_idx = []

    ODAtemp_idx = [idx for idx, x in enumerate(y_truths) if "TempODA" in x]
    SUPset_idx = [idx for idx, x in enumerate(y_truths) if "TempSUPSet" in x]
    valve_idx = [idx for idx, x in enumerate(y_preds) if x == "Temperature"]
    # print(len(y_truths[valve_idx]))
    # print(len(y_truths))
    for oda_idx in ODAtemp_idx:
        for sup_idx in SUPset_idx:
            allstamps = [idx for idx, x in enumerate(X_vals[oda_idx,:])]
            tstamps = [idx for idx in allstamps if X_vals[sup_idx,idx] - X_vals[oda_idx,idx] > 10.0] # find timestamps where SUP_Set - ODA_Temp > 5°C
            if tstamps: # check if list is not empty
                for v_idx in valve_idx:
                    matches = [True for x in X_vals[v_idx,tstamps] if x > 20.0] # check if valve > 20% 
                    overlap = len(matches) / len(X_vals[v_idx,tstamps])
                    if overlap > 0.9:
                        if v_idx not in valve_result_idx: 
                            valve_result_idx.append(v_idx)
    RH_result = dict()
    RH_result['RH'] = valve_result_idx

    RH_truth_idx = [idx for idx, x in enumerate(y_truths[:,0]) if class_toapply[0] in x]
    return RH_result, RH_truth_idx


if __name__ == "__main__":
    labellist = load_pred(dataset, log_dir)
    X_test, y_test = data_loader(dataset_dir, dataset)
    print(X_test.shape)
    print(y_test.shape)

    ### ---> RULE 11 ###
    rule_preds_SUPTempSet, trueidx_SUPSet = SUPTempSetpoint_rulelogic(X_test, labellist, y_test)
    sup_acc = compute_rule_acc(rule_preds_SUPTempSet, trueidx_SUPSet)
    sup_preci = sup_acc['Precision']
    sup_rec = sup_acc['Recall']
    print("SUP Setpoint --> Precision: {:.2f}, Recall: {:.2f}".format(sup_preci, sup_rec))

    # Confusion Matrix
    # predlabel = 'ADS.fAHUTempSUPSetADSInternalValuesMirror'
    # rule_confmat(rule_preds_SUPTempSet, y_test, predlabel, "TempSUPSet")
    
    ### ---> RULE 1 ###
    rule_preds_SUPTemp, trueidx_SUP = SUPTemp_rulelogic(X_test, labellist, y_test)
    filter_SUPTemp_preds = dict()
    filter_SUPTemp_preds['TempSUP'] = [x for x in rule_preds_SUPTemp['TempSUPADS'] if x not in rule_preds_SUPTempSet['TempSUPSet']]
    sup_acc = compute_rule_acc(filter_SUPTemp_preds, trueidx_SUP)
    sup_preci = sup_acc['Precision']
    sup_rec = sup_acc['Recall']
    print("SUP --> Precision: {:.2f}, Recall: {:.2f}".format(sup_preci, sup_rec))
    # print("SUP true idxs: ", trueidx_SUP)
    
    ### ---> RULE 2 ###
    rule_preds_ODATemp, trueidx_ODA = ODATemp_rulelogic(X_test, labellist, y_test)
    filter_ODATemp_preds_1 = dict()
    filter_ODATemp_preds_1['ODA'] = [x for x in rule_preds_ODATemp['ODA'] if x not in rule_preds_SUPTemp['TempSUPADS']] # if a point has already been predicted as SUP Temp, do not consider it for ODA
    oda_acc = compute_rule_acc(filter_ODATemp_preds_1, trueidx_ODA)
    oda_preci = oda_acc['Precision']
    oda_rec = oda_acc['Recall']
    # print("ODA true idxs: ", trueidx_ODA)
    print("ODA (Temp) --> Precision: {:.2f}, Recall: {:.2f}".format(oda_preci, oda_rec))
    # Confusion Matrix
    # predlabel = 'ADS.fAHUTempODAADSInternalValuesMirror'
    # rule_confmat(rule_preds_ODATemp, y_test, predlabel, "ODATemp")

    # oda_acc = compute_rule_acc(filter_ODATemp_preds, trueidx_ODA)
    # oda_preci = oda_acc['Precision']
    # oda_rec = oda_acc['Recall']
    # # print("ODA true idxs: ", trueidx_ODA)
    # print("ODA (Temp) after filtering out SUP preds --> Precision: {:.2f}, Recall: {:.2f}".format(oda_preci, oda_rec))
    # Confusion Matrix
    # predlabel = 'ADS.fAHUTempODAADSInternalValuesMirror'
    # rule_confmat(filter_ODATemp_preds, y_test, predlabel, "ODATemp")

    # oda_acc = compute_rule_acc(rule_preds_ODA_3, trueidx_ODA)
    # oda_preci = oda_acc['Precision']
    # oda_rec = oda_acc['Recall']
    # # print("ODA true idxs: ", trueidx_ODA)
    # print("ODA (Winter) --> Precision: {:.2f}, Recall: {:.2f}".format(oda_preci, oda_rec))

    ### ---> RULE 6 ###
    rule_preds_ODATemp_6, trueidx_ODA = delta_ODATemp_rulelogic(X_test, labellist, y_test)
    filter_ODATemp_preds = dict()
    filter_ODATemp_preds['ODA'] = [x for x in rule_preds_ODATemp_6['ODA'] if x not in rule_preds_SUPTemp['TempSUPADS']] # if a point has already been predicted as SUP Temp, do not consider it for ODA
    oda_acc = compute_rule_acc(filter_ODATemp_preds, trueidx_ODA)
    oda_preci = oda_acc['Precision']
    oda_rec = oda_acc['Recall']
    # print("ODA true idxs: ", trueidx_ODA)
    print("delta ODA (Temp) --> Precision: {:.2f}, Recall: {:.2f}".format(oda_preci, oda_rec))

    # Confusion Matrix
    # predlabel = 'ADS.fAHUTempODAADSInternalValuesMirror'
    # rule_confmat(filter_ODATemp_preds, y_test, predlabel, "TempODA")

    ### ---> RULE 12 ###
    rule_preds_PHvalve, trueidx_PH = ValveMin_winter_rulelogic(X_test, labellist, y_test)
    ph_acc = compute_rule_acc(rule_preds_PHvalve, trueidx_PH)
    ph_preci = ph_acc['Precision']
    ph_rec = ph_acc['Recall']
    print("PH (Valve) Min Logic --> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))

    ### ---> RULE 3 ###
    rule_preds_PH_3, rule_preds_ODA_3, trueidx_PH, trueidx_ODA = PHValve_winter_rulelogic(X_test, labellist, y_test, filter_ODATemp_preds_1)
    ph_acc = compute_rule_acc(rule_preds_PH_3, trueidx_PH)
    ph_preci = ph_acc['Precision']
    ph_rec = ph_acc['Recall']
    # print("PH true idxs: ", trueidx_PH)
    print("PH (Valve) Winter --> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))

    ### ---> RULE 5 ###
    # rule_preds_PH_5, rule_preds_ODA_5, trueidx_PH, trueidx_ODA= PHValve_summer_rulelogic(X_test, labellist, y_test)
    rule_preds_PH_5, rule_preds_ODA_5, trueidx_PH, trueidx_ODA= PHValve_summer_rulelogic(X_test, labellist, y_test, filter_ODATemp_preds_1)
    ph_acc = compute_rule_acc(rule_preds_PH_5, trueidx_PH)
    ph_preci = ph_acc['Precision']
    ph_rec = ph_acc['Recall']
    # print("PH true idxs: ", trueidx_PH)
    print("PH (Valve) Summer--> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))

    oda_acc = compute_rule_acc(rule_preds_ODA_5, trueidx_ODA)
    oda_preci = oda_acc['Precision']
    oda_rec = oda_acc['Recall']
    # print("ODA true idxs: ", trueidx_ODA)
    print("ODA (Summer) --> Precision: {:.2f}, Recall: {:.2f}".format(oda_preci, oda_rec))

    ### Union PH
    union_PH = dict()
    union_PH['PH'] = list(set(rule_preds_PHvalve['PH'] + rule_preds_PH_5['PH']))
    ph_acc = compute_rule_acc(union_PH, trueidx_PH)
    ph_preci = ph_acc['Precision']
    ph_rec = ph_acc['Recall']
    print("\nPH (Union) --> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))

    ## ---> RULE 4 ###
    rule_preds_PH_4, rule_preds_PHPump, trueidx_PH = PumpOperating_rulelogic(X_test, labellist, y_test, rule_preds_PH_5)
    # phpump_idx = [idx for idx, x in enumerate(y_test) if "PHPump" in x]
    rule4_PH = dict()
    rule4_PH['PH'] = list(set(rule_preds_PH_4['PH'] + rule_preds_PHPump['PH']))

    ph_acc = compute_rule_acc(rule_preds_PH_4, trueidx_PH)
    ph_preci = ph_acc['Precision']
    ph_rec = ph_acc['Recall']
    # print("PH true idxs: ", trueidx_PH)
    print("PH Valve --> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))
    ph_acc = compute_rule_acc(rule_preds_PHPump, trueidx_PH)
    ph_preci = ph_acc['Precision']
    ph_rec = ph_acc['Recall']
    print("PH Pump Operating --> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))
    # print("PH true idxs: ", trueidx_PH)
    ph_acc = compute_rule_acc(rule4_PH, trueidx_PH)
    ph_preci = ph_acc['Precision']
    ph_rec = ph_acc['Recall']
    print("PH (Pump + Valve) --> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))

    ### ---> RULE 10 ###
    rule_preds_RHPump, trueidx_RH = RHPump_rulelogic(X_test, labellist, y_test, rule_preds_PHPump)
    ph_acc = compute_rule_acc(rule_preds_RHPump, trueidx_RH)
    ph_preci = ph_acc['Precision']
    ph_rec = ph_acc['Recall']
    # print("PH true idxs: ", trueidx_PH)
    print("RH Pump Operating --> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))

    ## ---> RULE 8 ###
    rule_preds_RH, trueidx_RH = RemValve_rulelogic(X_test, labellist, y_test, rule_preds_PH_5)
    rh_acc = compute_rule_acc(rule_preds_RH, trueidx_RH)
    rh_preci = rh_acc['Precision']
    rh_rec = rh_acc['Recall']
    # print("PH true idxs: ", trueidx_PH)
    print("RH Valve --> Precision: {:.2f}, Recall: {:.2f}".format(rh_preci, rh_rec))


    

    # ### ---> RULE 8 ###
    # rule_preds_RH, trueidx_RH = RHValve_rulelogic(X_test, labellist, y_test)
    # # print(rule_preds_RH['RH'])
    # true_labels = y_test[rule_preds_RH['RH']]
    # target_names = list(set(true_labels))
    # pred_labels = ['ADS.fAHURHValveActADSInternalValuesMirror'] * len(rule_preds_RH['RH'])
    # # strip common text from labels
    # true_labels = [s.replace('ADSInternalValuesMirror','') for s in true_labels]
    # target_names = [s.replace('ADSInternalValuesMirror','') for s in target_names]
    # pred_labels = [s.replace('ADSInternalValuesMirror','') for s in pred_labels]

    # conf_mat = confusion_matrix(true_labels, pred_labels, labels=target_names)
    # conf_df = pd.DataFrame(conf_mat, index=target_names, columns=target_names)
    # print(conf_df)
    # confmat_path = os.path.join(log_dir, 'RULES_'+'confusion_mat.csv')
    # conf_df.to_csv(confmat_path)

    # rh_acc = compute_rule_acc(rule_preds_RH, trueidx_RH)
    # rh_preci = rh_acc['Precision']
    # rh_rec = rh_acc['Recall']
    # print("RH (Valve) --> Precision: {:.2f}, Recall: {:.2f}".format(rh_preci, rh_rec))

    # ### Union ODA
    # union_ODA = dict()
    # union_ODA['ODA'] = list(set(rule_preds_ODA_3['ODA'] + rule_preds_ODA_5['ODA'] + rule_preds_ODATemp_6['ODA']))
    # # print(union_ODA)
    # oda_acc = compute_rule_acc(union_ODA, trueidx_ODA)
    # oda_preci = oda_acc['Precision']
    # oda_rec = oda_acc['Recall']
    # print("\nODA Union --> Precision: {:.2f}, Recall: {:.2f}".format(oda_preci, oda_rec))

    # ### Union PH
    # union_PH = dict()
    # union_PH['PH'] = list(set(rule_preds_PH_3['PH'] + rule_preds_PH_5['PH']))
    # ph_acc = compute_rule_acc(union_PH, trueidx_PH)
    # ph_preci = ph_acc['Precision']
    # ph_rec = ph_acc['Recall']
    # print("\nPH (Union) --> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))

    # ### ---> RULE 7 ###
    # rule_preds_ETATemp, trueidx_ETA = ETATemp_rulelogic(X_test, labellist, y_test, rule_preds_SUPTemp, rule_preds_ODATemp_6)
    # print(rule_preds_ETATemp)
    # # filter_Temp_preds = dict()
    # # filter_Temp_preds['ETA'] = [x for x in rule_preds_ETATemp['ETA'] if x not in rule_preds_SUPTemp['TempSUP'] and x not in rule_preds_ODATemp_6['ODA']]
    # # print(filter_Temp_preds)
    # eta_acc = compute_rule_acc(rule_preds_ETATemp, trueidx_ETA)
    # eta_preci = eta_acc['Precision']
    # eta_rec = eta_acc['Recall']
    # print("ETA (Temp) --> Precision: {:.2f}, Recall: {:.2f}".format(eta_preci, eta_rec))

    print("\n ^^^^^^^^ Sub-systems ^^^^^^^^")
    ### Calculate "overall SUP" accuracy
    speed_sup_preds = [idx for idx, x in enumerate(labellist) if x == "Speed"]
    temp_sup_preds = list(set(rule_preds_SUPTemp['TempSUPADS'] + rule_preds_SUPTempSet['TempSUPSet']))
    valve_sup_preds = rule_preds_RH["RH"]
    oper_sup_preds = rule_preds_RHPump["RH"]
    all_SUP_preds = list(set(speed_sup_preds + temp_sup_preds + valve_sup_preds + oper_sup_preds))
    SUP_truth_idx = [idx for idx, x in enumerate(y_test[:,0]) if "SUP" in x or "RH" in x]
    sup_acc = compute_overall_acc(all_SUP_preds, SUP_truth_idx)
    sup_preci = sup_acc['Precision']
    sup_rec = sup_acc['Recall']
    # print("PH true idxs: ", trueidx_PH)
    print("Overall SUP --> Precision: {:.2f}, Recall: {:.2f}".format(sup_preci, sup_rec))

    ### Calculate "overall ODA" accuracy
    # temp_oda_preds = filter_ODATemp_preds_1['ODA']
    temp_oda_preds = rule_preds_ODA_5['ODA']
    valve_oda_preds = rule_preds_PH_5['PH']
    oper_oda_preds = rule_preds_PHPump['PH']
    all_ODA_preds = list(set(temp_oda_preds + valve_oda_preds + oper_oda_preds))
    ODA_truth_idx = [idx for idx, x in enumerate(y_test[:,0]) if "ODA" in x or "PH" in x]
    oda_acc = compute_overall_acc(all_ODA_preds, ODA_truth_idx)
    oda_preci = oda_acc['Precision']
    oda_rec = oda_acc['Recall']
    print("Overall ODA --> Precision: {:.2f}, Recall: {:.2f}".format(oda_preci, oda_rec))

    print("\n ^^^^^^^^ Components ^^^^^^^^")
    ### Calculate "Preheater" accuracy
    PH_preds = list(set(rule_preds_PHPump['PH'] + rule_preds_PH_5['PH'] + rule_preds_PHvalve['PH']))
    PH_truth_idx = [idx for idx, x in enumerate(y_test[:,0]) if "PH" in x]
    ph_acc = compute_overall_acc(PH_preds, PH_truth_idx)
    ph_preci = ph_acc['Precision']
    ph_rec = ph_acc['Recall']
    print("Overall Preheater --> Precision: {:.2f}, Recall: {:.2f}".format(ph_preci, ph_rec))

    ### Calculate "Reheater" accuracy
    RH_preds = list(set(rule_preds_RHPump['RH'] + rule_preds_RH['RH']))
    RH_truth_idx = [idx for idx, x in enumerate(y_test[:,0]) if "RH" in x]
    rh_acc = compute_overall_acc(RH_preds, RH_truth_idx)
    rh_preci = rh_acc['Precision']
    rh_rec = rh_acc['Recall']
    print("Overall Reheater --> Precision: {:.2f}, Recall: {:.2f}".format(rh_preci, rh_rec))