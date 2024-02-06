import os
import argparse

import numpy as np

from src.utils import read_Y

dataset_dir = './datasets/EBC'
output_dir = './tmp'

def argsparser():
    parser = argparse.ArgumentParser("Define edges for graph")
    parser.add_argument('--dataset', help='Dataset name', default='Coffee')

    return parser

def get_edges(Y):
    Y = Y.copy(order='C').astype(np.str)
    edges = np.zeros((Y.shape[0], Y.shape[0]), dtype=np.float64)
    flag_edges = np.full((Y.shape[0], Y.shape[0]), False, dtype=bool)

    for i in range(len(Y)):
        for j in range(len(Y)):
            if Y[i] == 'ADS.fAHUFlapETAActADSInternalValuesMirror' and Y[j] == 'ADS.fAHUFlapETASetADSInternalValuesMirror':
                edges[i,j] = 1
                flag_edges[i,j] = True
            elif Y[i] == 'ADS.fAHUFlapEHAActADSInternalValuesMirror' and Y[j] == 'ADS.fAHUFlapEHASetADSInternalValuesMirror':
                edges[i,j] = 1
                flag_edges[i,j] = True
            elif Y[i] == 'ADS.fAHUFlapODAActADSInternalValuesMirror' and Y[j] == 'ADS.fAHUFlapODASetADSInternalValuesMirror':
                edges[i,j] = 1
                flag_edges[i,j] = True
            elif Y[i] == 'ADS.fAHUFlapSUPActADSInternalValuesMirror' and Y[j] == 'ADS.fAHUFlapSUPSetADSInternalValuesMirror':
                edges[i,j] = 1
                flag_edges[i,j] = True
            elif Y[i] == 'ADS.fAHUFanETASpeedActADSInternalValuesMirror' and Y[j] == 'ADS.fAHUFanETASpeedSetADSInternalValuesMirror':
                edges[i,j] = 1
                flag_edges[i,j] = True
            elif Y[i] == 'ADS.fAHUFanSUPSpeedActADSInternalValuesMirror' and Y[j] == 'ADS.fAHUFanSUPSpeedSetADSInternalValuesMirror':
                edges[i,j] = 1
                flag_edges[i,j] = True
            elif Y[i] == 'ADS.fAHUHRBypValveAct1ADSInternalValuesMirror' and Y[j] == 'ADS.fAHUHRBypValveSetADSInternalValuesMirror':
                edges[i,j] = 1
                flag_edges[i,j] = True
            elif Y[i] == 'ADS.fAHUTempODAADSInternalValuesMirror' and Y[j] == 'ADS.fAHUPHTempAirOutADSInternalValuesMirror':
                edges[i,j] = 1
                flag_edges[i,j] = True
            elif Y[i] == Y[j]:
                edges[i,j] = 1
                flag_edges[i,j] = True
    return edges

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    result_dir = os.path.join(output_dir, 'ebc_dtw')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    Y = read_Y(dataset_dir, args.dataset)

    graph_edges = get_edges(Y)
    np.save(os.path.join(result_dir, args.dataset), graph_edges)