import uuid
import os
import argparse
import pickle

import numpy as np
import pandas as pd
import torch

from src.utils import read_dataset_from_npy, Logger
from src.simtsc.model import SimTSC, SimTSCTrainer

data_dir = './tmp'
log_dir = './logs'
tmp_dir = 'tmp'

multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']

def train(X, y, train_idx, test_idx, val_idx, distances, device, logger, K, alpha):
    nb_classes = len(np.unique(y, axis=0))
    input_size = X.shape[1]

    model = SimTSC(input_size, nb_classes)
    # model = SimTSC(input_size, nb_classes, num_layers=2)
    model = model.to(device)
    trainer = SimTSCTrainer(device, logger)

    model = trainer.fit(model, X, y, train_idx, val_idx, distances, K, alpha)
    acc, clf_report, labellist = trainer.test(model, test_idx)

    return acc, clf_report

def test(modelpath, X, y, test_idx, distances, device, logger, K, alpha):
    nb_classes = len(np.unique(y, axis=0))
    input_size = X.shape[1]

    model = SimTSC(input_size, nb_classes)
    model.load_state_dict(torch.load(modelpath))
    model = model.to(device)
    trainer = SimTSCTrainer(device, logger)
    model.eval()
    acc, clf_report, labellist = trainer.predict(model, X, y, test_idx, distances, K, alpha)
    # acc, clf_report, labellist = trainer.test(model, X, y, test_idx)
    return acc, clf_report, labellist


def argsparser():
    parser = argparse.ArgumentParser("SimTSC")
    parser.add_argument('--dataset', help='Dataset name', default='Coffee')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--shot', help='shot', type=int, default=1)
    parser.add_argument('--K', help='K', type=int, default=3)
    parser.add_argument('--alpha', help='alpha', type=float, default=0.3)
    parser.add_argument('--test', help='Flag for train or test mode', type=bool, default=False)

    return parser

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    # Setup the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data and distances (DTW) files
    if args.dataset in multivariate_datasets:
        dtw_dir = os.path.join('datasets/multivariate') 
        distances = np.load(os.path.join(dtw_dir, args.dataset+'_dtw.npy'))
    else:
        dtw_dir = os.path.join(data_dir, 'ebc_dtw') 
        distances = np.load(os.path.join(dtw_dir, args.dataset+'.npy'))

    out_dir = os.path.join(log_dir, 'simtsc_log_'+str(args.shot)+'_shot'+str(args.K)+'_'+str(args.alpha))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, args.dataset+'_'+str(args.seed)+'5day_surveyext_Nov22Feb24'+'.txt')
    report_path = os.path.join(out_dir, args.dataset+'5day_surveyext_Nov22Feb24_SimTSC_'+'clf_report.csv')

    # Read data
    if args.dataset in multivariate_datasets:
        X, y, train_idx, test_idx, val_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))
    else:
        X, y, train_idx, test_idx, val_idx = read_dataset_from_npy(os.path.join(data_dir, 'ebc_'+str(args.shot)+'_shot', args.dataset+'.npy'))

    if args.test:
        test_out_dir = os.path.join(log_dir, 'TEST')
        if not os.path.exists(test_out_dir):
            os.makedirs(test_out_dir)
        test_out_path = os.path.join(test_out_dir, args.dataset+'_'+'simtsc_'+str(args.shot)+'_shot'+str(args.K)+'_'+str(args.alpha)+'.txt')
        test_report_path = os.path.join(test_out_dir, args.dataset+'_SimTSC_'+'clf_report.csv')
        preds_out_path = os.path.join(test_out_dir, args.dataset+'_'+str(args.seed)+'_preds')
        with open(test_out_path, 'w') as f:
            logger = Logger(f)
            model_dir = os.path.join(tmp_dir, str('SimTSC_rules_winter'))
            test_acc, clf_report, labellist = test(model_dir, X, y, test_idx, distances, device, logger, args.K, args.alpha)
            # Log results
            clf_report = pd.DataFrame(clf_report).transpose()
            clf_report.to_csv(test_report_path)
            logger.log('--> {} Test Accuracy: {:5.4f}'.format(args.dataset, test_acc))
            logger.log('--> Label predictions: \n{}'.format(labellist))
            with open(preds_out_path, 'wb') as fp:
                pickle.dump(labellist, fp)
    
    else:
        with open(out_path, 'w') as f:
            logger = Logger(f)

            # Train the model
            test_acc, clf_report = train(X, y, train_idx, test_idx, val_idx, distances, device, logger, args.K, args.alpha)
            clf_report = pd.DataFrame(clf_report).transpose()
            clf_report.to_csv(report_path)
            logger.log('--> {} Test Accuracy: {:5.4f}'.format(args.dataset, test_acc))
            logger.log(str(test_acc))
