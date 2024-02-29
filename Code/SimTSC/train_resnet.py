import os
import argparse

import numpy as np
import pandas as pd
import torch

from src.utils import read_dataset_from_npy, Logger
from src.resnet.model import ResNet, ResNetTrainer

data_dir = './tmp'
log_dir = './logs'

multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']

def train(X_train, y_train, X_test, y_test, X_val, y_val, device, logger):
    nb_classes = len(np.unique(np.concatenate((y_train, y_test, y_val), axis=0)))

    input_size = X_train.shape[1]
    model = ResNet(input_size, nb_classes)
    model = model.to(device)
    trainer = ResNetTrainer(device, logger)

    model = trainer.fit(model, X_train, y_train, X_val, y_val)
    acc, clf_report, conf_df = trainer.test(model, X_test, y_test)

    return acc, clf_report, conf_df

def argsparser():
    parser = argparse.ArgumentParser("Timeseries classification")
    parser.add_argument('--dataset', help='Dataset name', default='Coffee')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--shot', help='shot', type=int, default=1)

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

    log_dir = os.path.join(log_dir, 'resnet_log_'+str(args.shot)+'_shot')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    out_path = os.path.join(log_dir, args.dataset+'_5day_surveyext_Nov22Feb24_stanscaler'+str(args.seed)+'.txt')
    report_path = os.path.join(log_dir, args.dataset+'5day_surveyext_Nov22Feb24_ResNet_stanscaler'+'clf_report.csv')
    confmat_path = os.path.join(log_dir, args.dataset+'5day_surveyext_Nov22Feb24_ResNet_stanscaler'+'confusion_mat.csv')

    with open(out_path, 'w') as f:
        logger = Logger(f)
        # Read data
        if args.dataset in multivariate_datasets:
            X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))
        else:
            X, y, train_idx, test_idx, val_idx = read_dataset_from_npy(os.path.join(data_dir, 'ebc_'+str(args.shot)+'_shot', args.dataset+'.npy'))

        # Train the model
        test_acc, clf_report, conf_mat = train(X[train_idx], y[train_idx], X[test_idx], y[test_idx], X[val_idx], y[val_idx], device, logger)
        clf_report = pd.DataFrame(clf_report).transpose()
        clf_report.to_csv(report_path)
        conf_mat.to_csv(confmat_path)
        logger.log('--> {} Test Accuracy: {:5.4f}'.format(args.dataset, test_acc))
        logger.log(str(test_acc))
