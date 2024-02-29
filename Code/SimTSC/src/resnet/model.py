import os
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import wandb
import pandas as pd

from sklearn.metrics import f1_score, classification_report, confusion_matrix

class ResNetTrainer:
    def __init__(self, device, logger):
        self.device = device
        self.logger = logger
        self.tmp_dir = 'tmp'
        self.test_flag = False
        self.wandb_logging = True
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def fit(self, model, X_train, y_train, X_val, y_val, epochs=200, batch_size=128, eval_batch_size=128):
        if self.wandb_logging:
            # Login to W<&B
            wandb.login()
        
        file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

        train_dataset = Dataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_dataset = Dataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=4e-3)

        best_acc = 0.0
        early_stopper = EarlyStopper(patience=10)

        if self.wandb_logging:
            # Initialize W&B run
            run_name = "AHU_surveyext_Nov22Feb24_5day-ResNet-standardscaler"
            run = wandb.init(
                # Set the project where this run will be logged
                project="BES Time-Series Classification",
                entity="vaibhavs",
                name=run_name,
                # Track hyperparameters and run metadata
                config={
                    "epochs": epochs
                },
            )

        # Training
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for i, inputs in enumerate(train_loader, 0):
                X, y = inputs
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = F.nll_loss(outputs, y)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            model.eval()
            acc = compute_accuracy(model, train_loader, self.device, self.test_flag)
            if acc >= best_acc:
                best_acc = acc
                torch.save(model.state_dict(), file_path)
            # validation loss
            predicted_total = []
            labels_total = []    
            val_loss = 0.0
            for i, inputs in enumerate(val_loader, 0):
                X, y = inputs
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                labels_total.extend(y)
                predicted_total.extend(predicted)
                f1_val = f1_score(labels_total, predicted_total, average='micro')
                vloss = F.nll_loss(outputs, y)
                val_loss += vloss.item()

            self.logger.log('--> Epoch {}: train loss {:5.4f}; validation loss {:5.4f}; train accuracy: {:5.4f}; best accuracy: {:5.4f}; validation F1-score: {:5.4f}'.format(epoch, train_loss, val_loss, acc, best_acc, f1_val))
            if self.wandb_logging:
                wandb.log({"train accuracy": acc, "best accuracy": best_acc, "train loss": train_loss, "validation loss": val_loss, "validation F1-score": f1_val})
            # early stopping
            if epoch > 10:    
                if early_stopper.early_stop(val_loss):             
                    break
                
        # Load the best model
        model.load_state_dict(torch.load(file_path))
        model.eval()
        os.remove(file_path)

        return model
    
    def test(self, model, X_test, y_test, batch_size=128):
        test_dataset = Dataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        acc, clf_report, conf_mat = compute_accuracy(model, test_loader, self.device, True)
        self.test_flag = False
        return acc, clf_report, conf_mat

def compute_accuracy(model, loader, device, test_flag):
    correct = 0
    total = 0
    clf_report = None
    predicted_total = []
    labels_total = []
    with torch.no_grad():
        for data in loader:
            X, labels = data
            X, labels = X.to(device), labels.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            labels_total.extend(labels)
            predicted_total.extend(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if test_flag:
        # calculate F1-score
        f1 = f1_score(labels_total, predicted_total, average='micro')
        print('--> Test F1-score {:5.4f}'.format(f1))
        # classification report
        target_names = ['Operating','Speed','Temperature', 'Valve']
        # target_names = ['Flap / Valve', 'Humidity', 'Speed', 'Temperature', 'Vdp', 'Volume']
        clf_report = classification_report(labels_total, predicted_total, target_names=target_names, output_dict=True)
        print('--> Classification Report: \n', classification_report(labels_total, predicted_total, target_names=target_names))
        conf_mat = confusion_matrix(labels_total, predicted_total)
        conf_df = pd.DataFrame(conf_mat, index=target_names, columns=target_names)
        print('--> Confusion Matrix: \n', conf_df)

    acc = correct / total
    if test_flag:
        return acc, clf_report, conf_df
    else:
        return acc
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print("early stopping counter at: ", self.counter)
            if self.counter >= self.patience:
                return True
        return False

class ResNet(nn.Module):

    def __init__(self, input_size, nb_classes):
        super(ResNet, self).__init__()
        n_feature_maps = 64

        self.block_1 = ResNetBlock(input_size, n_feature_maps)
        self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps)
        self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps)
        self.linear = nn.Linear(n_feature_maps, nb_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = F.avg_pool1d(x, x.shape[-1]).view(x.shape[0],-1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)

        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.expand = True if in_channels < out_channels else False

        self.conv_x = nn.Conv1d(in_channels, out_channels, 7, padding=3)
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, 5, padding=2)
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn_z = nn.BatchNorm1d(out_channels)

        if self.expand:
            self.shortcut_y = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_shortcut_y = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn_x(self.conv_x(x)))
        out = F.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))

        if self.expand:
            x = self.shortcut_y(x)
        x = self.bn_shortcut_y(x)
        out += x
        out = F.relu(out)

        return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


