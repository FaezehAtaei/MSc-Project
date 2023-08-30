from __future__ import print_function

import os
import numpy as np
import torch.utils.data

from test import test
from train import train
from anomaly import anomaly
from torch import nn, optim

from models.Mem_VAE_CIFAR import Mem_VAE_CIFAR
from models.Mem_VAE_MNIST import Mem_VAE_MNIST
from models.entropy_loss import EntropyLossEncap
from utils import plot_roc, plot_elbo, data_downloader


# Create a directory for saving the results
os.makedirs("./results", exist_ok=True)

# Define constants and hyperparameters
ANOMALY_TARGET = 0    # Anomaly number
entropy_loss_weight = 0.0002
batch_size = 32
log_interval = 10
epochs = 2

# Check and set the computation device (GPU if available, otherwise CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tr_entropy_loss_func = EntropyLossEncap().to(device)

# Instantiate the model and optimizer
model = Mem_VAE_MNIST().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Main code execution
if __name__ == "__main__":
    
    auc_list = []
    test_loss_list = []
    train_elbo_list = []
    train_loss_list = []
    anomaly_elbo_list = []
    
    train_loader, anomaly_loader, test_loader, test_dataset, anomaly_dataset, all_test_loader, all_anomaly_loader = data_downloader(ANOMALY_TARGET, batch_size)
    
#     model.load_state_dict(torch.load('./checkpoints/model_name'))

    # Main loop
    # During every epoch, the model undergoes training using a batch of data. Subsequently, the model is evaluated using anomaly         data, followed by an evaluation using normal data, at the end of each epoch the Area Under the Curve (AUC) is computed.
    for epoch in range(1, epochs + 1):
        avg_train_elbo   = train(model, epoch, train_loader, optimizer, entropy_loss_weight, batch_size)
        avg_anomaly_elbo = anomaly(model, epoch, anomaly_loader, entropy_loss_weight, batch_size)
        avg_test_elbo    = test(model, epoch, test_loader, entropy_loss_weight, batch_size)
        auc, cutoff      = plot_roc(model, test_dataset, anomaly_dataset, all_test_loader, all_anomaly_loader, entropy_loss_weight, ANOMALY_TARGET, plot=False)
        
        print("AUC:", auc)
        auc_list.append(auc)

        train_elbo_list.append(avg_train_elbo)
        anomaly_elbo_list.append(avg_anomaly_elbo)

        train_loss_list.append(-avg_train_elbo)
        test_loss_list.append(-avg_test_elbo)
    
#     torch.save(model.state_dict(), './checkpoints/model_name')

    # Plot ROC curve, AUC, and other evaluation metrics
    auc, cutoff = plot_roc(model, test_dataset, anomaly_dataset, all_test_loader, all_anomaly_loader, entropy_loss_weight, ANOMALY_TARGET, plot=True)

    plot_elbo(epochs, train_elbo_list, anomaly_elbo_list, train_loss_list, test_loss_list, auc_list)
    
    # Output Result
    print("AUC:" + str(np.round(auc, 2)))
    print("AUC:" + str(auc))
