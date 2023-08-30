import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt

from sklearn import metrics
from torchvision import datasets, transforms
from models.entropy_loss import EntropyLossEncap, loss_function

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tr_entropy_loss_func = EntropyLossEncap().to(device)


# download the dataset
def data_downloader(ANOMALY_TARGET, batch_size):
    # Training dataset excluding anomaly target numbers
    train_dataset = datasets.MNIST('./ODDdata', train=True, download=True, transform=transforms.ToTensor())

    train_dataset.targets = torch.tensor(train_dataset.targets)
    train_mask = (train_dataset.targets == ANOMALY_TARGET)
    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = train_dataset.targets[train_mask]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    all_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Testing dataset excluding anomaly target numbers
    test_dataset = datasets.MNIST('./ODDdata', train=False, download=True, transform=transforms.ToTensor())

    test_dataset.targets = torch.tensor(test_dataset.targets)
    test_mask = (test_dataset.targets == ANOMALY_TARGET)
    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = test_dataset.targets[test_mask]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Anomaly target numeric-only dataset
    anomaly_dataset = datasets.MNIST('./ODDdata', train=False, download=True, transform=transforms.ToTensor())

    anomaly_dataset.targets = torch.tensor(anomaly_dataset.targets)
    anomaly_mask = (anomaly_dataset.targets != ANOMALY_TARGET)
    anomaly_dataset.data = anomaly_dataset.data[anomaly_mask]
    anomaly_dataset.targets = anomaly_dataset.targets[anomaly_mask]
    anomaly_loader = torch.utils.data.DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=False)
    all_anomaly_loader = torch.utils.data.DataLoader(anomaly_dataset, batch_size=1, shuffle=False)
    
    return train_loader, anomaly_loader, test_loader, test_dataset, anomaly_dataset, all_test_loader, all_anomaly_loader
    

# returns AUC and plots the AUC at the end of traning
def plot_roc(model, test_dataset, anomaly_dataset, all_test_loader, all_anomaly_loader, entropy_loss_weight, ANOMALY_TARGET, plot):
    
    y_true = np.concatenate([np.zeros(len(test_dataset)), np.ones(len(anomaly_dataset))])
    y_score = []
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(all_test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, att = model(data)
            
            lossVAE = loss_function(recon_batch, data, mu, logvar)
            lossMem = tr_entropy_loss_func(att)
            train_score_loss = lossVAE + entropy_loss_weight * lossMem
            
            train_score_loss = train_score_loss.cpu()
            y_score.append(np.round(train_score_loss, 1).detach().numpy())
            
        for i, (data, _) in enumerate(all_anomaly_loader):
            data = data.to(device)
            recon_batch, mu, logvar, att = model(data)
            
            lossVAE = loss_function(recon_batch, data, mu, logvar)
            lossMem = tr_entropy_loss_func(att)
            anomaly_score_loss = lossVAE + entropy_loss_weight * lossMem
            
            anomaly_score_loss = anomaly_score_loss.cpu()
            y_score.append(np.round(anomaly_score_loss, 1).detach().numpy())

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    index_candidates = tpr - fpr
    index = np.where(index_candidates == max(index_candidates))[0][0]
    cutoff = thresholds[index]

    if plot:
        # Plot ROC
        plt.plot(fpr, tpr)
        plt.xlabel('FPR: False positive rate', fontsize=13)
        plt.ylabel('TPR: True positive rate', fontsize=13)
        plt.grid()
        plt.savefig('./results/roc' + str(ANOMALY_TARGET) + '.png')
        plt.close()

    return auc, - cutoff


# visualise the latent space
def plotting(step:int=0, show=False):
    
    model.eval()  # Switch the model to evaluation mode
    
    points = []
    label_idcs = []
    
    path = "./reconMem/ScatterPlots"
    if not os.path.exists(path): os.mkdir(path)
    
    for i, data in enumerate(all_Visualise_loader):
        img, label = [d.to(device) for d in data]
        # We only need to encode the validation images
        proj, k = model.encode(img)
        points.extend(proj.detach().cpu().numpy())
        label_idcs.extend(label.detach().cpu().numpy())
        del img, label
    
    points = np.array(points)
    
    # Creating a scatter plot
    fig, ax = plt.subplots(figsize=(10, 10) if not show else (8, 8))
    scatter = ax.scatter(x=points[:, 0], y=points[:, 1], s=2.0, 
                c=label_idcs, cmap='tab10', alpha=0.9, zorder=2)
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    if show: 
        ax.grid(True, color="lightgray", alpha=1.0, zorder=0)
        plt.show()
    else: 
        # Do not show but only save the plot in training
        plt.savefig(f"{path}/Step_{step:03d}.png", bbox_inches="tight")
        plt.close() # don't forget to close the plot, or it is always in memory
        

# Plot train, test, and anomaly loss at the end of training
def plot_elbo(epochs, train_elbo_list, anomaly_elbo_list, train_loss_list, test_loss_list, auc_list):
    
    # Plot ELBO
    plt.plot()
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('ELBO', fontsize=15)
    plt.plot(np.arange(epochs), train_elbo_list, color="blue", label="ELBO_Train")
    plt.plot(np.arange(epochs), anomaly_elbo_list, color="red", label="ELBO_Anomaly")
    plt.legend(loc="lower right")
    plt.savefig('./results/elbo.png')
    plt.close()
    
    # Plot loss
    plt.plot()
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.plot(np.arange(epochs), train_loss_list, color="blue", label="Train loss")
    plt.plot(np.arange(epochs), test_loss_list, color="red", label="Validation loss")
    plt.legend(loc="lower right")
    plt.savefig('./results/loss.png')
    plt.close()

    # Plot AUC
    plt.plot()
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('AUC', fontsize=15)
    plt.plot(np.arange(epochs), auc_list, color="blue", label="AUC")
    plt.legend(loc="lower right")
    plt.savefig('./results/AUC.png')
    plt.close()

