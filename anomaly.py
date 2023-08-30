import torch

from torchvision.utils import save_image
from models.entropy_loss import EntropyLossEncap, loss_function

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Instantiate the entropy loss function
tr_entropy_loss_func = EntropyLossEncap().to(device)


# Function to evaluate anomaly detection using the chosen model and anomaly data
def anomaly(model, epoch, anomaly_loader, entropy_loss_weight, batch_size):
    
    # Set the model to evaluation mode
    model.eval()
    anomaly_loss = 0
    
    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for i, (data, _) in enumerate(anomaly_loader):
            data = data.to(device)
            # Perform the forward pass through the model to obtain reconstructed output
            recon_batch, mu, logvar, att = model(data)
            # Calculate the VAE loss using the provided loss function
            lossVAE = loss_function(recon_batch, data, mu, logvar)
            # Calculate the memory loss using the entropy loss function
            lossMem = tr_entropy_loss_func(att)
            # Calculate the total loss as a combination of VAE loss and memory loss weighted by entropy_loss_weight
            loss = lossVAE + entropy_loss_weight * lossMem
            anomaly_loss += loss.item()

            # Reconstruction for anomaly data
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'results/recon_anomaly' + str(epoch) + '.png', nrow=n)

    anomaly_loss /= len(anomaly_loader)
    print('====> anomaly set loss: {:.6f}'.format(anomaly_loss/batch_size))

    return - anomaly_loss
