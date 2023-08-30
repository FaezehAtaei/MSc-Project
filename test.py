import torch

from torchvision.utils import save_image
from models.entropy_loss import EntropyLossEncap, loss_function


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tr_entropy_loss_func = EntropyLossEncap().to(device)


def test(model, epoch, test_loader, entropy_loss_weight, batch_size):
    
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    
    # Disable gradient computation during testing
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            # Forward pass through the model
            recon_batch, mu, logvar, att = model(data)
            # Calculate VAE and memory loss
            lossVAE = loss_function(recon_batch, data, mu, logvar)
            lossMem = tr_entropy_loss_func(att)
            loss = lossVAE + entropy_loss_weight * lossMem
            test_loss += loss.item()
            
            # Reconstruction for anomaly data
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'results/recon_test' + str(epoch) + '.png', nrow=n)
    
    # Calculate and print average test loss
    test_loss /= len(test_loader)
    print('====> test set loss: {:.6f}'.format(test_loss/batch_size))

    return - test_loss
