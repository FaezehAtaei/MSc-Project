import torch

from torchvision.utils import save_image
from models.entropy_loss import EntropyLossEncap, loss_function

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tr_entropy_loss_func = EntropyLossEncap().to(device)
# Set the interval for logging during training
log_interval = 10


def train(model, epoch, train_loader, optimizer, entropy_loss_weight, batch_size):
    
    # Set the model to training mode
    model.train()
    train_loss = 0
    
    # Iterate over batches of training data
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # Forward pass through the model
        # Data is passed to the model for training
        recon_batch, mu, logvar, att = model(data)
        # Loss is calculated
        lossVAE = loss_function(recon_batch, data, mu, logvar)
        lossMem = tr_entropy_loss_func(att)
        loss = lossVAE + entropy_loss_weight * lossMem
        # Backpropagation
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

        # Reconstruction for training data
        if batch_idx == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 28,28)[:n]])
            save_image(comparison.cpu(), 'results/recon_train' + str(epoch) + '.png', nrow=n)

    # Calculate and print average training loss for the epoch
    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, (train_loss / len(train_loader))/batch_size))

    return - (train_loss / len(train_loader))
