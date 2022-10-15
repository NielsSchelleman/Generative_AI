import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#Loading in the data
dataset_path = '~/datasets'
batch_size = 100

mnist_transform = transforms.Compose([
        transforms.ToTensor(),])

kwargs = {'num_workers': 1, 'pin_memory': True} 

train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True,  **kwargs)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
            
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2),  padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),  padding=1),
            nn.Flatten(),
        )
        
        self.z_mean = torch.nn.Linear(3136, 2)
        self.z_log_var = torch.nn.Linear(3136, 2)
        
        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 3136),
            Reshape([-1, 64, 7, 7]),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride = (1, 1),  padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride = (2, 2),  padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride = (2, 2),  padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride = (1, 1), padding=0),
            #Implement Trim function turning (1,29,29) into (1,28,28)
            nn.Sigmoid()
        )
    
    def reparametization(self, z_mu, z_log_var):
        eps = torch.randn_like(torch.exp(z_log_var))
        z = z_mu + torch.exp(z_log_var) * eps
        
        return z
    
    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparametization(z_mean, z_log_var)
        return encoded
    
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparametization(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return decoded, z_mean, z_log_var

vae = VAE()

lr = 1e-3

def loss_function(x, x_reconstr, mu, log_sigma):
    reconstr_loss = nn.functional.mse_loss(x_reconstr, x, reduction='sum')
    kl_loss = 0.5 * torch.sum(mu.pow(2) + (2*log_sigma).exp() - 2*log_sigma - 1)
    total_loss = reconstr_loss + kl_loss
    return total_loss, reconstr_loss, kl_loss

optimizer = Adam(vae.parameters(), lr=lr)
if __name__ == '__main__': 
    epochs = 1

    print("Start training VAE...")
    vae.train()

    for epoch in range(epochs):
        overall_loss = 0
        overall_reconstr_loss = 0
        overall_kl_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            print(x.shape)

            optimizer.zero_grad()

            x_reconstr, mu, log_sigma = vae(x)
            loss, reconstr_loss, kl_loss = loss_function(x, x_reconstr, mu, log_sigma)
            
            overall_loss += loss.item()
            overall_reconstr_loss += reconstr_loss.item()
            overall_kl_loss += kl_loss.item()
            
            loss.backward()
            optimizer.step()
            
        n_datapoints = batch_idx * batch_size
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / n_datapoints, "\tReconstruction Loss:", overall_reconstr_loss / n_datapoints, "\tKL Loss:", overall_kl_loss / n_datapoints)
        
    print("Training complete!")