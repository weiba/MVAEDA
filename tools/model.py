import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric import data as DATA
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse


class Classify(nn.Module):
    def __init__(self, input_dim):
        super(Classify, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return (self.net(x)).view(-1)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            # nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # return (self.net(x)).view(-1)
        return self.net(x)


class VAE_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE_Encoder, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, latent_size)
        self.sigma = torch.nn.Linear(hidden_size, latent_size)
    def forward(self, x):# x: bs,input_size
        x = F.relu(self.linear(x)) #-> bs,hidden_size
        mu = self.mu(x) #-> bs,latent_size
        sigma = self.sigma(x)#-> bs,latent_size
        return mu,sigma

class VAE_Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(VAE_Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.linear1(x)) #->bs,hidden_size
        # x = torch.sigmoid(self.linear2(x)) #->bs,output_size
        x = self.linear2(x)
        return x

class VAE_mask(torch.nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE_mask, self).__init__()
        self.encoder = VAE_Encoder(input_size, hidden_size, latent_size)
        self.decoder_mask = VAE_Decoder(latent_size+input_size, hidden_size, output_size)
        
        self.decoder = VAE_Decoder(latent_size, hidden_size, output_size)
        self.mask_predictor = nn.Linear(latent_size, input_size)
        self.mask_encoder = VAE_Encoder(input_size, hidden_size, latent_size)


    def encode(self, x):
        mu, sigma = self.encoder(x)  # mu,sigma: bs,latent_size
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)  # eps: bs,latent_size
        z = mu + eps * sigma  # z: bs,latent_size
        return z

    def loss_onlymask(self, x, mask):
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        predicted_mask = self.mask_predictor(z)
        mask_loss_weight = 0.7
        mask_loss = mask_loss_weight * bce_logits(predicted_mask, mask, reduction='mean')
        return z, mask_loss

    def loss_mask(self,y_x, x, mask):
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        predicted_mask = self.mask_predictor(z)

        re_x = self.decoder_mask(torch.cat([z, predicted_mask], dim=1)) # yuan

        masked_data_weight = 0.75
        mask_loss_weight = 0.7
        w_nums = mask * masked_data_weight + (1 - mask) * (1 - masked_data_weight)
        recon_loss = (1 - mask_loss_weight) * torch.mul(w_nums, mse(re_x, y_x, reduction='none'))
        mask_loss = mask_loss_weight * bce_logits(predicted_mask, mask, reduction='mean')
        recon_loss = recon_loss.mean()
        # recon_loss = mseloss(re_x, x)
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        alpha = 0.1
        loss = alpha * KLD + recon_loss + mask_loss
        return z, loss

    def loss_nomask(self, x):
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        re_x = self.decoder(z)  # re_x: bs,output_size
        mseloss = torch.nn.MSELoss()
        recon_loss = mseloss(re_x, x)
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        alpha = 0.1
        loss = alpha * KLD + recon_loss
        return z, loss
    # def forward_mask(self, x): #x: bs,input_size
    #     mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
    #     std = torch.exp(0.5 * sigma)
    #     eps = torch.randn_like(std)  #eps: bs,latent_size
    #     z = mu + eps*sigma  #z: bs,latent_size
    #     predicted_mask = self.mask_predictor(z)
    #     re_x = self.decoder(torch.cat([z, predicted_mask], dim=1))
    #     return re_x,z,mu,sigma,predicted_mask
