#!/usr/bin/env python3
import sys
import time as tm
import argparse
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data.dataset import Dataset
import optuna
from pathlib import Path
from typing import cast
import h5py
import pickle
import json
from datetime import datetime
import traceback


batch_size = 128
num_workers = 4

##########################################################################################
##########################################################################################
n_phen = 25
n_geno = 100000
n_alleles = 2
latent_space_g = 2449

n_loci = n_geno * n_alleles

g_params = {
    'input_length': 200000,  # n_loci * n_alleles
    'loci_count': 100000,
    'window_size': 100,
    'latent_space_g': 2449,
    'n_out_channels': 7,
    'window_stride': 12
}


latent_space_p = 750
learning_rate = 0.000079
phen_noise = 0.003
gen_noise = 0.99
num_epochs_pp = 10
num_epochs_gp = 15
weights_regularization = 0
##########################################################################################
##########################################################################################

class BaseDataset(Dataset):
    def __init__(self, hdf5_path: Path) -> None:
        self.h5 = h5py.File(hdf5_path, "r")

        self._strain_group = cast(h5py.Group, self.h5["strains"])
        self.strains: list[str] = list(self._strain_group.keys())

    def __len__(self) -> int:
        return len(self._strain_group)

###########
class GenoPhenoDataset(BaseDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        strain = self.strains[idx]

        strain_data = cast(Dataset, self._strain_group[strain])

        # Note: genotype is being cast as float32 here, reasons not well understood.
        phens = torch.tensor(strain_data["phenotype"][:], dtype=torch.float32)
        gens = torch.tensor(strain_data["genotype"][:], dtype=torch.float32).flatten()

        return phens, gens

class PhenoDataset(BaseDataset):
    def __getitem__(self, idx: int):
        strain = self.strains[idx]

        strain_data = cast(Dataset, self._strain_group[strain])

        # Note: genotype is being cast as float32 here, reasons not well understood.
        phens = torch.tensor(strain_data["phenotype"][:], dtype=torch.float32)


        return phens

class GenoDataset(BaseDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        strain = self.strains[idx]

        strain_data = cast(Dataset, self._strain_group[strain])

        # Note: genotype is being cast as float32 here, reasons not well understood.
        gens = torch.tensor(strain_data["genotype"][:], dtype=torch.float32).flatten()

        return  gens


##########################################################################################
##########################################################################################
train_data_gp = GenoPhenoDataset('gpatlas_input/test_sim_WF_10kbt_10000n_5000000bp_train.hdf5')
test_data_gp = GenoPhenoDataset('gpatlas_input/test_sim_WF_10kbt_10000n_5000000bp_test.hdf5')

train_data_pheno = PhenoDataset('gpatlas_input/test_sim_WF_10kbt_10000n_5000000bp_train.hdf5')
test_data_pheno = PhenoDataset('gpatlas_input/test_sim_WF_10kbt_10000n_5000000bp_test.hdf5')

##########################################################################################
##########################################################################################

train_loader_pheno = torch.utils.data.DataLoader(
    dataset=train_data_pheno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
test_loader_pheno = torch.utils.data.DataLoader(
    dataset=test_data_pheno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)

train_loader_gp = torch.utils.data.DataLoader(
    dataset=train_data_gp, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
test_loader_gp = torch.utils.data.DataLoader(
    dataset=test_data_gp, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
##########################################################################################
##########################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################################
##########################################################################################

# encoder
class Q_net(nn.Module):
    def __init__(self, phen_dim=None, N=None):
        super().__init__()
        #if N is None:
        #    N = p_latent_space
        if phen_dim is None:
            phen_dim = n_phen

        batchnorm_momentum = 0.8
        latent_dim = N
        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# decoder
class P_net(nn.Module):
    def __init__(self, phen_dim=None, N=None):
        #if N is None:
        #    N = p_latent_space
        if phen_dim is None:
            phen_dim = n_phen

        out_phen_dim = n_phen
        #vabs.n_locs * vabs.n_alleles
        latent_dim = N

        batchnorm_momentum = 0.8

        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=out_phen_dim),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
##########################################################################################
##########################################################################################

class LDEncoder(nn.Module):
    def __init__(self, input_length, loci_count, window_size, latent_dim, n_out_channels, window_stride):
        """
        Standalone encoder with the same architecture as in LDGroupedAutoencoder

        Args:
            input_length: Total length of input tensor (2 values per locus)
            loci_count: Number of genetic loci (half of input_length)
            window_size: Number of loci per window
            latent_dim: Dimension of the latent space
            n_out_channels: Number of output channels per window
            window_stride: Stride for the convolution
        """
        super().__init__()

        self.input_length = input_length
        self.loci_count = loci_count
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.n_out_channels = n_out_channels
        self.n_groups = loci_count // window_size

        # Encoder layers
        self.encoder_conv = nn.Conv1d(
            in_channels=self.n_groups,           # One channel per window
            out_channels=self.n_groups*n_out_channels,          # One output per window
            kernel_size=window_size * 2,         # Cover entire window (2 alleles per locus)
            stride=window_stride,              # Non-overlapping
            groups=self.n_groups,                # Each window processed independently
            bias=True
        )

        self.encoder_act = nn.LeakyReLU(0.1)

        # Fully connected layer to latent space
        self.encoder_fc = nn.Linear(self.n_groups*n_out_channels, latent_dim)
        self.encoder_fc_act = nn.LeakyReLU(0.1)

    def forward(self, x):
        """Forward pass through the encoder"""
        batch_size = x.size(0)

        # Reshape to group by windows
        x = x.reshape(batch_size, self.n_groups, self.window_size * 2)

        # Apply convolution
        x = self.encoder_conv(x)
        x = self.encoder_act(x)

        # Flatten and map to latent space
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels)
        x = self.encoder_fc(x)
        return self.encoder_fc_act(x)

# Load the saved encoder
def load_pretrained_encoder(encoder_path, params):
    encoder = LDEncoder(
        input_length=params['input_length'],
        loci_count=params['loci_count'],
        window_size=params['window_size'],
        latent_dim=params['latent_space_g'],
        n_out_channels=params['n_out_channels'],
        window_stride=params.get('window_stride', 5)
    )

    # Load state dict
    state_dict = torch.load(encoder_path)
    encoder.load_state_dict(state_dict)

    return encoder

GQ = load_pretrained_encoder("localgg/localgg_enc_1kbt_V1_state_dict.pt", g_params)

GQ.to(device)
GQ.requires_grad_(False)  # freeze weights
GQ.eval()

##########################################################################################
##########################################################################################
#g to p feed forward network

class GQ_to_P_net(nn.Module):
    def __init__(self, N, latent_space_g ):
        super().__init__()

        batchnorm_momentum =0.8
        g_latent_dim = latent_space_g
        latent_dim = N
        self.encoder = nn.Sequential(
            nn.Linear(in_features=g_latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


##########################################################################################
##########################################################################################


#Hyperparameters fixed


# Constants
EPS = 1e-15
adam_b = (0.5, 0.999)
n_loci = n_geno * n_alleles

# Initialize models
Q = Q_net(phen_dim=n_phen, N = latent_space_p).to(device)
P = P_net(phen_dim=n_phen, N = latent_space_p).to(device)

# Optimizers
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=learning_rate, betas=adam_b)
optim_P_dec = torch.optim.Adam(P.parameters(), lr=learning_rate, betas=adam_b)


##########################################################################################
# Training loop phenotype autoencoder
for epoch in range(num_epochs_pp):
    Q.train()
    P.train()

    epoch_losses = []

    for i, phens in enumerate(train_loader_pheno):
        phens = phens.to(device)
        batch_size = phens.shape[0]

        P.zero_grad()
        Q.zero_grad()

        # Apply noise
        noise_phens = phens + (phen_noise**0.5) * torch.randn(phens.shape).to(device)

        z_sample = Q(noise_phens)
        X_sample = P(z_sample)

        recon_loss = F.l1_loss(X_sample + EPS, phens[:, :n_phen] + EPS)

        # L1 and L2 regularization
        l1_reg = torch.linalg.norm(torch.sum(Q.encoder[0].weight, axis=0), 1)
        l2_reg = torch.linalg.norm(torch.sum(Q.encoder[0].weight, axis=0), 2)

        recon_loss = recon_loss + l1_reg * weights_regularization + l2_reg * weights_regularization

        recon_loss.backward()
        optim_Q_enc.step()
        optim_P_dec.step()

        epoch_losses.append(float(recon_loss.detach()))
        #print(f"Epoch {epoch}, train Loss pp: {recon_loss:.4f}")
##########################################################################################





#load optimizer gencoderm shouldn't be needed since frozen
#optim_GQ_enc = torch.optim.Adam(GQ.parameters(), lr=learning_rate, betas=adam_b)

GQP = GQ_to_P_net(N=latent_space_p, latent_space_g=latent_space_g).to(device)
optim_GQP_dec = torch.optim.Adam(GQP.parameters(), lr=learning_rate, betas=adam_b)

#Training loop GP network
P.eval() #set pheno decoder to eval only

for epoch_gp in range(num_epochs_gp):
    GQP.train()
    epoch_losses_gp = []

    for i, (phens, gens) in enumerate(train_loader_gp):
        phens = phens.to(device)
        gens = gens[:, : n_geno * n_alleles]

        pos_noise = np.random.binomial(1, gen_noise / 2, gens.shape)
        neg_noise = np.random.binomial(1, gen_noise / 2, gens.shape)
        noise_gens = torch.tensor(
            np.where((gens + pos_noise - neg_noise) > 0, 1, 0), dtype=torch.float32
        )

        noise_gens = gens
        noise_gens = noise_gens.to(device)

        batch_size = phens.shape[0]

        GQP.zero_grad()

        z_sample = GQ(noise_gens)
        z_sample = GQP(z_sample)
        X_sample = P(z_sample)


        g_p_recon_loss = F.l1_loss(X_sample + EPS, phens[:, :n_phen] + EPS)

        l1_reg = torch.linalg.norm(torch.sum(GQP.encoder[0].weight, axis=0), 1)
        l2_reg = torch.linalg.norm(torch.sum(GQP.encoder[0].weight, axis=0), 2)
        g_p_recon_loss = g_p_recon_loss + l1_reg * weights_regularization + l2_reg * weights_regularization

        g_p_recon_loss.backward()
        #optim_P_dec.step()
        #optim_GQ_enc.step()
        optim_GQP_dec.step()

# Test set evaluation GP
    if epoch_gp % 1 == 0:
        GQ.eval()
        GQP.eval()
        P.eval()
        test_losses_gp = []

        with torch.no_grad():
            for phens, gens in test_loader_gp:
                phens = phens.to(device)
                gens = gens.to(device)

                z_sample = GQ(gens)
                z_sample = GQP(z_sample)
                X_sample = P(z_sample)

                test_loss = F.l1_loss(X_sample + EPS, phens[:, :n_phen] + EPS)
                test_losses_gp.append(float(test_loss))

        avg_test_loss_gp = np.mean(test_losses_gp)
        print(f"Epoch {epoch_gp}, Test Loss gp: {avg_test_loss_gp:.4f}")


torch.save(P.state_dict(), "localgg/localgg_P_1kbt_V1_state_dict.pt")
torch.save(GQP.state_dict(), "localgg/localgg_GQP_1kbt_V1_state_dict.pt")


##########################################################################################
##########################################################################################
