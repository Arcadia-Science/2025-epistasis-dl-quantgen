#!/usr/bin/env python3

import pickle as pk
import time as tm
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from pathlib import Path
from typing import cast
import h5py
import csv
from datetime import datetime


batch_size = 128
num_workers = 10



############

n_geno = 100000
n_alleles = 2
latent_space_g = 3500
num_epochs_gen = 13
genetic_noise = 0.60
learning_rate = 0.00135

csv_output_loss = 'learning_dynamics/test_sim_WF_1kbt_13epoch_3500latentg_loss.csv'
csv_output_dimensionality = 'learning_dynamics/test_sim_WF_1kbt_13epoch_3500latentg_dimensionality.csv'
loci_to_track = 500  # You may want to track a subset if n_geno*n_alleles is large
############


l1_lambda = 0.00000000000001
l2_lambda = 0.00000000000001
############################################################################################################
############################################################################################################
############################################################################################################

def convert_pickle_to_hdf5(pickle_path: Path, hdf5_path: Path, gzip: bool = True) -> Path:
    data = pickle.load(open(pickle_path, "rb"))
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(hdf5_path, "w") as h5f:
        metadata_group = h5f.create_group("metadata")

        loci_array = np.array(data["loci"], dtype=str_dt)
        metadata_group.create_dataset("loci", data=loci_array)

        pheno_names_array = np.array(data["phenotype_names"], dtype=str_dt)
        metadata_group.create_dataset("phenotype_names", data=pheno_names_array)

        strains_group = h5f.create_group("strains")

        for idx, strain_id in enumerate(data["strain_names"]):
            strain_grp = strains_group.create_group(strain_id)

            pheno = np.array(data["phenotypes"][idx], dtype=np.float64)
            strain_grp.create_dataset("phenotype", data=pheno)

            genotype = np.array(data["genotypes"][idx], dtype=np.int8)
            strain_grp.create_dataset(
                "genotype",
                data=genotype,
                chunks=True,
                compression="gzip" if gzip else None,
            )

        print(f"{hdf5_path} generated from {pickle_path}.")

    return hdf5_path

############################################################################################################
############################################################################################################
############################################################################################################


class GenoPhenoDataset(Dataset):
    def __init__(self, hdf5_path: Path) -> None:
        self.h5 = h5py.File(hdf5_path, "r")

        self._strain_group = cast(h5py.Group, self.h5["strains"])
        self.strains: list[str] = list(self._strain_group.keys())

    def __len__(self) -> int:
        return len(self._strain_group)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        strain = self.strains[idx]

        strain_data = cast(Dataset, self._strain_group[strain])

        # Note: genotype is being cast as float32 here, reasons not well understood.
        phenotype = torch.tensor(strain_data["phenotype"][:], dtype=torch.float32).flatten()
        genotype = torch.tensor(strain_data["genotype"][:], dtype=torch.float32)

        return phenotype, genotype


############################################################################################################
############################################################################################################
############################################################################################################


class dataset_geno(Dataset):
    """a class for importing simulated genotype-phenotype data.
    It expects a pickled object that is organized as a list of tensors:
    genotypes[n_animals, n_loci, n_alleles] (one hot at allelic state)
    gen_locs[n_animals, n_loci] (index of allelic state)
    weights[n_phens, n_loci, n_alleles] float weight for allelic contribution to phen
    phens[n_animals,n_phens] float value for phenotype
    indexes_of_loci_influencing_phen[n_phens,n_loci_ip] integer indicies of loci that influence a phenotype
    interaction_matrix[FILL THIS IN]
    pleiotropy_matrix[n_phens, n_phens, gen_index]"""

    def __init__(self, data_file, n_geno):
        self.datset = pk.load(open(data_file, "rb"))
        self.genotypes = torch.tensor(np.array(self.datset["genotypes"]), dtype=torch.float32)
        self.data_file = data_file
        self.n_geno = n_geno

    def __len__(self):
        return len(self.genotypes)

    def __getitem__(self, idx):
        genotypes = torch.flatten(self.genotypes[idx])
        return genotypes


############################################################################################################
############################################################################################################
############################################################################################################

train_data_geno = dataset_geno('test_sim_WF_1kbt_10000n_5000000bp_train.pk', n_geno=n_geno)
test_data_geno = dataset_geno('test_sim_WF_1kbt_10000n_5000000bp_test.pk', n_geno=n_geno)

############################################################################################################
############################################################################################################
############################################################################################################
#feature dimensionality functions
def calculate_feature_dimensionality_optimized(model, loci_indices=None):
    """
    Calculate D_i more efficiently using matrix operations

    Parameters:
    model: The encoder model
    loci_indices: Optional indices of loci to calculate D_i for (if None, calculate for all)

    Returns:
    Array of D_i values for the specified loci
    """
    # Extract weight matrix
    weight_matrix = model.encoder[0].weight.detach().cpu().numpy()

    # If we're only calculating for specific loci, subset the matrix
    if loci_indices is not None:
        # Get the squared norms for the selected loci
        selected_weights = weight_matrix[:, loci_indices]
    else:
        selected_weights = weight_matrix
        loci_indices = range(weight_matrix.shape[1])

    # Calculate squared norms for all selected loci
    squared_norms = np.sum(selected_weights**2, axis=0)

    # Create normalized weights for dot products
    norms = np.sqrt(squared_norms)
    normalized_weights = selected_weights / norms[np.newaxis, :]

    # Replace NaN values (from zero division) with zeros
    normalized_weights = np.nan_to_num(normalized_weights)

    # Calculate all pairwise dot products between normalized vectors at once
    # This gives a matrix of shape (n_selected, n_all)
    dot_products = normalized_weights.T @ weight_matrix

    # Square the dot products
    squared_dots = dot_products**2

    # Sum along axis 1 to get the denominator for each locus
    denominators = np.sum(squared_dots, axis=1)

    # Calculate D_i values
    D_i_values = squared_norms / denominators

    return D_i_values

############################################################################################################
############################################################################################################
############################################################################################################

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


test_loader_geno = torch.utils.data.DataLoader(
    dataset=test_data_geno, batch_size=200, num_workers=1, shuffle=True
)


train_loader_geno = torch.utils.data.DataLoader(
    dataset=train_data_geno, batch_size=200, num_workers=1, shuffle=True
)


############################################################################################################
############################################################################################################
############################################################################################################

# gencoder
class GQ_net(nn.Module):
    def __init__(self, n_loci=None, N=None):
        super().__init__()
        if N is None:
            N = latent_space_g
        if n_loci is None:
            n_loci = n_geno * n_alleles

        batchnorm_momentum = 0.8
        g_latent_dim = latent_space_g
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=N),
            nn.BatchNorm1d(N, momentum=0.8),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=g_latent_dim),
            nn.BatchNorm1d(g_latent_dim, momentum=0.8),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# gendecoder
class GP_net(nn.Module):
    def __init__(self, n_loci=None, N=None):
        super().__init__()
        if N is None:
            N = latent_space_g
        if n_loci is None:
            n_loci = n_geno * n_alleles

        batchnorm_momentum = 0.8
        g_latent_dim = latent_space_g
        self.encoder = nn.Sequential(
            nn.Linear(in_features=g_latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=n_loci),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

############################################################################################################
############################################################################################################
############################################################################################################

GQ = GQ_net()
GP = GP_net()

GQ.to(device)
GP.to(device)

EPS = 1e-15
#reg_lr = 0.001
adam_b = (0.5, 0.999)

optim_GQ_enc = torch.optim.Adam(GQ.parameters(), lr=learning_rate, betas=adam_b)
optim_GP_dec = torch.optim.Adam(GP.parameters(), lr=learning_rate, betas=adam_b)

############################################################################################################
############################################################################################################
############################################################################################################

g_rcon_loss = []
start_time = tm.time()

gen_noise = 1 - genetic_noise

csv_path = csv_output_loss
loci_indices = np.random.choice(n_geno*n_alleles, loci_to_track, replace=False)

#output for loss
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'batch', 'train_loss', 'test_loss'])

#output for dimensionality
with open(csv_output_dimensionality, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header with loci indices
    header = ['epoch', 'batch'] + [f'locus_{idx}' for idx in loci_indices]
    writer.writerow(header)



for n in range(num_epochs_gen):
    for i, (gens) in enumerate(train_loader_geno):
        batch_size = gens.shape[0]  # redefine batch size here to allow for incomplete batches

        # reconstruction loss
        GP.zero_grad()
        GQ.zero_grad()

        gens = gens[:, : n_geno*n_alleles]

        pos_noise = np.random.binomial(1, gen_noise / 2, gens.shape)

        neg_noise = np.random.binomial(1, gen_noise / 2, gens.shape)

        noise_gens = torch.tensor(
            np.where((gens + pos_noise - neg_noise) > 0, 1, 0), dtype=torch.float32
        )

        noise_gens = noise_gens.to(device)

        gens = gens.to(device)

        z_sample = GQ(noise_gens)
        X_sample = GP(z_sample)


        g_recon_loss = F.binary_cross_entropy(X_sample + EPS, gens + EPS)

        l1_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 1)
        l2_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 2)

        g_recon_loss = g_recon_loss + l1_reg * 0 + l2_reg * 0
        g_rcon_loss.append(float(g_recon_loss.detach()))

        g_recon_loss.backward()
        optim_GQ_enc.step()
        optim_GP_dec.step()

        #test eval every batch
        GP.eval()
        GQ.eval()
        g_rcon_loss_validate = []


        ###############################
        #estimate feature dimensionality
        di_values_over_time = []
            #evaluate dimensionality of features every so often
            # Calculate D_i values periodically (e.g., every 10 batches)
        if i % 10 == 0:
            # Calculate D_i for all loci
            all_di_values = calculate_feature_dimensionality_optimized(GQ, loci_indices)

            # Extract D_i for the loci we're tracking
            tracked_di_values = all_di_values

            # Store D_i values
            di_values_over_time.append((n, i, tracked_di_values))

            # Write to CSV
            with open(csv_output_dimensionality, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = [n+1, i+1] + tracked_di_values.tolist()
                writer.writerow(row)

            # Optional: Print average D_i
            print(f"Epoch {n+1}, Batch {i+1}, Avg D_i: {np.mean(tracked_di_values):.4f}")
        ###############################

        #validation test
        with torch.no_grad():
            for test_gens in test_loader_geno:
                test_gens = test_gens[:, : n_geno*n_alleles]
                test_gens = test_gens.to(device)
                z_sample = GQ(test_gens)
                X_sample = GP(z_sample)
                test_loss = F.binary_cross_entropy(X_sample + EPS, test_gens + EPS)
                g_rcon_loss_validate.append(float(test_loss)) #append batch loses

        mean_test_loss = np.mean(g_rcon_loss_validate) #calculate mean for all test batches

        GP.train()
        GQ.train()

        #save train loss output
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([n+1, i+1, float(g_recon_loss.detach()),float(mean_test_loss)])
    print(f'Done epoch {n+1}, last test loss = {mean_test_loss:.4f}')






#torch.save(GQ.state_dict(), "gpatlas/optuna/trial_GQ_encoder_state_dict.pt")
#torch.save(GQ, "gpatlas/optuna/trial_GQ_encoder_full.pt")

#GQ.load_state_dict(torch.load("GQ_encoder.pth"))
#GQ = torch.load("GQ_encoder_full.pth")
