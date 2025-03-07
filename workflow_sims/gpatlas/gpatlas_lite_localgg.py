#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset

from pathlib import Path
from typing import cast
import h5py
import time as tm
import numpy as np

#variables

n_loci = 100000
n_alleles = 2
window_step = 10
window_stride = 10
glatent = 2000
input_length = n_loci * n_alleles
genetic_noise = 0.95
learning_rate = 0.001

n_epochs = 51
batch_size = 128
num_workers = 9
base_file_name = 'gpatlas_input/test_sim_WF_1kbt_10000n_5000000bp_'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class GenoDataset(BaseDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        strain = self.strains[idx]

        strain_data = cast(Dataset, self._strain_group[strain])

        # Note: genotype is being cast as float32 here, reasons not well understood.
        gens = torch.tensor(strain_data["genotype"][:], dtype=torch.float32).flatten()
        #print(f"Sample {idx} shape: {gens.shape}")
        return  gens


##########################################################################################
##########################################################################################

train_data_geno = GenoDataset(f''+base_file_name+'train.hdf5')
test_data_geno = GenoDataset(f''+base_file_name+'test.hdf5')

##########################################################################################
##########################################################################################

train_loader_geno = torch.utils.data.DataLoader(
    dataset=train_data_geno, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False, drop_last=False
)
test_loader_geno = torch.utils.data.DataLoader(
    dataset=test_data_geno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)

#####################################################################################################################
#####################################################################################################################

class local_gg_encoder(nn.Module):
    def __init__(self, input_length=input_length, loci_count=n_loci, window_size=window_step, latent_dim=glatent, window_stride=window_stride):
        """
        LD-aware autoencoder for genetic data using grouped convolution.
        Each window of loci is processed independently.

        Args:
            input_length: Total length of input tensor (one-hot encoded, so 2 values per locus)
            loci_count: Actual number of genetic loci (half of input_length)
            window_size: Number of loci to group together in local connections
            latent_dim: Dimension of the latent space
        """
        super().__init__()

        self.input_length = input_length  # 200,000 values for 100,000 loci
        self.loci_count = loci_count      # 100,000 loci
        self.window_size = window_size    # 10 loci per group
        self.latent_dim = latent_dim      # Latent space dimension
        self.window_stride = window_stride #steps between LD windows

        # Calculate the number of groups
        self.n_groups = loci_count // window_size  # 10,000 groups

        # Locally connected block to catch LD
        self.encoder_conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_groups,
                out_channels=self.n_groups,
                kernel_size=window_size * 2,
                stride=window_stride * 2,
                groups=self.n_groups,
                bias=True
            ),
            nn.LeakyReLU(0.1)
        )

        # Fully connected block for encoder
        self.encoder_fc_block = nn.Sequential(
            nn.Linear(self.n_groups, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.8),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Reshape to group by windows
        x = x.reshape(batch_size, self.n_groups, self.window_size * 2)
        # Apply convolutional block
        x = self.encoder_conv_block(x)
        # Flatten
        x = x.reshape(batch_size, self.n_groups)
        # Apply FC block to get latent representation
        latent = self.encoder_fc_block(x)
        return(latent)

#####################################################################################################################

class local_gg_decoder(nn.Module):
    def __init__(self, input_length=input_length, loci_count=n_loci, window_size=window_step, latent_dim=glatent, window_stride=window_stride):
        super().__init__()

        self.input_length = input_length  # 200,000 values for 100,000 loci
        self.loci_count = loci_count      # 100,000 loci
        self.window_size = window_size    # 10 loci per group
        self.latent_dim = latent_dim      # Latent space dimension
        self.window_stride = window_stride #steps between LD windows

        # Calculate the number of groups
        self.n_groups = loci_count // window_size  # 10,000 groups

        # Fully connected block for decoder
        self.decoder_fc_block = nn.Sequential(
            nn.Linear(latent_dim, self.n_groups),
            nn.BatchNorm1d(self.n_groups, momentum=0.8),
            nn.LeakyReLU(0.1)
        )

        # Locally connected block
        self.decoder_conv_block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.n_groups,
                out_channels=self.n_groups,
                kernel_size=window_size * 2,
                stride=window_stride * 2,
                groups=self.n_groups,
                bias=True
            ),
            nn.LeakyReLU(0.1)
        )
        # Final sigmoid transform
        self.decoder_final_act = nn.Sigmoid()


    def forward(self, z):
        batch_size = z.size(0)
        x = self.decoder_fc_block(z)
        x = x.reshape(batch_size, self.n_groups, 1)

        x = self.decoder_conv_block(x)
        x = x.reshape(batch_size, self.input_length)

        x = self.decoder_final_act(x)
        return x

#####################################################################################################################
#####################################################################################################################


GQ = local_gg_encoder()
GP = local_gg_decoder()

GQ.to(device)
GP.to(device)

EPS = 1e-15
#reg_lr = 0.001
adam_b = (0.5, 0.999)

optim_GQ_enc = torch.optim.Adam(GQ.parameters(), lr=learning_rate, betas=adam_b)
optim_GP_dec = torch.optim.Adam(GP.parameters(), lr=learning_rate, betas=adam_b)

############################################################################################################
############################################################################################################


g_rcon_loss = []
val_losses = []  # To store validation losses per epoch
start_time = tm.time()
validation_interval = 5
best_val_loss = float('inf')
best_epoch = 0

gen_noise = 1 - genetic_noise

print("start training")
for n in range(n_epochs):
    GQ.train()
    GP.train()

    for i, (gens) in enumerate(train_loader_geno):
        batch_size = gens.shape[0]  # redefine batch size here to allow for incomplete batches

        # reconstruction loss
        GP.zero_grad()
        GQ.zero_grad()

        #gens = gens[:, : n_loci*n_alleles]

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

        l1_reg = torch.linalg.norm(torch.sum(GQ.encoder_fc_block[0].weight, axis=0), 1)
        l2_reg = torch.linalg.norm(torch.sum(GQ.encoder_fc_block[0].weight, axis=0), 2)
        g_recon_loss = g_recon_loss + l1_reg * 0 + l2_reg * 0

        g_rcon_loss.append(float(g_recon_loss.detach()))

        g_recon_loss.backward()
        optim_GQ_enc.step()
        optim_GP_dec.step()

    ###################################
    #validation
    GQ.eval()  # Set models to evaluation mode
    GP.eval()

    val_loss_epoch = 0
    num_val_batches = 0

    with torch.no_grad():  # No need to track gradients for validation
        for i, (gens) in enumerate(test_loader_geno):  # Using test loader for validation
            batch_size = gens.shape[0]

            gens = gens.to(device)

            # Forward pass
            z_sample = GQ(gens)  # No noise during validation
            X_sample = GP(z_sample)

            # Calculate validation loss
            val_loss = F.binary_cross_entropy(X_sample + EPS, gens + EPS)

            val_loss_epoch += float(val_loss)
            num_val_batches += 1

    # Calculate average validation loss
    avg_val_loss = val_loss_epoch / num_val_batches
    val_losses.append(avg_val_loss)

    # Check if this is the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = n

    cur_time = tm.time() - start_time
    start_time = tm.time()

    print(
        "Epoch num: "
        + str(n)
        + " , training loss "
        + str(g_rcon_loss[-1])
        + " , validation_loss: "
        + str(avg_val_loss)
        + " , best validation: "
        + str(best_val_loss)
        + " , best epoch: "
        + str(best_epoch)
        + " , epoch duration: "
        + str(cur_time)
    )

torch.save(GQ.state_dict(), "localgg/localgg_GQ_encoder_state_dict.pt")
torch.save(GP.state_dict(), "localgg/localgg_GP_decoder_state_dict.pt")
