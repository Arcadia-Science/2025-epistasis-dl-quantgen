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

#variables

n_loci = 100000
n_alleles = 2
window_step = 10
window_stride = 5
glatent = 2000
input_length = n_loci * n_alleles
n_out_channels = 3

n_epochs = 15
batch_size = 128
num_workers = 3
base_file_name = 'gpatlas_input/test_sim_WF_1kbt_10000n_5000000bp_'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"using device: {device}")
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

class LDGroupedAutoencoder(nn.Module):
    def __init__(self, input_length=input_length, loci_count=n_loci, window_size=window_step, latent_dim=glatent):
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

        # Calculate the number of groups
        self.n_groups = loci_count // window_size  # 10,000 groups

        self.input_length = input_length  # 200,000 values for 100,000 loci
        self.loci_count = loci_count      # 100,000 loci
        self.window_size = window_size    # 10 loci per group
        self.latent_dim = latent_dim      # Latent space dimension
        self.n_out_channels = n_out_channels  # number of output channels per LD window
        self.n_alleles = n_alleles

        # convolve one-hot encoded alleles into 1 feature
        self.allele_conv = nn.Conv1d(
            in_channels=self.loci_count,   # One channel per locus
            out_channels=self.loci_count,  # 100,000 output loci
            kernel_size=n_alleles,                 # 2 position per one-hot encoded locus
            stride=n_alleles,                      # Move to next one-hot pair
            groups=self.loci_count,        # Each locus processed independently
            bias=True
        )

        self.allele_conv_act = nn.LeakyReLU(0.1)

        #local LD based convolution
        self.encoder_conv = nn.Conv1d(
            in_channels=self.n_groups,           # One channel per window
            out_channels=self.n_groups*n_out_channels,          # One output per window
            kernel_size=window_size,         # Cover entire window (2 alleles per locus)
            stride=window_stride,              # Non-overlapping
            groups=self.n_groups,                # Each window processed independently
            bias=True
        )

        self.encoder_conv_act = nn.LeakyReLU(0.1)

        # Fully connected layer to latent space
        self.encoder_fc = nn.Linear(self.n_groups*n_out_channels, latent_dim)
        self.encoder_fc_act = nn.LeakyReLU(0.1)

        ################    Latent space    ################

        # Decoder - mirror of encoder
        self.decoder_fc = nn.Linear(latent_dim, self.n_groups*n_out_channels)
        self.decoder_fc_act = nn.LeakyReLU(0.1)

        # Expand each window back to original size
        self.decoder_conv = nn.ConvTranspose1d(
            in_channels=self.n_out_channels * self.n_groups,
            out_channels=self.n_groups,
            kernel_size=window_size,
            stride=window_stride,
            groups=self.n_groups,
            bias=True
        )

        self.decoder_conv_act = nn.LeakyReLU(0.1)

        # expand to one hot encoded state
        self.allele_deconv = nn.ConvTranspose1d(
            in_channels=self.loci_count,   # One channel per locus
            out_channels=self.loci_count,  # Output channel per locus
            kernel_size=n_alleles,                 # Expand each value to two values
            stride=n_alleles,                      # Match stride from encoder
            groups=self.loci_count,        # Each locus processed independently
            bias=True
        )

        #self.allele_deconv_act = nn.LeakyReLU(0.1)
        self.final_act = nn.Sigmoid()

    def encode(self, x):
        """
        Encode data to latent space

        Args:
            x: Input tensor of shape [batch_size, input_length]

        Returns:
            Latent representation of shape [batch_size, latent_dim]
        """
        batch_size = x.size(0)

        # 1. Reshape to process each locus
        x = x.reshape(batch_size, self.loci_count, self.n_alleles)

        # 2. Apply locus-level convolution
        x = self.allele_conv(x)
        x = self.allele_conv_act(x)
        x = x.squeeze(-1)  # Remove the last dimension

        # 3. Reshape for window-based processing
        x = x.reshape(batch_size, self.n_groups, self.window_size)

        # 4. Apply window-based convolution
        x = self.encoder_conv(x)
        x = self.encoder_conv_act(x)

        # 5. Flatten and map to latent space
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels)
        x = self.encoder_fc(x)
        x = self.encoder_fc_act(x)

        return x

    def decode(self, z):
        """
        Decode from latent space

        Args:
            z: Latent representation of shape [batch_size, latent_dim]

        Returns:
            Reconstructed tensor of shape [batch_size, input_length]
        """
        batch_size = z.size(0)

        # 1. Map from latent space
        x = self.decoder_fc(z)
        x = self.decoder_fc_act(x)

        # 2. Reshape for transposed convolution
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels, 1)

        # 3. Expand each window back to original size
        x = self.decoder_conv(x)
        x = self.decoder_conv_act(x)

        # 4. Reshape to loci format
        x = x.reshape(batch_size, self.loci_count, 1)

        # 5. Expand each locus back to one-hot encoding
        x = self.allele_deconv(x)
        #x = self.allele_deconv_act(x)

        # 6. Reshape to original format and apply sigmoid
        x = x.reshape(batch_size, self.input_length)
        x = self.final_act(x)

        return x

    def forward(self, x):
        """
        Forward pass through the autoencoder

        Args:
            x: Input tensor of shape [batch_size, input_length]

        Returns:
            Reconstructed tensor of shape [batch_size, input_length]
        """
        # Simply use encode and decode sequentially
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
#####################################################################################################################
#####################################################################################################################
start_time = tm.time()
def train_baseline_model(model, train_loader, test_loader=None, epochs=n_epochs,
                         learning_rate=0.001, weight_decay=1e-5, device=device):
    """
    Train the baseline LD-aware autoencoder with no special weighting

    Args:
        model: The autoencoder model
        train_loader: DataLoader with training data
        test_loader: Optional DataLoader with test data
        epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        weight_decay: Weight decay for regularization
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    global start_time
    # Initialize optimizer with proper weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Track metrics
    history = {
        'train_loss': [],
        'test_loss': [],
    }

    # Training loop
    for epoch in range(epochs):

        # Training
        model.train()
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            # Get data
            if isinstance(data, (list, tuple)):
                            # If it's a tuple or list, take the first element
                            data = data[0]

            data = data.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)

            # Standard BCE loss - NO WEIGHTING
            loss = F.binary_cross_entropy(output, data)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'Epoch: {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}')

        # Validation
        if test_loader is not None:
            model.eval()
            test_loss = 0

            with torch.no_grad():
                for data in test_loader:
                    if isinstance(data, (list, tuple)):
                    # If it's a tuple or list, take the first element
                        data = data[0]
                    data = data.to(device)

                    output = model(data)
                    # Standard BCE loss for evaluation
                    test_loss += F.binary_cross_entropy(output, data).item()

            avg_test_loss = test_loss / len(test_loader)
            history['test_loss'].append(avg_test_loss)
            cur_time = tm.time() - start_time
            start_time = tm.time()

            print(f'Epoch: {epoch+1}/{epochs}, Test Loss: {avg_test_loss:.6f}, Epoch time: {cur_time}')

            # Update learning rate
            scheduler.step(avg_test_loss)

    return model, history

#####################################################################################################################
#####################################################################################################################
#for saving encoder for G->P

class LDEncoder(nn.Module):
    def __init__(self, autoencoder):
        """
        Extract the encoder part of the trained LDGroupedAutoencoder

        Args:
            autoencoder: Trained LDGroupedAutoencoder model
        """
        super().__init__()

        # Copy all encoder-related parameters from the trained autoencoder
        self.input_length = autoencoder.input_length
        self.loci_count = autoencoder.loci_count
        self.window_size = autoencoder.window_size
        self.latent_dim = autoencoder.latent_dim
        self.n_groups = autoencoder.n_groups
        self.n_out_channels = autoencoder.n_out_channels
        self.n_alleles = autoencoder.n_alleles

        # Copy the encoder layers
        self.allele_conv = autoencoder.allele_conv
        self.allele_conv_act = autoencoder.allele_conv_act
        self.encoder_conv = autoencoder.encoder_conv
        self.encoder_conv_act = autoencoder.encoder_conv_act
        self.encoder_fc = autoencoder.encoder_fc
        self.encoder_fc_act = autoencoder.encoder_fc_act

    def forward(self, x):
        """
        Encode data to latent space

        Args:
            x: Input tensor of shape [batch_size, input_length]

        Returns:
            Latent representation of shape [batch_size, latent_dim]
        """
        batch_size = x.size(0)

        # 1. Reshape to process each locus
        x = x.reshape(batch_size, self.loci_count, self.n_alleles)

        # 2. Apply locus-level convolution
        x = self.allele_conv(x)
        x = self.allele_conv_act(x)
        x = x.squeeze(-1)  # Remove the last dimension

        # 3. Reshape for window-based processing
        x = x.reshape(batch_size, self.n_groups, self.window_size)

        # 4. Apply window-based convolution
        x = self.encoder_conv(x)
        x = self.encoder_conv_act(x)

        # 5. Flatten and map to latent space
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels)
        x = self.encoder_fc(x)
        x = self.encoder_fc_act(x)


#####################################################################################################################
#####################################################################################################################

model = LDGroupedAutoencoder(
    input_length=input_length,
    loci_count=n_loci,
    window_size=window_step,
    latent_dim=glatent)

#train and save full model
model, history = train_baseline_model(model, train_loader_geno,test_loader=test_loader_geno, device=device)
torch.save(model.state_dict(), "localgg/localgg_autenc_1kbt_V2_state_dict.pt")

#save gg encoder only for G->P
encoder = LDEncoder(model)
torch.save(encoder.state_dict(), "localgg/localgg_enc_1kbt_V2_state_dict.pt")