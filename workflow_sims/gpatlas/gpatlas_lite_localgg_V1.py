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

#variables

n_loci = 100000
n_alleles = 2
window_step = 10
window_stride = 5
glatent = 10000
input_length = n_loci * n_alleles

n_epochs = 15
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

        self.input_length = input_length  # 200,000 values for 100,000 loci
        self.loci_count = loci_count      # 100,000 loci
        self.window_size = window_size    # 10 loci per group
        self.latent_dim = latent_dim      # Latent space dimension

        # Calculate the number of groups
        self.n_groups = loci_count // window_size  # 10,000 groups

        # Encoder layers
        self.encoder_conv = nn.Conv1d(
            in_channels=self.n_groups,           # One channel per window
            out_channels=self.n_groups,          # One output per window
            kernel_size=window_size * 2,         # Cover entire window (2 alleles per locus)
            stride=window_stride,              # Non-overlapping
            groups=self.n_groups,                # Each window processed independently
            bias=True
        )

        self.encoder_act = nn.LeakyReLU(0.2)

        # Fully connected layer to latent space
        self.encoder_fc = nn.Linear(self.n_groups, latent_dim)
        self.encoder_fc_act = nn.LeakyReLU(0.1)

        # Decoder - mirror of encoder
        self.decoder_fc = nn.Linear(latent_dim, self.n_groups)
        self.decoder_fc_act = nn.LeakyReLU(0.1)

        # Expand each window back to original size
        self.decoder_conv = nn.ConvTranspose1d(
            in_channels=self.n_groups,
            out_channels=self.n_groups,
            kernel_size=window_size * 2,
            stride=window_stride,
            groups=self.n_groups,
            bias=True
        )

        self.final_act = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the autoencoder

        Args:
            x: Input tensor of shape [batch_size, input_length]

        Returns:
            Reconstructed tensor of shape [batch_size, input_length]
        """
        # Input shape: [batch_size, input_length]
        #print(f"Input shape: {x.shape}")
        #print(f"n_groups: {self.n_groups}, window_size: {self.window_size}")


        batch_size = x.size(0)
        #print(f"Attempting to reshape to: {batch_size}, {self.n_groups}, {self.window_size * 2}")

        # Reshape to group by windows
        # [batch_size, input_length] -> [batch_size, n_groups, window_size*2]
        x = x.reshape(batch_size, self.n_groups, self.window_size * 2)

        # Apply grouped convolution - each window processed independently
        x = self.encoder_conv(x)
        x = self.encoder_act(x)

        # Flatten to [batch_size, n_groups]
        x = x.reshape(batch_size, self.n_groups)

        # Map to latent space
        latent = self.encoder_fc(x)
        latent = self.encoder_fc_act(latent)

        # Decode from latent space
        x = self.decoder_fc(latent)
        x = self.decoder_fc_act(x)

        # Reshape for transposed convolution
        # [batch_size, n_groups] -> [batch_size, n_groups, 1]
        x = x.reshape(batch_size, self.n_groups, 1)

        # Expand each window back to original size
        x = self.decoder_conv(x)

        # Reshape to original format
        # [batch_size, n_groups, window_size*2] -> [batch_size, input_length]
        x = x.reshape(batch_size, self.input_length)

        # Apply sigmoid
        x = self.final_act(x)

        return x

    def encode(self, x):
        """
        Encode data to latent space

        Args:
            x: Input tensor of shape [batch_size, input_length]

        Returns:
            Latent representation of shape [batch_size, latent_dim]
        """
        batch_size = x.size(0)

        # Reshape to group by windows
        x = x.reshape(batch_size, self.n_groups, self.window_size * 2)

        # Apply convolution
        x = self.encoder_conv(x)
        x = self.encoder_act(x)

        # Flatten and map to latent space
        x = x.reshape(batch_size, self.n_groups)
        x = self.encoder_fc(x)
        return self.encoder_fc_act(x)

    def decode(self, z):
        """
        Decode from latent space

        Args:
            z: Latent representation of shape [batch_size, latent_dim]

        Returns:
            Reconstructed tensor of shape [batch_size, input_length]
        """
        batch_size = z.size(0)

        # Map from latent space to window representations
        x = self.decoder_fc(z)
        x =  self.decoder_fc_act(x)

        # Reshape for transposed convolution
        x = x.reshape(batch_size, self.n_groups, 1)

        # Expand each window back to original size
        x = self.decoder_conv(x)

        # Reshape to original format
        x = x.reshape(batch_size, self.input_length)

        # Apply sigmoid
        x = self.final_act(x)

        return x

#####################################################################################################################
#####################################################################################################################

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
        'epoch_time': []
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

            # Print the actual batch shape for debugging
            #print(f"Batch shape: {data.shape}")




            # Forward pass
            optimizer.zero_grad()
            output = model(data)

            # Standard BCE loss - NO WEIGHTING
            loss = F.binary_cross_entropy(output, data)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}, Loss: {loss.item():.6f}')

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'Epoch: {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}')

        # Validation
        if test_loader is not None:
            model.eval()
            test_loss = 0

            with torch.no_grad():
                for data in test_loader:
                    data = data[0].to(device)
                    output = model(data)
                    # Standard BCE loss for evaluation
                    test_loss += F.binary_cross_entropy(output, data).item()

            avg_test_loss = test_loss / len(test_loader)
            history['test_loss'].append(avg_test_loss)
            print(f'Epoch: {epoch+1}/{epochs}, Test Loss: {avg_test_loss:.6f}')

            # Update learning rate
            scheduler.step(avg_test_loss)

    return model, history

#####################################################################################################################
#####################################################################################################################

def evaluate_allele_reconstruction(model, test_loader, device=device):
    """
    Evaluate model performance specifically for allele reconstruction

    Args:
        model: Trained autoencoder model
        test_loader: DataLoader with test data
        device: Device to evaluate on

    Returns:
        Dictionary with various metrics
    """
    model.eval()

    # Metrics to track
    metrics = {
        'overall_accuracy': 0,
        'rare_allele_accuracy': 0,
        'common_allele_accuracy': 0,
        'rare_allele_recall': 0,
        'total_samples': 0,
        'rare_allele_count': 0,
        'common_allele_count': 0
    }

    with torch.no_grad():
        #for data in test_loader:
        for batch_idx, data in enumerate(test_loader):
            # Get data
            if isinstance(data, (list, tuple)):
                # If it's a tuple or list, take the first element
                data = data[0]
            output = model(data)

            # Convert probabilities to binary predictions
            predictions = (output > 0.5).float()

            # Calculate metrics
            correct = (predictions == data).float()

            # Count rare and common alleles
            rare_allele_mask = (data > 0.5)
            common_allele_mask = (data <= 0.5)

            metrics['total_samples'] += data.size(0) * data.size(1)
            metrics['rare_allele_count'] += rare_allele_mask.sum().item()
            metrics['common_allele_count'] += common_allele_mask.sum().item()

            # Accuracy metrics
            metrics['overall_accuracy'] += correct.sum().item()
            metrics['rare_allele_accuracy'] += (correct * rare_allele_mask).sum().item()
            metrics['common_allele_accuracy'] += (correct * common_allele_mask).sum().item()

            # Recall for rare alleles (true positives / total positives)
            true_positives = (predictions * data * rare_allele_mask).sum().item()
            metrics['rare_allele_recall'] += true_positives

    # Normalize metrics
    metrics['overall_accuracy'] /= metrics['total_samples']
    metrics['rare_allele_accuracy'] /= max(1, metrics['rare_allele_count'])
    metrics['common_allele_accuracy'] /= max(1, metrics['common_allele_count'])
    metrics['rare_allele_recall'] /= max(1, metrics['rare_allele_count'])

    return metrics

#####################################################################################################################
#####################################################################################################################
model = LDGroupedAutoencoder(
    input_length=input_length,
    loci_count=n_loci,
    window_size=window_step,
    latent_dim=glatent)

model, history = train_baseline_model(model, train_loader_geno, device=device)
metrics = evaluate_allele_reconstruction(model, test_loader_geno, device=device)

import json; json.dump(metrics, open('local_autoencoder_metrics.json', 'w'), indent=4)

torch.save(model.state_dict(), "localgg_TEST.pt")
