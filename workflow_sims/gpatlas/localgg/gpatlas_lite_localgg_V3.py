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

n_epochs = 40
batch_size = 128
num_workers = 3
base_file_name = 'gpatlas_input/test_sim_WF_10kbt_10000n_5000000bp_'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gamma_fc_loss = 1.5
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
        self.n_out_channels = n_out_channels  # number of output channels per LD window

        # Calculate the number of groups
        self.n_groups = loci_count // window_size  # 10,000 groups

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

        # Decoder - mirror of encoder
        self.decoder_fc = nn.Linear(latent_dim, self.n_groups*n_out_channels)
        self.decoder_fc_act = nn.LeakyReLU(0.1)

        # Expand each window back to original size
        self.decoder_conv = nn.ConvTranspose1d(
            in_channels=self.n_out_channels * self.n_groups,
            out_channels=self.n_groups,
            kernel_size=window_size * 2,
            stride=window_stride,
            groups=self.n_groups,
            bias=True
        )

        #self.decoder_conv_act = nn.LeakyReLU(0.1)
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
        x = x.reshape(batch_size, self.n_groups * n_out_channels)

        # Map to latent space
        latent = self.encoder_fc(x)
        latent = self.encoder_fc_act(latent)

        # Decode from latent space
        x = self.decoder_fc(latent)
        x = self.decoder_fc_act(x)

        # Reshape for transposed convolution
        # [batch_size, n_groups*n_out_channels] -> [batch_size, n_groups, n_out_channels]
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels, 1)

        # Expand each window back to original size
        x = self.decoder_conv(x)
        #x = self.decoder_conv_act(x)

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
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels)
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
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels, 1)

        # Expand each window back to original size
        x = self.decoder_conv(x)
        #x = self.decoder_conv_act(x)

        # Reshape to original format
        x = x.reshape(batch_size, self.input_length)

        # Apply sigmoid
        x = self.final_act(x)

        return x

#####################################################################################################################
#####################################################################################################################

def focal_loss_for_genetic_data(predictions, targets, gamma=gamma_fc_loss, alpha=None):
    """
    Focal loss customized for genetic data with rare alleles.

    Args:
        predictions: Model output probabilities [batch_size, input_length]
        targets: Ground truth one-hot encoded alleles [batch_size, input_length]
        gamma: Focusing parameter (2-5 recommended for rare alleles)
        alpha: Optional static weighting factor (0-1)
        allele_freqs: Optional pre-computed allele frequencies to inform alpha

    Returns:
        Scalar loss value
    """
    # Clip predictions to prevent numerical issues
    epsilon = 1e-7
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)

    # Compute binary cross entropy
    bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')

    # Compute probability for the correct class
    p_t = targets * predictions + (1 - targets) * (1 - predictions)

    # Apply focusing parameter
    focal_weight = (1 - p_t) ** gamma

    # Apply even alpha weight for now (balanced class weights)
    alpha_weight = 0.5

    focal_weight = focal_weight * alpha_weight

    # Compute final loss
    loss = (focal_weight * bce_loss).mean()

    return loss

#####################################################################################################################
#####################################################################################################################

start_time = tm.time()
def train_baseline_model(model, train_loader, test_loader=None,
                         epochs=n_epochs,
                         learning_rate=0.001,
                         weight_decay=1e-5,
                         device=device,
                         patience=6,
                         min_delta=0.001):
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
        'epochs_trained': 0
    }

    # Early stopping variables
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

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
            #loss = F.binary_cross_entropy(output, data)
            loss = focal_loss_for_genetic_data(output, data, gamma=gamma_fc_loss)

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

            #early stoppage loop
            # Check for improvement
            if avg_test_loss < (best_loss - min_delta):
                best_loss = avg_test_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
                print(f"New best model at epoch {epoch+1} with test loss: {best_loss:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs (best: {best_loss:.6f})")

            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Record how many epochs were actually used
    history['epochs_trained'] = epoch + 1

    # Restore best model
    if best_model_state is not None:
        print(f"Restoring best model from epoch {best_epoch+1}")
        model.load_state_dict(best_model_state)

    return model, best_loss, history


    #return model, history

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

        # Copy the encoder layers
        self.encoder_conv = autoencoder.encoder_conv
        self.encoder_act = autoencoder.encoder_act
        self.encoder_fc = autoencoder.encoder_fc
        self.encoder_fc_act = autoencoder.encoder_fc_act

    def forward(self, x):
        """
        Forward pass through the encoder

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
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels)
        x = self.encoder_fc(x)
        return self.encoder_fc_act(x)


#####################################################################################################################
#####################################################################################################################

model = LDGroupedAutoencoder(
    input_length=input_length,
    loci_count=n_loci,
    window_size=window_step,
    latent_dim=glatent)

#train and sacve full model
model, best_loss, history = train_baseline_model(model, train_loader_geno,test_loader=test_loader_geno, device=device)
torch.save(model.state_dict(), "localgg/localgg_autenc_10kbt_V3_state_dict.pt")
print(f'saved best model with loss: {best_loss}')

#save gg encoder only for G->P
encoder = LDEncoder(model)
torch.save(encoder.state_dict(), "localgg/localgg_enc_10kbt_V3_state_dict.pt")