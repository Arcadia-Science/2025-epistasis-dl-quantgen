#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset
import numpy as np
from sklearn.metrics import r2_score

import pandas as pd
from pathlib import Path
from typing import cast
import h5py
import time as tm
from datetime import datetime
import optuna
import json


#variables
n_trials_optuna = 50
n_phen=25

n_loci = 100000
n_alleles = 2
window_step = 200
window_stride = 10
#glatent = 3000
input_length = n_loci * n_alleles
n_out_channels = 7

batch_size = 128
num_workers = 3
base_file_name = 'gpatlas_input/test_sim_WF_10kbt_10000n_5000000bp_'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-15

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

train_data_geno = GenoDataset(f''+base_file_name+'train.hdf5')
test_data_geno = GenoDataset(f''+base_file_name+'test.hdf5')

train_data_gp = GenoPhenoDataset(f''+base_file_name+'train.hdf5')
test_data_gp = GenoPhenoDataset(f''+base_file_name+'test.hdf5')

train_data_pheno = PhenoDataset(f''+base_file_name+'train.hdf5')
test_data_pheno = PhenoDataset(f''+base_file_name+'test.hdf5')

##########################################################################################
##########################################################################################

train_loader_geno = torch.utils.data.DataLoader(
    dataset=train_data_geno, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False, drop_last=False
)
test_loader_geno = torch.utils.data.DataLoader(
    dataset=test_data_geno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)


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
#####################################################################################################################
#####################################################################################################################

class LDGroupedAutoencoder(nn.Module):
    def __init__(self, input_length, loci_count, window_size, window_stride, n_out_channels, latent_dim):
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
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.n_groups, self.window_size * 2)

        x = self.encoder_conv(x)
        x = self.encoder_act(x)

        x = x.reshape(batch_size, self.n_groups * self.n_out_channels)
        latent = self.encoder_fc(x)
        latent = self.encoder_fc_act(latent)

        # Decode from latent space
        x = self.decoder_fc(latent)
        x = self.decoder_fc_act(x)

        x = x.reshape(batch_size, self.n_groups * self.n_out_channels, 1)

        x = self.decoder_conv(x)
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
# phenotypes AEs

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
#g to p feed forward network

class GQ_to_P_net(nn.Module):
    def __init__(self, N, latent_space_g, latent_space_gp ):
        super().__init__()

        batchnorm_momentum =0.8
        g_latent_dim = latent_space_g
        self.encoder = nn.Sequential(
            nn.Linear(in_features=g_latent_dim, out_features=latent_space_gp),
            nn.BatchNorm1d(latent_space_gp, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=latent_space_gp, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

#####################################################################################################################
#####################################################################################################################

def focal_loss_for_genetic_data(predictions, targets, gamma, alpha=None):
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
    alpha_weight = 1

    focal_weight = focal_weight * alpha_weight

    # Compute final loss
    loss = (focal_weight * bce_loss).mean()

    return loss

#####################################################################################################################
#####################################################################################################################

start_time = tm.time()
def train_localgg_model(model, train_loader, test_loader=None,
                         gamma_fc_loss=None,
                         max_epochs=50,  # Set a generous upper limit
                         patience=6,      # Number of epochs to wait for improvement
                         min_delta=0.005, # Minimum change to count as improvement
                         learning_rate=0.001, weight_decay=1e-5, device=device):
    """
    Train model with early stopping to prevent overtraining
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
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            # Get data
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)

            # focal loss
            loss = focal_loss_for_genetic_data(output, data, gamma=gamma_fc_loss)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        if test_loader is not None:
            model.eval()
            test_loss = 0

            with torch.no_grad():
                for data in test_loader:
                    if isinstance(data, (list, tuple)):
                        data = data[0]
                    data = data.to(device)

                    output = model(data)
                    test_loss += F.binary_cross_entropy(output, data).item()

            avg_test_loss = test_loss / len(test_loader)
            history['test_loss'].append(avg_test_loss)

            print(f'Epoch: {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.6f}, '
                  f'Test Loss: {avg_test_loss:.6f}')

            # Update learning rate
            scheduler.step(avg_test_loss)

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

#####################################################################################################################
#####################################################################################################################

def train_pp_model(Q, P, train_loader, test_loader=None,
                         num_epochs_pp=25,  # Set a generous upper limit
                         n_phen=25,
                         phen_noise=None,
                         weights_regularization=0,
                         learning_rate=0.001, device=device):
    """
    Train phenotype autoencoder
    """
    # Initialize models
    adam_b = (0.5, 0.999)
    Q = Q.to(device)
    P = P.to(device)

    # Optimizers
    optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=learning_rate, betas=adam_b)
    optim_P_dec = torch.optim.Adam(P.parameters(), lr=learning_rate, betas=adam_b)

    # Training loop phenotype autoencoder
    for epoch in range(num_epochs_pp):
        Q.train()
        P.train()

        epoch_losses = []

        for i, phens in enumerate(train_loader):
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

    ####check pp prediction
    Q.eval()
    P.eval()

    # Collect true and predicted phenotypes
    true_phenos = []
    pred_phenos = []

    with torch.no_grad():
        for phens in test_loader:  # or train_loader_pheno, depending on which you want to use
            phens = phens.to(device)

            z_sample = Q(phens)
            X_sample = P(z_sample)

            # Convert to numpy for R-squared calculation
            true_phenos.extend(phens[:, :n_phen].cpu().numpy())
            pred_phenos.extend(X_sample.cpu().numpy())

    # Calculate R-squared
    true_phenos = np.array(true_phenos)
    pred_phenos = np.array(pred_phenos)

    # Calculate R-squared for each phenotype
    r_squared_values = []
    for i in range(n_phen):
        r_sq = r2_score(true_phenos[:, i], pred_phenos[:, i])
        r_squared_values.append(r_sq)

    # Print R-squared values
    print("R-squared values for each phenotype:")
    for i, r_sq in enumerate(r_squared_values):
        print(f"Phenotype {i+1}: {r_sq:.4f}")


    return Q, P

#####################################################################################################################
#####################################################################################################################
#train GP layers

def train_gp_model(GQP, train_loader, test_loader=None,
                         num_epochs_gp=50,  # Set a generous upper limit
                         n_loci=None,
                         n_alleles=n_alleles,
                         gen_noise=None,
                         GQ=None,
                         P=None,
                         weights_regularization=0,
                         patience=6,      # Number of epochs to wait for improvement
                         min_delta=0.005, # Minimum change to count as improvement
                         learning_rate=0.0001, device=device):
    """
    Train phenotype autoencoder
    """
    adam_b = (0.5, 0.999)

    GQP = GQP.to(device)
    optim_GQP_dec = torch.optim.Adam(GQP.parameters(), lr=learning_rate, betas=adam_b)

    #Training loop GP network
    P.eval() #set pheno decoder to eval only

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_GQP_dec, mode='min', factor=0.5, patience=3, verbose=True
    )

    history = {'epochs_trained': 0}

    # Early stopping variables
    best_loss_gp = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    for epoch_gp in range(num_epochs_gp):
        GQP.train()
        epoch_losses_gp = []

        for i, (phens, gens) in enumerate(train_loader):
            phens = phens.to(device)
            gens = gens[:, : n_loci * n_alleles]

            pos_noise = np.random.binomial(1, gen_noise / 2, gens.shape)
            neg_noise = np.random.binomial(1, gen_noise / 2, gens.shape)
            noise_gens = torch.tensor(
                np.where((gens + pos_noise - neg_noise) > 0, 1, 0), dtype=torch.float32
            )

            noise_gens = gens
            noise_gens = noise_gens.to(device)

            batch_size = phens.shape[0]

            GQP.zero_grad()

            z_sample = GQ.encode(noise_gens)
            z_sample = GQP(z_sample)
            X_sample = P(z_sample)


            g_p_recon_loss = F.l1_loss(X_sample + EPS, phens[:, :n_phen] + EPS)

            l1_reg = torch.linalg.norm(torch.sum(GQP.encoder[0].weight, axis=0), 1)
            l2_reg = torch.linalg.norm(torch.sum(GQP.encoder[0].weight, axis=0), 2)
            g_p_recon_loss = g_p_recon_loss + l1_reg * weights_regularization + l2_reg * weights_regularization

            g_p_recon_loss.backward()
            optim_GQP_dec.step()


    # Test set evaluation GP
        if epoch_gp % 1 == 0:
            GQ.eval()
            GQP.eval()
            P.eval()
            test_losses_gp = []

            with torch.no_grad():
                for phens, gens in test_loader:
                    phens = phens.to(device)
                    gens = gens.to(device)

                    z_sample = GQ.encode(gens)
                    z_sample = GQP(z_sample)
                    X_sample = P(z_sample)

                    test_loss = F.l1_loss(X_sample + EPS, phens[:, :n_phen] + EPS)
                    test_losses_gp.append(float(test_loss))

            avg_test_loss_gp = np.mean(test_losses_gp)
            print(f"Epoch {epoch_gp+1} test loss: {avg_test_loss_gp}")
            scheduler.step(avg_test_loss_gp)

            #early stoppage loop
            # Check for improvement
            if avg_test_loss_gp < (best_loss_gp - min_delta):
                best_loss_gp = avg_test_loss_gp
                best_epoch = epoch_gp
                patience_counter = 0
                # Save best model state
                best_model_state = {k: v.cpu().detach().clone() for k, v in GQP.state_dict().items()}
                print(f"New best gp model at epoch {epoch_gp+1} with test loss: {best_loss_gp:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs (best: {best_loss_gp:.6f})")

            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch_gp+1} epochs")
                break

    #reload best model after training done
    if best_model_state is not None:
        print(f"Restoring best model from epoch {best_epoch+1}")
        GQP.load_state_dict(best_model_state)

    return best_loss_gp, GQP



#torch.save(P.state_dict(), "localgg/localgg_P_1kbt_V1_state_dict.pt")
#torch.save(GQP.state_dict(), "localgg/localgg_GQP_1kbt_V1_state_dict.pt")



#####################################################################################################################
#####################################################################################################################


def objective(trial: optuna.Trial,
             n_loci: int,
             n_alleles: int,
             device: torch.device) -> float:
    """
    Objective function for Optuna that uses early stopping
    """

    ##################
    # geno autoencoder
    glatent = trial.suggest_int('glatent', 500, 3500)
    gamma_fc_loss = trial.suggest_float('gamma_fc_loss', 0, 3)

    GQ = LDGroupedAutoencoder(
        input_length=input_length,
        loci_count=n_loci,
        window_size=200,
        window_stride=10,
        n_out_channels=7,
        latent_dim=glatent)

    # Use early stopping with appropriate patience
    GQ, best_loss, history = train_localgg_model(
        GQ,
        train_loader=train_loader_geno,
        test_loader=test_loader_geno,
        gamma_fc_loss=gamma_fc_loss,
        max_epochs=50,          # Set a generous maximum
        patience=6,             # Wait 7 epochs without improvement
        min_delta=0.001,       # Minimum improvement threshold
        device=device
    )

    ##################
    #pheno autoencdoer
    latent_space_p = trial.suggest_int('latent_space_p', 100, 3000)

    Q = Q_net(phen_dim=n_phen, N = latent_space_p).to(device)
    P = P_net(phen_dim=n_phen, N = latent_space_p).to(device)

    Q, P = train_pp_model(Q, P,
                          train_loader=train_loader_pheno,
                          test_loader=test_loader_pheno,
                          phen_noise=0,
                          weights_regularization=0.00000001)

    ##################
    #gp network
    latent_space_gp = trial.suggest_int('latent_space_gp', 100, 3000)

    GQ.eval()
    P.eval()

    GQP = GQ_to_P_net(N=latent_space_p, latent_space_g=glatent, latent_space_gp=latent_space_gp).to(device)

    best_loss_gp, GQP = train_gp_model(GQP,
                         P=P,
                         GQ=GQ,
                         train_loader=train_loader_gp,
                         test_loader=test_loader_gp,
                         n_loci=n_loci,
                         n_alleles=n_alleles,
                         gen_noise=0.99,
                         weights_regularization=0.000000001)

    # Log useful information for this trial
    trial.set_user_attr('epochs_trained', history['epochs_trained'])
    trial.set_user_attr('training_history', {
        'train_loss': history['train_loss'],
        'test_loss': history['test_loss']
    })

    return best_loss_gp


#####################################################################################################################
#####################################################################################################################

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = Path('localgg/optuna')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create study
    study = optuna.create_study(direction='minimize')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run optimization
    n_trials = n_trials_optuna

    try:
        # Create a list to store results as we go
        trial_results = []

        study.optimize(
            lambda trial: objective(
                trial=trial,
                n_loci=n_loci,
                n_alleles=n_alleles,
                device=device
            ),
            n_trials=n_trials,
            callbacks=[
                lambda study, trial: trial_results.append({
                    'trial_number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'state': trial.state.name
                })
            ]
        )

    finally:
        print("\nStudy completed!")
        print(f"Best parameters found: {study.best_params}")
        print(f"Best value achieved: {study.best_value}")

        # Save results to CSV
        results_df = pd.DataFrame(trial_results)
        results_df.to_csv(f'localgg/optuna/optuna_trials_gp_{timestamp}.csv', index=False)

        # Save detailed study information to JSON
        study_info = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': n_trials,
            'datetime': timestamp,
            'all_trials': trial_results
        }

        with open(f'localgg/optuna/optuna_study_gp_{timestamp}.json', 'w') as f:
            json.dump(study_info, f, indent=4)

        print(f"\nResults saved to:")
        print(f"- optuna_trials_gp_{timestamp}.csv")
        print(f"- optuna_study_gp_{timestamp}.json")

if __name__ == "__main__":
    main()