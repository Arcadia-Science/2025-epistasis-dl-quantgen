
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path
from torch.utils.data.dataset import Dataset
import h5py
from typing import cast
#####################################################################
#####################################################################

n_phen = 25
n_geno = 100000
n_alleles = 2
n_loci = n_geno * n_alleles
batch_size = 128
num_workers = 10

timestamp = 20240221
#####################################################################
#####################################################################

# Dataset classes
class BaseDataset(Dataset):
    def __init__(self, hdf5_path: Path) -> None:
        self.h5 = h5py.File(hdf5_path, "r")
        self._strain_group = cast(h5py.Group, self.h5["strains"])
        self.strains: list[str] = list(self._strain_group.keys())

    def __len__(self) -> int:
        return len(self._strain_group)

class GenoPhenoDataset(BaseDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        strain = self.strains[idx]
        strain_data = cast(Dataset, self._strain_group[strain])
        phens = torch.tensor(strain_data["phenotype"][:], dtype=torch.float32)
        gens = torch.tensor(strain_data["genotype"][:], dtype=torch.float32).flatten()
        return phens, gens

# Create datasets and loaders
test_data_gp = GenoPhenoDataset('gpatlas/test_sim_WF_1kbt_10000n_5000000bp_test.hdf5')

test_loader_gp = torch.utils.data.DataLoader(
    dataset=test_data_gp,
    batch_size=batch_size,  # You'll need to define batch_size
    num_workers=num_workers,  # You'll need to define num_workers
    shuffle=False  # Keep False for evaluation
)

#####################################################################
#####################################################################

# Add your model classes
class GQ_net(nn.Module):
    def __init__(self, n_loci=None, N=None):
        super().__init__()
        if n_loci is None:
            n_loci = n_geno * n_alleles

        batchnorm_momentum = 0.8
        g_latent_dim = N
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


class P_net(nn.Module):
    def __init__(self, phen_dim=None, N=None):
        if phen_dim is None:
            phen_dim = n_phen

        out_phen_dim = n_phen
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

class GQ_to_P_net(nn.Module):
    def __init__(self, N, latent_space_g):
        super().__init__()

        batchnorm_momentum = 0.8
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

#####################################################################
#####################################################################

def evaluate_models(timestamp, test_loader_gp, n_phen, device):
    # Load GQ encoder (assuming this is needed)
    gg_checkpoint = torch.load('gpatlas/optuna/best_encoder_gg_20250214_224526.pt')
    GQ = GQ_net(n_loci=n_loci, N=gg_checkpoint['hyperparameters']['latent_space_g']).to(device)
    GQ.load_state_dict(gg_checkpoint['model_state_dict'])
    GQ.eval()

    # Load PP autoencoder
    pp_files = list(Path('gpatlas/optuna').glob(f'best_encoder_pp_{timestamp}*.pt'))
    pp_checkpoint = torch.load(sorted(pp_files)[-1])  # Load the last (best) checkpoint
    latent_space_p = pp_checkpoint['hyperparameters']['latent_space_p']

    P = P_net(phen_dim=n_phen, N=latent_space_p).to(device)
    P.load_state_dict(pp_checkpoint['decoder_state_dict'])
    P.eval()

    # Load GP network
    gp_files = list(Path('gpatlas/optuna').glob(f'best_encoder_gp_{timestamp}*.pt'))
    gp_checkpoint = torch.load(sorted(gp_files)[-1])  # Load the last (best) checkpoint

    GQP = GQ_to_P_net(N=latent_space_p, latent_space_g=gg_checkpoint['hyperparameters']['latent_space_g']).to(device)
    GQP.load_state_dict(gp_checkpoint['model_state_dict'])
    GQP.eval()

    # Collect predictions and true values
    true_phenotypes = []
    predicted_phenotypes = []

    with torch.no_grad():
        for phens, gens in test_loader_gp:
            phens = phens.to(device)
            gens = gens.to(device)

            # Get predictions
            z_sample = GQ(gens)
            z_sample = GQP(z_sample)
            predictions = P(z_sample)

            # Store results
            true_phenotypes.append(phens.cpu().numpy())
            predicted_phenotypes.append(predictions.cpu().numpy())

    # Concatenate batches
    true_phenotypes = np.concatenate(true_phenotypes)
    predicted_phenotypes = np.concatenate(predicted_phenotypes)

    # Calculate correlations for each phenotype
    correlations = []
    p_values = []
    for i in range(n_phen):
        corr, p_val = pearsonr(true_phenotypes[:, i], predicted_phenotypes[:, i])
        correlations.append(corr)
        p_values.append(p_val)

    # Create plot
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.ravel()

    for i in range(n_phen):
        ax = axes[i]
        ax.scatter(true_phenotypes[:, i], predicted_phenotypes[:, i], alpha=0.5)
        ax.set_title(f'Phenotype {i+1}\nr = {correlations[i]:.3f}')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')

        # Add diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'gpatlas/optuna/phenotype_predictions_{timestamp}.png')
    plt.close()

    # Save correlations
    np.savetxt(
        f'gpatlas/optuna/phenotype_correlations_{timestamp}.csv',
        np.column_stack((correlations, p_values)),
        delimiter=',',
        header='correlation,p_value'
    )

    print("\nPhenotype Correlations:")
    for i, (corr, p_val) in enumerate(zip(correlations, p_values)):
        print(f"Phenotype {i+1}: r = {corr:.3f} (p = {p_val:.3e})")

    return correlations, p_values

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # You'll need to specify the timestamp from your training run
    timestamp = "YOUR_TIMESTAMP_HERE"

    correlations, p_values = evaluate_models(timestamp, test_loader_gp, n_phen, device)