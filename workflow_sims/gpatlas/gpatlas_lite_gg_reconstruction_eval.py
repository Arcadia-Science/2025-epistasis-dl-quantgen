#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import cast
import h5py
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Constants
batch_size = 128
num_workers = 10
n_geno = 100000
n_alleles = 2
n_loci = n_geno * n_alleles

# Dataset classes
class BaseDataset(Dataset):
    def __init__(self, hdf5_path: Path) -> None:
        self.h5 = h5py.File(hdf5_path, "r")
        self._strain_group = cast(h5py.Group, self.h5["strains"])
        self.strains: list[str] = list(self._strain_group.keys())

    def __len__(self) -> int:
        return len(self._strain_group)

class GenoDataset(BaseDataset):
    def __getitem__(self, idx: int) -> torch.Tensor:
        strain = self.strains[idx]
        strain_data = cast(Dataset, self._strain_group[strain])
        gens = torch.tensor(strain_data["genotype"][:], dtype=torch.float32).flatten()
        return gens

# Model classes
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

class GP_net(nn.Module):
    def __init__(self, n_loci=None, N=None):
        super().__init__()
        if n_loci is None:
            n_loci = n_geno * n_alleles

        batchnorm_momentum = 0.8
        g_latent_dim = N
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

def load_models(encoder_path, decoder_path=None, device='cuda'):
    """Load the encoder and decoder models from checkpoints"""
    # Load encoder
    encoder_checkpoint = torch.load(encoder_path, map_location=device)
    latent_space_g = encoder_checkpoint['hyperparameters']['latent_space_g']

    GQ = GQ_net(n_loci=n_loci, N=latent_space_g).to(device)
    GQ.load_state_dict(encoder_checkpoint['model_state_dict'])
    GQ.eval()

    # Load decoder if provided separately
    if decoder_path:
        decoder_checkpoint = torch.load(decoder_path, map_location=device)
        GP = GP_net(n_loci=n_loci, N=latent_space_g).to(device)
        GP.load_state_dict(decoder_checkpoint['model_state_dict'])
    else:
        # If no separate decoder, assume encoder checkpoint has all model info
        GP = GP_net(n_loci=n_loci, N=latent_space_g).to(device)
        if 'decoder_state_dict' in encoder_checkpoint:
            GP.load_state_dict(encoder_checkpoint['decoder_state_dict'])

    GP.eval()

    return GQ, GP, latent_space_g

def evaluate_autoencoder(GQ, GP, test_loader, device, threshold=0.5):
    """
    Evaluate the autoencoder performance at the allele level

    Returns:
    - correct_allele: count of correctly reconstructed alleles
    - wrong_allele: count of incorrectly reconstructed alleles
    - invalid_state: count of reconstructed invalid states
    - minor_allele_freq: minor allele frequency for each locus
    """
    # Initialize counters per locus position
    n_positions = n_geno
    correct_allele = np.zeros(n_positions)
    wrong_allele = np.zeros(n_positions)
    invalid_state = np.zeros(n_positions)

    # Initialize counters for allele frequencies
    allele1_count = np.zeros(n_positions)  # Count of (1,0) genotypes

    total_samples = 0

    # Process batches
    with torch.no_grad():
        for gens in test_loader:
            batch_size = gens.shape[0]
            total_samples += batch_size

            gens = gens.to(device)

            # Run through autoencoder
            latent = GQ(gens)
            reconstructed = GP(latent)

            # Convert to binary representation (threshold outputs)
            reconstructed_binary = (reconstructed > threshold).float()

            # Analyze per locus
            for pos in range(n_positions):
                # Get allele pairs for this locus
                allele1_idx = pos * n_alleles
                allele2_idx = pos * n_alleles + 1

                for sample_idx in range(batch_size):
                    true_allele1 = gens[sample_idx, allele1_idx].item()
                    true_allele2 = gens[sample_idx, allele2_idx].item()

                    # Count allele1 (1,0) genotypes for MAF calculation
                    if true_allele1 == 1 and true_allele2 == 0:
                        allele1_count[pos] += 1

                    recon_allele1 = reconstructed_binary[sample_idx, allele1_idx].item()
                    recon_allele2 = reconstructed_binary[sample_idx, allele2_idx].item()

                    # Case 1: Correct allele reconstruction
                    if (true_allele1 == recon_allele1 and true_allele2 == recon_allele2):
                        correct_allele[pos] += 1
                    # Case 3: Invalid allelic state
                    elif (recon_allele1 == 1 and recon_allele2 == 1) or (recon_allele1 == 0 and recon_allele2 == 0):
                        invalid_state[pos] += 1
                    # Case 2: Wrong allele
                    else:
                        wrong_allele[pos] += 1

    # Calculate MAF
    maf = np.minimum(allele1_count, total_samples - allele1_count) / total_samples

    # Convert to percentages
    correct_allele = (correct_allele / total_samples) * 100
    wrong_allele = (wrong_allele / total_samples) * 100
    invalid_state = (invalid_state / total_samples) * 100

    return correct_allele, wrong_allele, invalid_state, maf, total_samples

def plot_results(correct_allele, wrong_allele, invalid_state, output_path, bin_size=1000):
    """
    Plot the evaluation results

    Args:
    - correct_allele: percentage of correctly reconstructed alleles
    - wrong_allele: percentage of incorrectly reconstructed alleles
    - invalid_state: percentage of reconstructed invalid states
    - output_path: where to save the plots
    - bin_size: size of bins for aggregating results (reduce noise)
    """
    n_positions = len(correct_allele)

    # Bin the data for smoother visualization
    n_bins = n_positions // bin_size

    binned_correct = np.zeros(n_bins)
    binned_wrong = np.zeros(n_bins)
    binned_invalid = np.zeros(n_bins)

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, n_positions)

        binned_correct[i] = np.mean(correct_allele[start_idx:end_idx])
        binned_wrong[i] = np.mean(wrong_allele[start_idx:end_idx])
        binned_invalid[i] = np.mean(invalid_state[start_idx:end_idx])

    # Create bin positions for x-axis
    bin_positions = np.arange(n_bins) * bin_size + bin_size/2

    # Create dataframe for seaborn
    data = []
    for i in range(n_bins):
        data.append({'Position': bin_positions[i], 'Percentage': binned_correct[i], 'Category': 'Correct Allele'})
        data.append({'Position': bin_positions[i], 'Percentage': binned_wrong[i], 'Category': 'Wrong Allele'})
        data.append({'Position': bin_positions[i], 'Percentage': binned_invalid[i], 'Category': 'Invalid State'})

    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(15, 8))
    sns.set_style('whitegrid')

    # Line plot
    sns.lineplot(data=df, x='Position', y='Percentage', hue='Category', linewidth=2)

    plt.title('Genotype Reconstruction Performance by Position', fontsize=16)
    plt.xlabel('Genomic Position (binned)', fontsize=14)
    plt.ylabel('Percentage of Samples (%)', fontsize=14)
    plt.legend(title='', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_path}_lineplot.png", dpi=300)

    # Create summary statistics
    overall_correct = np.mean(correct_allele)
    overall_wrong = np.mean(wrong_allele)
    overall_invalid = np.mean(invalid_state)

    # Plot summary bar chart
    plt.figure(figsize=(10, 6))
    summary_data = pd.DataFrame({
        'Category': ['Correct Allele', 'Wrong Allele', 'Invalid State'],
        'Percentage': [overall_correct, overall_wrong, overall_invalid]
    })

    sns.barplot(data=summary_data, x='Category', y='Percentage')
    plt.title('Overall Genotype Reconstruction Performance', fontsize=16)
    plt.ylabel('Percentage of Samples (%)', fontsize=14)
    plt.ylim(0, 100)

    # Add percentages on top of bars
    for i, v in enumerate(summary_data['Percentage']):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_path}_summary.png", dpi=300)

    # Save raw data
    with open(f"{output_path}_data.csv", 'w') as f:
        f.write(f"Overall statistics:\n")
        f.write(f"Correct Allele: {overall_correct:.2f}%\n")
        f.write(f"Wrong Allele: {overall_wrong:.2f}%\n")
        f.write(f"Invalid State: {overall_invalid:.2f}%\n\n")

        f.write("Position-wise statistics:\n")
        f.write("Position,Correct_Allele,Wrong_Allele,Invalid_State\n")
        for i in range(n_positions):
            f.write(f"{i},{correct_allele[i]:.2f},{wrong_allele[i]:.2f},{invalid_state[i]:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Genotype Autoencoder Performance')
    parser.add_argument('--encoder', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--decoder', type=str, help='Path to decoder checkpoint (optional if encoder checkpoint includes decoder)')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data HDF5 file')
    parser.add_argument('--output', type=str, required=True, help='Path prefix for output files')
    parser.add_argument('--bin_size', type=int, default=1000, help='Size of bins for aggregating results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Check if CUDA is available if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        args.device = 'cpu'

    device = torch.device(args.device)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    test_data = GenoDataset(args.test_data)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # Load models
    GQ, GP, latent_space_g = load_models(args.encoder, args.decoder, device)

    print(f"Models loaded successfully. Latent space size: {latent_space_g}")
    print(f"Evaluating on {len(test_data)} test samples...")

    # Evaluate
    correct_allele, wrong_allele, invalid_state, total_samples = evaluate_autoencoder(
        GQ, GP, test_loader, device, args.threshold
    )

    print(f"Evaluation complete. Total samples processed: {total_samples}")

    # Plot results
    plot_results(correct_allele, wrong_allele, invalid_state, args.output, args.bin_size)

    print(f"Results saved to {args.output}_*.png and {args.output}_data.csv")

    # Print summary
    print("\nSummary Statistics:")
    print(f"Correct Allele: {np.mean(correct_allele):.2f}%")
    print(f"Wrong Allele: {np.mean(wrong_allele):.2f}%")
    print(f"Invalid State: {np.mean(invalid_state):.2f}%")

if __name__ == "__main__":
    main()