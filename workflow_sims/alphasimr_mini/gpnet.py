#!/usr/bin/env python3

import gpatlas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

import pandas as pd
from pathlib import Path
from typing import cast
import h5py
import time as tm
from datetime import datetime



#variables
n_phen=2
n_loci = 2000
n_alleles = 2
latent_space_g = 3500
EPS = 1e-15


batch_size = 128
num_workers = 3

base_file_name = 'alphasimr_mini/test_sim_qhaplo_100k_1000sites_Ve0_'
base_file_name_out = 'alphasimr_mini/experiments/test_sim_qhaplo_100k_1000sites_Ve0_'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##########################################################################################
##########################################################################################

loaders = gpatlas.create_data_loaders(base_file_name, batch_size=128, num_workers=3, shuffle=True)

train_loader_geno = loaders['train_loader_geno']
test_loader_geno = loaders['test_loader_geno']

train_loader_pheno = loaders['train_loader_pheno']
test_loader_pheno = loaders['test_loader_pheno']

train_loader_gp = loaders['train_loader_gp']
test_loader_gp = loaders['test_loader_gp']

##########################################################################################
##########################################################################################

def train_gpnet(model, train_loader, test_loader=None,
                         n_loci=None,
                         n_alleles=2,
                         max_epochs=50,  # Set a generous upper limit
                         patience=10,      # Number of epochs to wait for improvement
                         min_delta=0.003, # Minimum change to count as improvement
                         learning_rate=0.0001, weight_decay=1e-5, device=device):
    """
    Train model with early stopping to prevent overtraining
    """
    # Move model to device
    model = model.to(device)

    # Initialize optimizer with proper weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

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

        for i, (phens, gens) in enumerate(train_loader):

            phens = phens.to(device)
            gens = gens[:, : n_loci * n_alleles]
            gens = gens.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(gens)

            # focal loss
            g_p_recon_loss = F.l1_loss(output + EPS, phens + EPS)

            # Backward and optimize
            g_p_recon_loss.backward()
            optimizer.step()

            train_loss += g_p_recon_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        if test_loader is not None:
            model.eval()
            test_loss = 0

            with torch.no_grad():
                for phens, gens in test_loader:
                    phens = phens.to(device)
                    gens = gens[:, : n_loci * n_alleles]
                    gens = gens.to(device)

                    output = model(gens)
                    test_loss += F.l1_loss(output + EPS, phens + EPS)

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

def run_full_pipeline():
    """
    Objective function for Optuna that uses early stopping
    """
    model = gpatlas.GP_net(
    n_loci=n_loci,
    latent_space_g=latent_space_g,
    n_pheno=n_phen,
    )

    model, best_loss_gp, history = train_gpnet(model=model,
                                            train_loader=train_loader_gp,
                                            test_loader=test_loader_gp,
                                            n_loci=n_loci,
                                            device=device)
    model.eval()

    #visualize results
    true_phenotypes = []
    predicted_phenotypes = []

    with torch.no_grad():
        for phens, gens in test_loader_gp:
            phens = phens.to(device)
            gens = gens.to(device)

            # Get predictions
            predictions = model(gens)

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

    # Create a detailed DataFrame with all results
    results_df = pd.DataFrame({
        'trait_number': range(1, n_phen + 1),
        'pearson_correlation': correlations,
        'p_value': p_values,
        'true_mean': [np.mean(true_phenotypes[:, i]) for i in range(n_phen)],
        'pred_mean': [np.mean(predicted_phenotypes[:, i]) for i in range(n_phen)],
        'true_std': [np.std(true_phenotypes[:, i]) for i in range(n_phen)],
        'pred_std': [np.std(predicted_phenotypes[:, i]) for i in range(n_phen)]
    })

    # Save to CSV
    results_df.to_csv(f'{base_file_name_out}_pheno_corr.csv', index=False)

    model.eval()

    loss_landscape_data = gpatlas.visualize_loss_landscape(
        model=model,
        train_loader=train_loader_gp,
        test_loader=test_loader_gp,
        loss_fn=F.l1_loss,  # Using L1 loss as in your training
        device=device,
        n_points=21,  # Lower resolution for faster computation
        alpha_range=(-1, 1),
        beta_range=(-1, 1),
        EPS=EPS,
        n_loci=n_loci,
        n_alleles=n_alleles,
        base_file_name_out=base_file_name_out
    )


    return best_loss_gp

#####################################################################################################################
#####################################################################################################################

def main():
    best_loss_gp = run_full_pipeline()
    print(f"Final loss: {best_loss_gp}")

if __name__ == "__main__":
    main()