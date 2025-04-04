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
n_phen=25
n_loci = 200000
n_alleles = 2
latent_space_g = 3000
EPS = 1e-15


batch_size = 384
num_workers = 3

#base_file_name = 'gpatlas_input/test_sim_WF_1kbt_10000n_5000000bp_'
base_file_name = '../alphasimr_upsample/test_sim_WF_1kbt_100kups_5mb_'
base_file_name_out = 'gplinear/test_sim_WF_1kbt_100kups_5mb_'


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

def kl_divergence_loss(model, prior_var=1.0):
    """KL divergence to standard normal prior N(0,1)"""
    kl_loss = 0
    for param in model.parameters():
        # Simplified KL for fixed variance posterior to N(0,1) prior
        kl_loss += 0.5 * torch.sum(param ** 2)
    return kl_loss

##########################################################################################
##########################################################################################

# Training loop with tunable regularization strength
def train_gplinear(model, train_loader, test_loader,
          kl_weight = 0.01,
          learning_rate=0.0001,
          max_epochs=200,
          min_delta = 0.001,
          patience = 20,
          device=device):

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
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


    for epoch in range(max_epochs):
        model.train()
        train_loss = 0

        for i, (phens, gens) in enumerate(train_loader):
            # Convert data to tensors
            phens = phens.to(device)
            gens = gens[:, : n_loci * n_alleles]
            gens = gens.to(device)

            # Forward pass
            output = model(gens)

            mse_loss = F.l1_loss(output + EPS, phens + EPS)
            kl_loss = kl_divergence_loss(model)
            # Combined loss (equivalent to ridge regression with lambda = kl_weight)
            total_loss = mse_loss + kl_weight * kl_loss

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

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
                    #evaluate on ordinary MSE loss
                    mse_loss = F.l1_loss(output + EPS, phens + EPS)
                    test_loss += mse_loss.item()

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

##########################################################################################
##########################################################################################

def train_gpcombi(model, train_loader, test_loader=None,
                         linear_model=None,
                         n_loci=None,
                         n_alleles=2,
                         max_epochs=100,  # Set a generous upper limit
                         patience=10,      # Number of epochs to wait for improvement
                         min_delta=0.001, # Minimum change to count as improvement
                         learning_rate=0.001, weight_decay=1e-5, device=device):
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

            phens_add = linear_model(gens)

            # Forward pass
            optimizer.zero_grad()
            output = model(gens)

            # focal loss
            g_p_recon_loss = F.l1_loss(output + EPS, phens - (phens_add + EPS))

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

                    phens_add = linear_model(gens)

                    output = model(gens)
                    test_loss += F.l1_loss(phens_add + output + EPS, phens)

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
    linear_model = gpatlas.gplinear_kl(n_loci=n_loci,
                                n_phen=n_phen,
                                ).to(device)

    linear_model, best_loss_gp, history = train_gplinear(model=linear_model,
                                                train_loader=train_loader_gp,
                                                test_loader=test_loader_gp,
                                                device=device)
    linear_model.eval()



    model = gpatlas.GP_net_combi(
        n_loci=n_loci,
        latent_space_g=latent_space_g,
        n_pheno=n_phen,
        linear_model=linear_model)

    model, best_loss_gp, history = train_gpcombi(model=model,
                                            train_loader=train_loader_gp,
                                            test_loader=test_loader_gp,
                                            linear_model=linear_model,
                                            n_loci=n_loci,
                                            device=device)
    model.eval()

    #visualize results
    true_phenotypes = []
    predicted_phenotypes = []
    non_linear_preds = []
    linear_preds = []

    with torch.no_grad():
        for phens, gens in test_loader_gp:
            gens = gens[:, : n_loci * n_alleles]
            phens = phens.to(device)
            gens = gens.to(device)

            # Get predictions
            linear_pred = linear_model(gens)
            non_linear_pred = model(gens)
            predictions = linear_pred + non_linear_pred

            # Store results
            non_linear_preds.append(non_linear_pred.cpu().numpy())
            linear_preds.append(linear_pred.cpu().numpy())
            true_phenotypes.append(phens.cpu().numpy())
            predicted_phenotypes.append(predictions.cpu().numpy())

    # Concatenate batches
    true_phenotypes = np.concatenate(true_phenotypes)
    predicted_phenotypes = np.concatenate(predicted_phenotypes)
    linear_preds = np.concatenate(linear_preds)
    non_linear_preds = np.concatenate(non_linear_preds)


    # Calculate correlations for each phenotype
    correlations = []
    p_values = []
    correlations_epi = []
    correlations_add = []

    for i in range(n_phen):
        corr, _ = pearsonr(true_phenotypes[:, i], predicted_phenotypes[:, i])
        correlations.append(corr)
        corr_resi, _ = pearsonr((true_phenotypes[:, i] - linear_preds[:, i]), non_linear_preds[:, i])
        correlations_epi.append(corr_resi)
        corr_add, _ = pearsonr(true_phenotypes[:, i] , linear_preds[:, i])
        correlations_add.append(corr_add)

    # Create data for the boxplot
    boxplot_data = []
    for i in range(0, n_phen, 5):
        end_idx = min(i+5, n_phen)
        group_name = f"{i+1}-{end_idx}"
        for j in range(i, end_idx):
            boxplot_data.append({
                'trait_architecture': group_name,
                'pearson_corr': correlations[j],
                'trait_number': j+1
            })

    # Convert to DataFrame
    corr_df = pd.DataFrame(boxplot_data)

    # Create the boxplot with Seaborn
    plt.figure(figsize=(4, 3.5))

    sns.boxplot(x="trait_architecture", y="pearson_corr", data=corr_df)


    # Customize the plot
    plt.ylabel('Pearson Correlation (r)')
    plt.xlabel('Trait trait_architecture')
    plt.ylim(0, 1)
    plt.axhline(y=0.7, color='red', linestyle='--')


    plt.tight_layout()
    plt.savefig(f'{base_file_name_out}_pheno_corr.png')

    # Create a detailed DataFrame with all results
    results_df = pd.DataFrame({
        'trait_number': range(1, n_phen + 1),
        'pearson_correlation': correlations,
        'correlations_epi': correlations_epi,
        'correlations_add': correlations_add,
        'true_mean': [np.mean(true_phenotypes[:, i]) for i in range(n_phen)],
        'pred_mean': [np.mean(predicted_phenotypes[:, i]) for i in range(n_phen)],
        'true_std': [np.std(true_phenotypes[:, i]) for i in range(n_phen)],
        'pred_std': [np.std(predicted_phenotypes[:, i]) for i in range(n_phen)]
    })

    # Add trait architecture group
    results_df['trait_architecture'] = results_df['trait_number'].apply(
        lambda x: f"{((x-1)//5)*5+1}-{min(((x-1)//5+1)*5, n_phen)}"
    )

    # Save to CSV
    results_df.to_csv(f'{base_file_name_out}_pheno_corr.csv', index=False)



    return best_loss_gp

#####################################################################################################################
#####################################################################################################################

def main():
    best_loss_gp = run_full_pipeline()
    print(f"Final loss: {best_loss_gp}")

if __name__ == "__main__":
    main()