#!/usr/bin/env python3
from snakemake.script import snakemake

import gpatlas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

#variables

batch_size = 128
num_workers = 3

sample_size = snakemake.params['sample_size']
qtl_n = snakemake.params['qtl_n']
rep = snakemake.params['rep']


sim_name = f'qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}'
base_file_name = f'gpnet/input_data/{sim_name}_'

n_loci = int(qtl_n) * 2
n_alleles = 2
n_phen=5
EPS = 1e-15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##########################################################################################
##########################################################################################

loaders = gpatlas.create_data_loaders(base_file_name, batch_size=128, num_workers=3, shuffle=True)

train_loader_gp = loaders['train_loader_gp']
test_loader_gp = loaders['test_loader_gp']

##########################################################################################
##########################################################################################

class gplinear_kl(nn.Module):
    def __init__(self, n_loci, n_phen):
        super(gplinear_kl, self).__init__()
        self.linear = nn.Linear(n_loci, n_phen)

    def forward(self, x):
        return self.linear(x)

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
          kl_weight = 0.1,
          learning_rate=0.1,
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

def run_full_pipeline():
    """
    Objective function for Optuna that uses early stopping
    """
    model = gplinear_kl(
    n_loci=n_loci,
    n_phen=n_phen,
    ).to(device)

    model, best_loss_gp, history = train_gplinear(model=model,
                                            train_loader=train_loader_gp,
                                            test_loader=test_loader_gp,
                                            device=device)
    model.eval()

    #visualize results
    true_phenotypes = []
    predicted_phenotypes = []

    with torch.no_grad():
        for phens, gens in test_loader_gp:
            phens = phens.to(device)
            gens = gens[:, : n_loci * n_alleles]
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

    # Add trait architecture group
    results_df['trait_architecture'] = results_df['trait_number'].apply(
        lambda x: f"{((x-1)//5)*5+1}-{min(((x-1)//5+1)*5, n_phen)}"
    )

    # Save to CSV
    results_df.to_csv(f'gplinear/{sim_name}_phenotype_correlations_untuned.csv', index=False)



    return best_loss_gp

#####################################################################################################################
#####################################################################################################################

def main():
    best_loss_gp = run_full_pipeline()
    print(f"Final loss: {best_loss_gp}")

if __name__ == "__main__":
    main()