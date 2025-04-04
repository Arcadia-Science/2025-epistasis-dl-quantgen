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
n_phen=46

n_loci = 11623
n_alleles = 2
window_step = 197
window_stride = 10
#glatent = 3000
input_length = n_loci * n_alleles
n_out_channels = 7

batch_size = 128
num_workers = 3
base_file_name = 'BYxRM_'
base_file_name_out = 'BYxRM_gglocal'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-15

##########################################################################################
##########################################################################################

loaders = gpatlas.create_data_loaders(base_file_name, batch_size=batch_size, num_workers=num_workers, shuffle=True)


train_loader_geno = loaders['train_loader_geno']
test_loader_geno = loaders['test_loader_geno']

train_loader_pheno = loaders['train_loader_pheno']
test_loader_pheno = loaders['test_loader_pheno']

train_loader_gp = loaders['train_loader_gp']
test_loader_gp = loaders['test_loader_gp']

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
                         max_epochs=200,  # Set a generous upper limit
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
        optimizer, mode='min', factor=0.5, patience=3
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
                         num_epochs_pp=200,  # Set a generous upper limit
                         n_phen=46,
                         phen_noise=0.8,
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

            recon_loss = F.mse_loss(X_sample + EPS, phens[:, :n_phen] + EPS)

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

 # Create simplified visualization of predicted vs actual phenotype
    fig, ax = plt.subplots(figsize=(5, 5))

    # Find global min and max for plot limits
    min_val = min(true_phenos.min(), pred_phenos.min())
    max_val = max(true_phenos.max(), pred_phenos.max())

    # Plot all phenotypes on one plot
    for i in range(n_phen):
        # Get the true and predicted values for this phenotype
        true_vals = true_phenos[:, i]
        pred_vals = pred_phenos[:, i]

        # Plot predicted vs actual (let matplotlib assign colors automatically)
        ax.scatter(true_vals, pred_vals)

    # Plot the perfect prediction line
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # Set labels
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

    # Remove gridlines
    plt.tight_layout()
    plt.savefig(f'{base_file_name_out}_pp_predictions.png')
    plt.close()

    return Q, P

#####################################################################################################################
#####################################################################################################################
#train GP layers

def train_gp_model(GQP, train_loader, test_loader=None,
                         num_epochs_gp=100,  # Set a generous upper limit
                         n_loci=None,
                         n_alleles=n_alleles,
                         gen_noise=None,
                         GQ=None,
                         P=None,
                         weights_regularization=0,
                         train_gq = True,
                         phen_indices=None,
                         patience=10,      # Number of epochs to wait for improvement
                         min_delta=0.003, # Minimum change to count as improvement
                         learning_rate=0.001, device=device):
    """
    Train gp network
    """
    adam_b = (0.5, 0.999)

    GQP = GQP.to(device)

    #decide if genotype autoencoder will also be trained or not (default no)
    if train_gq == True:
        print("Training both GQP and GQ encoder together")
        GQ.train()
        optim_GQP_dec = torch.optim.Adam(
        list(GQP.parameters()) + list(GQ.parameters()), lr=learning_rate, betas=adam_b)
    else:
        print("Training only GQP with GQ encoder frozen")
        GQ.eval()
        optim_GQP_dec = torch.optim.Adam(GQP.parameters(), lr=learning_rate, betas=adam_b)

    #Training loop GP network
    P.eval() #set pheno decoder to eval only

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_GQP_dec, mode='min', factor=0.5, patience=3
    )

    history = {'epochs_trained': 0}

    # Early stopping variables
    best_loss_gp = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

   # Create mask for phenotype selection
    if phen_indices is None:
        # If no specific indices provided, use all phenotypes
        phen_mask = torch.ones(n_phen, dtype=torch.bool)
    else:
        # Create a mask with 1s only at the specified indices
        phen_mask = torch.zeros(n_phen, dtype=torch.bool)
        for idx in phen_indices:
            if 0 <= idx < n_phen:  # Ensure index is valid
                phen_mask[idx] = True

    # Make sure we have at least one phenotype selected
    if not torch.any(phen_mask):
        raise ValueError("No valid phenotype indices selected for training")

    phen_mask = phen_mask.to(device)

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

            X_selected = X_sample[:, phen_mask]
            phens_selected = phens[:, phen_mask]

            g_p_recon_loss = F.l1_loss(X_selected + EPS, phens_selected + EPS)

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

                    X_selected = X_sample[:, phen_mask]
                    phens_selected = phens[:, phen_mask]

                    test_loss = F.l1_loss(X_selected + EPS, phens_selected + EPS)
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


def run_full_pipeline():
    """
    Objective function for Optuna that uses early stopping
    """

    ##################
    # geno autoencoder
    glatent = 3500
    gamma_fc_loss = 0

    GQ = gpatlas.LDGroupedAutoencoder(
        input_length=input_length,
        loci_count=n_loci,
        window_size=window_step,
        window_stride=window_stride,
        n_out_channels=7,
        latent_dim=glatent)

    # Use early stopping with appropriate patience
    GQ, best_loss, history = train_localgg_model(
        GQ,
        train_loader=train_loader_geno,
        test_loader=test_loader_geno,
        gamma_fc_loss=gamma_fc_loss,
        max_epochs=50,          # Set a generous maximum
        patience=7,             # Wait 7 epochs without improvement
        min_delta=0.001,       # Minimum improvement threshold
        device=device
    )

    ##################
    #pheno autoencdoer
    hidden_dim = 64
    latent_space_p = 32

    Q = gpatlas.Q_net(phen_dim=n_phen, N = latent_space_p, hidden_dim = hidden_dim).to(device)
    P = gpatlas.P_net(phen_dim=n_phen, N = latent_space_p, hidden_dim = hidden_dim).to(device)

    Q, P = train_pp_model(Q, P,
                          train_loader=train_loader_pheno,
                          test_loader=test_loader_pheno,
                          phen_noise=0.8,
                          weights_regularization=0)

    ##################
    #gp network
    latent_space_gp = 2048

    GQP = gpatlas.GQ_to_P_net(N=latent_space_p, latent_space_g=glatent, latent_space_gp=latent_space_gp).to(device)

    best_loss_gp, GQP = train_gp_model(GQP,
                         P=P,
                         GQ=GQ,
                         train_loader=train_loader_gp,
                         test_loader=test_loader_gp,
                         train_gq=True,
                         #phen_indices=[0,1,2,3,4],
                         n_loci=n_loci,
                         n_alleles=n_alleles,
                         gen_noise=0.99,
                         weights_regularization=0.000000001)



    ####plot results pf GP prediction
    GQP.eval()

    # Collect predictions and true values
    true_phenotypes = []
    predicted_phenotypes = []

    with torch.no_grad():
        for phens, gens in test_loader_gp:
            phens = phens.to(device)
            gens = gens.to(device)

            # Get predictions
            z_sample = GQ.encode(gens)
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

    # Create data for the boxplot
     # Create simplified visualization of predicted vs actual phenotype
    fig, ax = plt.subplots(figsize=(5, 5))

    # Find global min and max for plot limits
    min_val = min(true_phenotypes.min(), predicted_phenotypes.min())
    max_val = max(true_phenotypes.max(), predicted_phenotypes.max())

    # Plot all phenotypes on one plot
    for i in range(n_phen):
        # Get the true and predicted values for this phenotype
        true_vals = true_phenotypes[:, i]
        pred_vals = predicted_phenotypes[:, i]

        # Plot predicted vs actual (let matplotlib assign colors automatically)
        ax.scatter(true_vals, pred_vals)

    # Plot the perfect prediction line
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # Set labels
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

    # Remove gridlines
    plt.tight_layout()
    plt.savefig(f'{base_file_name_out}_gp_predictions.png')
    plt.close()

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
    results_df.to_csv(f'{base_file_name_out}_pheno_corr.csv', index=False)



    return best_loss_gp


#####################################################################################################################
#####################################################################################################################

def main():
    #run_full_pipeline()
    best_loss_gp = run_full_pipeline()
    print(f"Final loss: {best_loss_gp}")

if __name__ == "__main__":
    main()
