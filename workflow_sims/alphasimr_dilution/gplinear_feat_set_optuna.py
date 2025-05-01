#!/usr/bin/env python3

import gpatlas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import optuna

random.seed(42)

#variables
n_phen=2
n_loci = 2000
n_alleles = 2
EPS = 1e-15


batch_size = 128
num_workers = 3

base_file_name = 'gpnet/input_data/qhaplo_100qtl_1000marker_10000n_'
base_file_name_out = 'qhaplo_100qtl_1000marker_100000n'


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

def laplace_regularization(model):
    """
    Calculate L1 regularization (Laplace prior) on model weights
    """
    l1_reg = 0.0
    # Apply regularization only to the weight parameters, not biases
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only apply to weights, not biases
            l1_reg += torch.sum(torch.abs(param))
    return l1_reg

##########################################################################################
##########################################################################################
def train_gplinear_lasso(model, train_loader, test_loader,
          l1_weight=0.01,  # Renamed from kl_weight to l1_weight
          learning_rate=0.1,
          max_epochs=3,
          min_delta=0.001,
          patience=20,
          feature_selection_threshold=1e-4,  # Threshold for considering a weight significant
          device=device):

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = {
        'train_loss': [],
        'test_loss': [],
        'epochs_trained': 0,
        'active_features': []  # Track how many features remain active
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

            # MAE loss (same as before)
            mse_loss = F.l1_loss(output + EPS, phens + EPS)

            # Laplace prior instead of KL divergence
            l1_loss = laplace_regularization(model)

            # Combined loss (with L1 regularization)
            total_loss = mse_loss + l1_weight * l1_loss

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Count active features
        with torch.no_grad():
            weights = None
            for name, param in model.named_parameters():
                if 'weight' in name and weights is None:  # Get the first layer weights
                    weights = param.detach().cpu()
                    break

            if weights is not None:
                # Count non-zero features (columns with at least one significant weight)
                significant_features = torch.sum(torch.abs(weights) > feature_selection_threshold, dim=0)
                active_features = torch.sum(significant_features > 0).item()
                history['active_features'].append(active_features)
                print(f"Active features: {active_features}/{weights.shape[1]}")

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
                    #evaluate on ordinary MAE loss
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

def get_selected_features(model, threshold=None):
    """
    Extract the indices of features that have significant weights

    Args:
        model: Trained linear model
        threshold: Weight magnitude threshold for significance

    Returns:
        selected_indices: Indices of selected features
    """
    # Get the first layer weights
    weights = None
    for name, param in model.named_parameters():
        if 'weight' in name and weights is None:
            weights = param.detach().cpu().numpy()
            break

    if weights is None:
        raise ValueError("Could not find weight parameters in the model")

    # For each feature (column), calculate the maximum absolute weight
    feature_importance = np.max(np.abs(weights), axis=0)

    # Select features with importance above threshold
    selected_indices = np.where(feature_importance > threshold)[0]

    print(f"Selected {len(selected_indices)} features out of {len(feature_importance)}")

    return selected_indices, feature_importance

##########################################################################################
##########################################################################################

def evaluate_model(model, test_loader):
    model.eval()
    true_phenotypes = []
    predicted_phenotypes = []

    with torch.no_grad():
        for phens, gens in test_loader:
            phens = phens.to(device)
            if hasattr(test_loader.dataset, 'tensors'):
                # For filtered loaders that use selected features
                gens = gens.to(device)
            else:
                # For original loaders
                gens = gens[:, : n_loci * n_alleles].to(device)

            # Get predictions
            predictions = model(gens)

            # Store results
            true_phenotypes.append(phens.cpu().numpy())
            predicted_phenotypes.append(predictions.cpu().numpy())

    # Concatenate batches
    true_phenotypes = np.concatenate(true_phenotypes)
    predicted_phenotypes = np.concatenate(predicted_phenotypes)

    # Calculate correlation for first phenotype
    correlations = []

    for i in range(n_phen):
        corr, _ = pearsonr(true_phenotypes[:, i], predicted_phenotypes[:, i])

        correlations.append(corr)

    # Create a detailed DataFrame with all results
    results_df = pd.DataFrame({
        'trait_number': range(1, n_phen + 1),
        'pearson_correlation': correlations,
        'true_mean': [np.mean(true_phenotypes[:, i]) for i in range(n_phen)],
        'pred_mean': [np.mean(predicted_phenotypes[:, i]) for i in range(n_phen)],
        'true_std': [np.std(true_phenotypes[:, i]) for i in range(n_phen)],
        'pred_std': [np.std(predicted_phenotypes[:, i]) for i in range(n_phen)]
    })

    return results_df

##########################################################################################
##########################################################################################

def train_gpnet(model, train_loader, test_loader=None,
                         n_loci=None,
                         n_alleles=2,
                         max_epochs=5,  # Set a generous upper limit
                         patience=50,      # Number of epochs to wait for improvement
                         min_delta=0.001, # Minimum change to count as improvement
                         learning_rate=0.01,
                         l1_lambda=0, weight_decay=1e-7, device=device):
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

            first_layer_l1 = 0

            for name, param in model.named_parameters():
                # Check if the parameter belongs to the first linear layer
                if 'gpnet.0.weight' in name:
                    first_layer_l1 += torch.sum(torch.abs(param))

            # Combined loss with L1 penalty
            g_p_recon_loss = g_p_recon_loss + l1_lambda * first_layer_l1


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

##########################################################################################
##########################################################################################
def pretrain_lasso_model(l1_weight=None):

    #Train linear model with L1 regularization
    linear_model = gplinear_kl(
        n_loci=n_loci,
        n_phen=n_phen,
    ).to(device)

    linear_model, best_loss_linear, history_linear = train_gplinear_lasso(
        model=linear_model,
        train_loader=train_loader_gp,
        test_loader=test_loader_gp,
        l1_weight=l1_weight,
        device=device
    )
    return linear_model

def run_lasso_mlp_pipeline(linear_model=None,l1_weight=None, feature_threshold=None,
                          mlp_hidden_size=None, max_selected_features=None):
    """
    Run a two-stage pipeline:
    1. Select important featuresfrom pretrained lasso model
    2. Train an MLP on the selected features
    """

    # Stage 1: Feature selection
    selected_indices, feature_importance = get_selected_features(
        linear_model, threshold=feature_threshold
    )

    # Limit the number of features if needed
    if len(selected_indices) > max_selected_features:
        # Sort by importance and take top features
        sorted_indices = np.argsort(-feature_importance)
        selected_indices = sorted_indices[:max_selected_features]
        print(f"Limited selection to top {max_selected_features} features")

    # Create new datasets with only selected features
    def create_filtered_loader(original_loader, selected_indices):
        filtered_data = []
        filtered_labels = []

        for phens, gens in original_loader:
            # Select only the important features
            filtered_gens = gens[:, selected_indices]
            filtered_data.append(filtered_gens)
            filtered_labels.append(phens)

        # Concatenate all batches
        filtered_data = torch.cat(filtered_data, dim=0)
        filtered_labels = torch.cat(filtered_labels, dim=0)

        # Create new dataset and loader
        filtered_dataset = TensorDataset(filtered_labels, filtered_data)
        filtered_loader = DataLoader(
            filtered_dataset,
            batch_size=original_loader.batch_size,
            shuffle=True
        )

        return filtered_loader

    # Create filtered loaders
    filtered_train_loader = create_filtered_loader(train_loader_gp, selected_indices)
    filtered_test_loader = create_filtered_loader(test_loader_gp, selected_indices)

    # Stage 2: Train linear then MLP models on selected features only
    #linear
    linear_model_pruned = gplinear_kl(
        n_loci=len(selected_indices),
        n_phen=n_phen,
    ).to(device)

    linear_model_pruned, best_loss_linear, history_linear = train_gplinear_lasso(
        model=linear_model_pruned,
        train_loader=filtered_train_loader,
        test_loader=filtered_test_loader,
        l1_weight=l1_weight,
        device=device
    )

    #MLP
    mlp_model = gpatlas.GP_net(
        n_loci=len(selected_indices),
        latent_space_g1=mlp_hidden_size,
        latent_space_g=mlp_hidden_size,
        n_pheno=n_phen
    ).to(device)


    mlp_model, best_loss_mlp, history_mlp = train_gpnet(
        model=mlp_model,
        train_loader=filtered_train_loader,
        test_loader=filtered_test_loader,
        n_loci=len(selected_indices),
        device=device,
        l1_lambda=0  # No need for L1 here as we've already done feature selection
    )

    return linear_model_pruned, mlp_model, filtered_test_loader, best_loss_mlp

#####################################################################################################################
#####################################################################################################################
#pretrain linear model globally
linear_model = pretrain_lasso_model(l1_weight=0.001)# L1 regularization strength

def objective(trial: optuna.Trial,
             device: torch.device) -> float:
    """
    Objective function for Optuna that uses early stopping
    """
    # Hyperparameters to optimize
    #predefined_feat_sel_amounts = [100, 200, 300, 400, 500, 750, 1000]
    feat_threshold = trial.suggest_int('feat_threshold', 100,1000)

    _, _, _, best_loss_mlp = run_lasso_mlp_pipeline(
        linear_model=linear_model,
        l1_weight=0.001,
        feature_threshold=0.03,       # Threshold for feature selection
        mlp_hidden_size=4096,           # Size of hidden layers in MLP
        max_selected_features=feat_threshold     # Maximum number of features to use
    )
    return best_loss_mlp


def main():
    # Create study
    study = optuna.create_study(direction='minimize')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run optimization
    n_trials = n_trials_optuna

    #linear_model = pretrain_lasso_model(l1_weight=0.001)# L1 regularization strength

    try:
        # Create a list to store results as we go
        trial_results = []

        study.optimize(
            lambda trial: objective(
                trial=trial,
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
        results_df.to_csv(f'gphybrid/optuna/{base_file_name_out}_optuna.csv', index=False)

        # Reload best model with the optimized hyperparameters
        print("\nReloading best model with optimized hyperparameters...")
        best_predefined_feat_sel_amounts = study.best_params['feat_threshold']

        # Train the best models
        linear_model_pruned, mlp_model, filtered_test_loader, _ = run_lasso_mlp_pipeline(
            linear_model=linear_model,
            l1_weight=0.001,
            feature_threshold=0.03,       # Threshold for feature selection
            mlp_hidden_size=4096,           # Size of hidden layers in MLP
            max_selected_features=best_predefined_feat_sel_amounts     # Maximum number of features to use
        )

        # Evaluate both models and write results
        linear_corr = evaluate_model(linear_model, test_loader_gp)
        linear_corr_pruned = evaluate_model(linear_model_pruned, filtered_test_loader)
        mlp_corr = evaluate_model(mlp_model, filtered_test_loader)

        results_linear_corr = f'gphybrid/{base_file_name_out}_linear_correlations.csv'
        results_linear_corr_pruned = f'gphybrid/{base_file_name_out}_linear_pruned_correlations.csv'
        results_mlp_corr = f'gphybrid/{base_file_name_out}_mlp_pruned_correlations.csv'

        linear_corr.to_csv(results_linear_corr, index=False)
        linear_corr_pruned.to_csv(results_linear_corr_pruned, index=False)
        mlp_corr.to_csv(results_mlp_corr, index=False)

if __name__ == "__main__":
    main()