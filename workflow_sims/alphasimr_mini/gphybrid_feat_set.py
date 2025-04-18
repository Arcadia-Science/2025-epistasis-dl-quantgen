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

random.seed(42)

#variables
n_phen=2
n_loci = 2000
n_alleles = 2
latent_space_g = 3500
EPS = 1e-15


batch_size = 128
num_workers = 3

base_file_name = 'test_sim_qhaplo_10k_1ksites_100qtl_Ve0_'
base_file_name_out = 'experiments/test_sim_qhaplo_10k_1ksites_100qtl_Ve0_HYBRID_FEATURE_SELN.csv'


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
class HybridSkipModel(nn.Module):
    """
    Hybrid model for genomic prediction:
    - Selected features go through the MLP path
    - All features go through a linear skip connection

    Args:
        n_loci: Total number of loci/features
        selected_indices: Indices of informative features for non-linear processing
        latent_space_g: Hidden layer size for MLP
        n_pheno: Number of phenotypes to predict
    """
    def __init__(self, n_loci, selected_indices, latent_space_g, n_pheno):
        super().__init__()

        # Store feature indices
        self.selected_indices = selected_indices

        # MLP for selected features
        self.mlp = nn.Sequential(
            nn.Linear(len(selected_indices), latent_space_g),
            nn.BatchNorm1d(latent_space_g, momentum=0.8),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(latent_space_g, latent_space_g),
            nn.BatchNorm1d(latent_space_g, momentum=0.8),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(latent_space_g, n_pheno)
        )

        # Linear layer for all features
        self.linear = nn.Linear(n_loci, n_pheno)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize MLP weights
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize linear weights for skip connection
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Extract selected features for the MLP path
        x_selected = torch.index_select(x, 1, torch.tensor(self.selected_indices,
                                                        device=x.device))

        # Process through MLP
        mlp_output = self.mlp(x_selected)

        # Process all features through linear layer
        linear_output = self.linear(x)

        # Combine outputs
        return mlp_output + linear_output

###################################

class HybridSkipModelPruned(nn.Module):
    """
    Hybrid model for genomic prediction:
    - Selected features go through the MLP path ONLY
    - Unselected features go through a linear skip connection ONLY

    Args:
        n_loci: Total number of loci/features
        selected_indices: Indices of informative features for non-linear processing
        latent_space_g: Hidden layer size for MLP
        n_pheno: Number of phenotypes to predict
    """
    def __init__(self, n_loci, selected_indices, latent_space_g, n_pheno):
        super().__init__()

        # Store feature indices
        self.selected_indices = selected_indices

        # Create tensor of all indices
        all_indices = set(range(n_loci))

        # Create tensor of unselected indices
        self.unselected_indices = list(all_indices - set(selected_indices))

        # MLP for selected features
        self.mlp = nn.Sequential(
            nn.Linear(len(selected_indices), latent_space_g),
            nn.BatchNorm1d(latent_space_g, momentum=0.8),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(latent_space_g, latent_space_g),
            nn.BatchNorm1d(latent_space_g, momentum=0.8),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(latent_space_g, n_pheno)
        )

        # Linear layer for unselected features only
        self.linear = nn.Linear(len(self.unselected_indices), n_pheno)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize MLP weights
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize linear weights for skip connection
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Create tensors for device compatibility
        selected_indices_tensor = torch.tensor(self.selected_indices, device=x.device)
        unselected_indices_tensor = torch.tensor(self.unselected_indices, device=x.device)

        # Extract selected features for the MLP path
        x_selected = torch.index_select(x, 1, selected_indices_tensor)

        # Extract unselected features for the linear path
        x_unselected = torch.index_select(x, 1, unselected_indices_tensor)

        # Process selected features through MLP
        mlp_output = self.mlp(x_selected)

        # Process unselected features through linear layer
        linear_output = self.linear(x_unselected)

        # Combine outputs
        return mlp_output + linear_output

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
          max_epochs=200,
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

def get_selected_features(model, threshold=1e-4, max_features=None, n_loci=2000):
    """
    Extract the indices of features that have significant weights
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

    # Sort features by importance
    sorted_indices = np.argsort(-feature_importance)

    # Select features
    if max_features is not None:
        # Limit to top max_features
        selected_indices = sorted_indices[:max_features]
    else:
        # Or select based on threshold
        selected_indices = np.where(feature_importance > threshold)[0]

    # Ensure indices are within bounds
    selected_indices = selected_indices[selected_indices < n_loci]

    print(f"Selected {len(selected_indices)} features out of {len(feature_importance)}")

    return selected_indices, feature_importance[selected_indices]

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
    corr, _ = pearsonr(true_phenotypes[:, 1], predicted_phenotypes[:, 1])

    return corr


def save_feature_selection_info(model, threshold=1e-4, file_name=None):
    """
    Save feature selection information to a CSV file

    Args:
        model: Trained linear model
        threshold: Weight magnitude threshold for significance
        file_name: Output CSV file name
    """
    # Get the first layer weights
    weights = None
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.detach().cpu().numpy()
            break

    if weights is None:
        raise ValueError("Could not find weight parameters in the model")

    # For each feature (column), calculate the maximum absolute weight
    feature_importance = np.max(np.abs(weights), axis=0)

    # Create a DataFrame
    import pandas as pd

    features_df = pd.DataFrame({
        'feature_index': range(len(feature_importance)),
        'importance': feature_importance,
        'selected': feature_importance > threshold
    })

    # Sort by importance (descending)
    features_df = features_df.sort_values('importance', ascending=False)

    # Save to CSV
    features_df.to_csv(file_name, index=False)
    print(f"Feature selection info saved to {file_name}")

    return features_df
##########################################################################################
##########################################################################################

def train_gpnet(model, train_loader, test_loader=None,
                         n_loci=None,
                         n_alleles=2,
                         max_epochs=250,  # Set a generous upper limit
                         patience=30,      # Number of epochs to wait for improvement
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

def run_lasso_mlp_pipeline(l1_weight=None, feature_threshold=None,
                          mlp_hidden_size=None, max_selected_features=None):
    """
    Run a two-stage pipeline:
    1. Train a linear model with L1 regularization (LASSO)
    2. Select important features
    3. Train an MLP on the selected features
    """
    # Stage 1: Train linear model with L1 regularization
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

    features_df = save_feature_selection_info(linear_model, threshold=1e-4,
                                            file_name=base_file_name_out)

    # Stage 2: Feature selection
    selected_indices, feature_importance = get_selected_features(
        linear_model, threshold=feature_threshold
    )

    # Limit the number of features if needed
    if max_selected_features and len(selected_indices) > max_selected_features:
        # Sort by importance and take top features
        sorted_indices = np.argsort(-feature_importance)
        selected_indices = selected_indices[sorted_indices[:max_selected_features]]
        feature_importance = feature_importance[sorted_indices[:max_selected_features]]
        print(f"Limited selection to top {max_selected_features} features")

    # Convert selected_indices to a list of integers
    selected_indices = selected_indices.tolist()
    print(f"Selected {len(selected_indices)} features")
    print(f"Max index in selected indices: {max(selected_indices)}")

    # Stage 3: Train MLP on original data but with special handling of selected features
    mlp_model = HybridSkipModelPruned(
        n_loci=n_loci,
        selected_indices=selected_indices,
        latent_space_g=mlp_hidden_size,
        n_pheno=n_phen
    ).to(device)

    mlp_model, best_loss_mlp, history_mlp = train_gpnet(
        model=mlp_model,
        train_loader=train_loader_gp,  # Use original loader with all features
        test_loader=test_loader_gp,    # Use original loader with all features
        n_loci=n_loci,                 # Use original number of loci
        device=device,
        l1_lambda=0
    )

    # Evaluate both models
    linear_corr = evaluate_model(linear_model, test_loader_gp)
    mlp_corr = evaluate_model(mlp_model, test_loader_gp)

    print(f"Linear model correlation: {linear_corr:.4f}")
    print(f"MLP model correlation: {mlp_corr:.4f}")

    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.axhline(y=feature_threshold, color='r', linestyle='-', label=f'Threshold: {feature_threshold}')
    plt.title('Feature Importance from LASSO')
    plt.xlabel('Feature Index')
    plt.ylabel('Max Absolute Weight')
    plt.legend()
    plt.tight_layout()
    plt.show()

    results = {
        'linear_model': linear_model,
        'mlp_model': mlp_model,
        'selected_indices': selected_indices,
        'feature_importance': feature_importance,
        'linear_corr': linear_corr,
        'mlp_corr': mlp_corr,
        'history_linear': history_linear,
        'history_mlp': history_mlp
    }

    return results

##########################################################################################
##########################################################################################



#####################################################################################################################
#####################################################################################################################

def main():
    results = run_lasso_mlp_pipeline(
        l1_weight=0.001,               # L1 regularization strength
        feature_threshold=0.01,       # Threshold for feature selection
        mlp_hidden_size=4096,           # Size of hidden layers in MLP
        max_selected_features=400     # Maximum number of features to use
    )

if __name__ == "__main__":
    main()