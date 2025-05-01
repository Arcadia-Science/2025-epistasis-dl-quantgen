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

#random.seed(42)

#variables
n_phen=2
n_loci = 2000
n_alleles = 2
latent_space_g = 3500
EPS = 1e-15


batch_size = 128
num_workers = 3

base_file_name = 'gpnet/input_data/qhaplo_100qtl_1000marker_10000n_'
base_file_name_out = 'gphybrid/qhaplo_100qtl_1000marker_10000n_featseln.csv'


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

class TwoPhaseHybridModel(nn.Module):
    """
    Hybrid model with two-phase training capability:
    - Phase 1: Only MLP for selected features is trained
    - Phase 2: Linear layer for unselected features is trained

    Args:
        n_loci: Total number of loci/features
        selected_indices: Indices of informative features for MLP
        latent_space_g: Hidden layer size for MLP
        n_pheno: Number of phenotypes to predict
        training_phase: Which phase of training (1 or 2)
    """
    def __init__(self, n_loci, selected_indices, latent_space_g, n_pheno, training_phase=1):
        super().__init__()

        # Store feature indices and training phase
        self.selected_indices = selected_indices
        self.training_phase = training_phase

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

    def set_training_phase(self, phase):
        """
        Set the current training phase
        Phase 1: Train only MLP
        Phase 2: Train only linear layer
        """
        self.training_phase = phase

        # Freeze/unfreeze parameters based on phase
        if phase == 1:
            # Freeze linear layer
            for param in self.linear.parameters():
                param.requires_grad = False
            # Unfreeze MLP
            for param in self.mlp.parameters():
                param.requires_grad = True
        elif phase == 2:
            # Freeze MLP
            for param in self.mlp.parameters():
                param.requires_grad = False
            # Unfreeze linear layer
            for param in self.linear.parameters():
                param.requires_grad = True

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

        # Combine outputs based on training phase
        if self.training_phase == 1 and self.training:
            return mlp_output  # Only MLP output during phase 1 training
        else:
            return mlp_output + linear_output  # Combined output otherwise

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

def train_two_phase_model(model, train_loader, test_loader,
                         n_loci=None, n_alleles=2,
                         phase1_max_epochs=250, phase1_patience=30,
                         phase2_max_epochs=100, phase2_patience=20,
                         min_delta=0.001,
                         phase1_lr=0.01, phase2_lr=0.01,
                         weight_decay=1e-7, device=device):
    """
    Train model in two phases:
    Phase 1: Train only the MLP components
    Phase 2: Train only the linear/residual components
    """
    # Phase 1: Train MLP
    print("=" * 50)
    print("Phase 1: Training MLP components")
    print("=" * 50)

    # Set model to phase 1
    model.set_training_phase(1)

    # Initialize optimizer with only MLP parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase1_lr,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = {
        'phase1_train_loss': [],
        'phase1_test_loss': [],
        'phase1_epochs_trained': 0,
        'phase2_train_loss': [],
        'phase2_test_loss': [],
        'phase2_epochs_trained': 0,
    }

    # Early stopping variables
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Phase 1 Training loop
    for epoch in range(phase1_max_epochs):
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

            # Loss
            loss = F.l1_loss(output + EPS, phens + EPS)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['phase1_train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for phens, gens in test_loader:
                phens = phens.to(device)
                gens = gens[:, : n_loci * n_alleles]
                gens = gens.to(device)

                # For evaluation, we always use the full model output
                model.training = False
                output = model(gens)
                model.training = True

                loss = F.l1_loss(output + EPS, phens + EPS)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        history['phase1_test_loss'].append(avg_test_loss)

        print(f'Phase 1 - Epoch: {epoch+1}/{phase1_max_epochs}, '
              f'Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')

        # Update learning rate
        scheduler.step(avg_test_loss)

        # Check for improvement
        if avg_test_loss < (best_loss - min_delta):
            best_loss = avg_test_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            print(f"New best model with test loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best: {best_loss:.6f})")

        # Early stopping check
        if patience_counter >= phase1_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Record how many epochs were used in phase 1
    history['phase1_epochs_trained'] = epoch + 1

    # Restore best model from phase 1
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Phase 2: Train Linear Components
    print("=" * 50)
    print("Phase 2: Training Linear/Residual components")
    print("=" * 50)

    # Set model to phase 2
    model.set_training_phase(2)

    # Initialize new optimizer with only linear parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase2_lr,
        weight_decay=weight_decay
    )

    # Reset scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Reset early stopping variables
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Phase 2 Training loop
    for epoch in range(phase2_max_epochs):
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

            # Loss
            loss = F.l1_loss(output + EPS, phens + EPS)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['phase2_train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for phens, gens in test_loader:
                phens = phens.to(device)
                gens = gens[:, : n_loci * n_alleles]
                gens = gens.to(device)

                output = model(gens)
                loss = F.l1_loss(output + EPS, phens + EPS)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        history['phase2_test_loss'].append(avg_test_loss)

        print(f'Phase 2 - Epoch: {epoch+1}/{phase2_max_epochs}, '
              f'Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')

        # Update learning rate
        scheduler.step(avg_test_loss)

        # Check for improvement
        if avg_test_loss < (best_loss - min_delta):
            best_loss = avg_test_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            print(f"New best model with test loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best: {best_loss:.6f})")

        # Early stopping check
        if patience_counter >= phase2_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Record how many epochs were used in phase 2
    history['phase2_epochs_trained'] = epoch + 1

    # Restore best model from phase 2
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Set model back to normal operation (all components active)
    model.set_training_phase(0)

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
    mlp_model = TwoPhaseHybridModel(
        n_loci=n_loci,
        selected_indices=selected_indices,
        latent_space_g=mlp_hidden_size,
        n_pheno=n_phen,
        training_phase=1  # Start in phase 1
    ).to(device)

    mlp_model, best_loss_mlp, history_mlp = train_two_phase_model(
        model=mlp_model,
        train_loader=train_loader_gp,
        test_loader=test_loader_gp,
        n_loci=n_loci,
        device=device,
        phase1_max_epochs=200,  # Maximum epochs for phase 1
        phase1_patience=30,     # Early stopping patience for phase 1
        phase2_max_epochs=100,  # Maximum epochs for phase 2
        phase2_patience=30      # Early stopping patience for phase 2
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
        feature_threshold=0.03,       # Threshold for feature selection
        mlp_hidden_size=4096,           # Size of hidden layers in MLP
        #max_selected_features=400     # Maximum number of features to use
    )

if __name__ == "__main__":
    main()