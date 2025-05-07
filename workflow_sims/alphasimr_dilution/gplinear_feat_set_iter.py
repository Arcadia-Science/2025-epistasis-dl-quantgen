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
import copy

random.seed(42)

#variables
n_phen=2
n_loci = 5000
n_alleles = 2
EPS = 1e-15

batch_size = 128
num_workers = 3

base_file_name = 'gpnet/input_data/qhaplo_100qtl_2500marker_10000n_'
base_file_name_out = 'qhaplo_100qtl_2500marker_100000n'

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
          max_epochs=150,
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

def get_feature_importance(model):
    """
    Extract feature importance scores based on model weights

    Args:
        model: Trained linear model

    Returns:
        feature_importance: Array of importance scores for each feature
        weights: Raw weight matrix
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

    return feature_importance, weights

##########################################################################################
##########################################################################################

def get_selected_features(model, percent_to_keep=None, max_features=None):
    """
    Select the top X% of features by importance

    Args:
        model: Trained linear model
        percent_to_keep: Percentage of features to keep (0-100)
        max_features: Maximum number of features to keep

    Returns:
        selected_indices: Indices of selected features
        feature_importance: Importance scores for all features
    """
    # Get feature importance
    feature_importance, weights = get_feature_importance(model)

    # Sort features by importance (descending)
    sorted_indices = np.argsort(-feature_importance)

    # Determine how many features to keep
    if percent_to_keep is not None:
        num_to_keep = max(1, int(len(feature_importance) * percent_to_keep / 100))
    else:
        num_to_keep = len(feature_importance)  # Keep all if no percentage specified

    # Apply maximum limit if specified
    if max_features is not None:
        num_to_keep = min(num_to_keep, max_features)

    # Select the top features
    selected_indices = sorted_indices[:num_to_keep]

    print(f"Selected {len(selected_indices)} features out of {len(feature_importance)}")

    return selected_indices, feature_importance

##########################################################################################
##########################################################################################

def evaluate_model(model, test_loader, num_features=None):
    """
    Evaluate model performance with detailed metrics
    """
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
    mae_values = []

    for i in range(n_phen):
        corr, _ = pearsonr(true_phenotypes[:, i], predicted_phenotypes[:, i])
        mae = np.mean(np.abs(true_phenotypes[:, i] - predicted_phenotypes[:, i]))

        correlations.append(corr)
        mae_values.append(mae)

    # Create a detailed DataFrame with all results
    results_df = pd.DataFrame({
        'trait_number': range(1, n_phen + 1),
        'pearson_correlation': correlations,
        'mae': mae_values,
        'num_features': num_features if num_features is not None else "all",
        'true_mean': [np.mean(true_phenotypes[:, i]) for i in range(n_phen)],
        'pred_mean': [np.mean(predicted_phenotypes[:, i]) for i in range(n_phen)],
        'true_std': [np.std(true_phenotypes[:, i]) for i in range(n_phen)],
        'pred_std': [np.std(predicted_phenotypes[:, i]) for i in range(n_phen)]
    })

    # Calculate average performance across traits
    avg_correlation = np.mean(correlations)
    avg_mae = np.mean(mae_values)

    return results_df, avg_correlation, avg_mae

##########################################################################################
##########################################################################################

def train_gpnet(model, train_loader, test_loader=None,
                         n_loci=None,
                         n_alleles=2,
                         max_epochs=200,  # Set a generous upper limit
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

def create_filtered_loader(original_loader, selected_indices):
    """
    Create a new DataLoader with only selected features
    """
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

##########################################################################################
##########################################################################################

def iterative_feature_selection(l1_weight=0.001,
                               initial_feature_percent=100,
                               reduction_factor=0.7,  # Keep 70% of features in each iteration
                               min_feature_percent=5,  # Stop when we reach 5% of features
                               max_iterations=10,
                               max_features=None,
                               min_features=10,
                               patience=3):  # Stop if no improvement for 3 iterations
    """
    Iteratively train linear models, reducing features until performance stops improving
    """
    current_feature_percent = initial_feature_percent
    current_indices = np.arange(n_loci)  # Start with all features

    best_performance = -float('inf')  # For correlation, higher is better
    best_indices = current_indices.copy()
    best_model = None
    best_iteration = 0

    iterations_without_improvement = 0
    iteration_results = []

    print(f"Starting iterative feature selection with {len(current_indices)} features")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration+1} ---")
        print(f"Training with {len(current_indices)} features ({current_feature_percent:.1f}% of original)")

        # Create filtered loaders with current feature set
        if len(current_indices) == n_loci:
            # First iteration - use original loaders
            train_loader = train_loader_gp
            test_loader = test_loader_gp
        else:
            # Subsequent iterations - use filtered data
            train_loader = create_filtered_loader(train_loader_gp, current_indices)
            test_loader = create_filtered_loader(test_loader_gp, current_indices)

        # Train linear model on current feature set
        linear_model = gplinear_kl(
            n_loci=len(current_indices),
            n_phen=n_phen
        ).to(device)

        linear_model, best_loss, history = train_gplinear_lasso(
            model=linear_model,
            train_loader=train_loader,
            test_loader=test_loader,
            l1_weight=l1_weight,
            device=device
        )

        # Evaluate model performance
        results_df, avg_correlation, avg_mae = evaluate_model(
            linear_model,
            test_loader,
            num_features=len(current_indices)
        )

        # Store results for this iteration
        iteration_results.append({
            'iteration': iteration,
            'num_features': len(current_indices),
            'feature_percent': current_feature_percent,
            'avg_correlation': avg_correlation,
            'avg_mae': avg_mae
        })

        print(f"Iteration {iteration+1}: {len(current_indices)} features, Avg Correlation: {avg_correlation:.4f}, Avg MAE: {avg_mae:.4f}")

        # Check if this is the best model so far
        if avg_correlation > best_performance:
            best_performance = avg_correlation
            best_indices = current_indices.copy()
            best_model = copy.deepcopy(linear_model)
            best_iteration = iteration
            iterations_without_improvement = 0
            print(f"New best model with {len(best_indices)} features!")
        else:
            iterations_without_improvement += 1
            print(f"No improvement for {iterations_without_improvement} iterations")

            if iterations_without_improvement >= patience:
                print(f"Early stopping: No improvement for {patience} iterations")
                break

        # Calculate new feature percentage for next iteration
        current_feature_percent *= reduction_factor

        # Don't go below minimum feature percentage
        if current_feature_percent < min_feature_percent:
            print(f"Reached minimum feature percentage ({min_feature_percent}%)")
            break

        # Select features for next iteration
        if iteration < max_iterations - 1:  # Don't do this on the last iteration
            # Get importance scores from current model
            feature_importance, _ = get_feature_importance(linear_model)

            # Calculate number of features to keep
            if max_features is not None:
                num_to_keep = min(int(len(current_indices) * reduction_factor), max_features)
            else:
                num_to_keep = int(len(current_indices) * reduction_factor)

            # Ensure we don't go below minimum number of features
            num_to_keep = max(num_to_keep, min_features)

            # Sort features by importance and keep top ones
            sorted_indices = np.argsort(-feature_importance)
            top_local_indices = sorted_indices[:num_to_keep]

            # Convert local indices back to original feature space
            current_indices = current_indices[top_local_indices]

    # Create results DataFrame
    results_df = pd.DataFrame(iteration_results)

    print("\n=== Iterative Feature Selection Results ===")
    print(f"Best model found at iteration {best_iteration+1}")
    print(f"Selected {len(best_indices)} features with avg correlation: {best_performance:.4f}")

    # Save results to CSV
    results_file = f'gphybrid/{base_file_name_out}_iterative_selection_results.csv'
    #results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    return best_indices, best_model, results_df

##########################################################################################
##########################################################################################

def train_mlp_on_selected_features(selected_indices, l1_weight=0.001, mlp_hidden_size=4096):
    """
    Train an MLP model on the optimally selected features
    """
    print(f"\n=== Training MLP on {len(selected_indices)} selected features ===")

    # Create filtered loaders with selected features
    filtered_train_loader = create_filtered_loader(train_loader_gp, selected_indices)
    filtered_test_loader = create_filtered_loader(test_loader_gp, selected_indices)

    # Train a final linear model on selected features (for comparison)
    linear_model_final = gplinear_kl(
        n_loci=len(selected_indices),
        n_phen=n_phen
    ).to(device)

    linear_model_final, _, _ = train_gplinear_lasso(
        model=linear_model_final,
        train_loader=filtered_train_loader,
        test_loader=filtered_test_loader,
        l1_weight=l1_weight,
        device=device
    )

    # Train MLP on selected features
    mlp_model = gpatlas.GP_net(
        n_loci=len(selected_indices),
        latent_space_g1=mlp_hidden_size,
        latent_space_g=mlp_hidden_size,
        n_pheno=n_phen
    ).to(device)

    mlp_model, _, _ = train_gpnet(
        model=mlp_model,
        train_loader=filtered_train_loader,
        test_loader=filtered_test_loader,
        n_loci=len(selected_indices),
        device=device,
        l1_lambda=0  # No need for L1 here as we've already done feature selection
    )

    # Evaluate both models
    linear_results, linear_corr, linear_mae = evaluate_model(
        linear_model_final,
        filtered_test_loader,
        num_features=len(selected_indices)
    )

    mlp_results, mlp_corr, mlp_mae = evaluate_model(
        mlp_model,
        filtered_test_loader,
        num_features=len(selected_indices)
    )

    print("\n=== Model Comparison ===")
    print(f"Linear Model: Avg Correlation: {linear_corr:.4f}, Avg MAE: {linear_mae:.4f}")
    print(f"MLP Model: Avg Correlation: {mlp_corr:.4f}, Avg MAE: {mlp_mae:.4f}")

    # Save results
    linear_results_file = f'gphybrid/{base_file_name_out}_optimal_linear_results_TEST.csv'
    mlp_results_file = f'gphybrid/{base_file_name_out}_optimal_mlp_results_TEST.csv'

    linear_results.to_csv(linear_results_file, index=False)
    mlp_results.to_csv(mlp_results_file, index=False)

    print(f"Linear results saved to {linear_results_file}")
    print(f"MLP results saved to {mlp_results_file}")

    return linear_model_final, mlp_model

##########################################################################################
##########################################################################################

def main():
    # Run iterative feature selection
    best_indices, best_model, results_df = iterative_feature_selection(
        l1_weight=0.001,                 # L1 regularization strength
        initial_feature_percent=100,     # Start with all features
        reduction_factor=0.7,            # Keep 70% of features in each iteration
        min_feature_percent=5,           # Stop at 5% of original features
        max_iterations=10,               # Maximum number of iterations
        max_features=None,               # No maximum limit on features
        min_features=10,                 # Minimum number of features to keep
        patience=3                       # Stop if no improvement for 3 iterations
    )

    # Train MLP on the best feature set
    linear_model_final, mlp_model = train_mlp_on_selected_features(
        selected_indices=best_indices,
        l1_weight=0.001,
        mlp_hidden_size=4096
    )

    # Create visualization of feature selection progress
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['num_features'], results_df['avg_correlation'], 'o-', linewidth=2)
    plt.axvline(x=len(best_indices), color='r', linestyle='--',
                label=f'Best model: {len(best_indices)} features')
    plt.xlabel('Number of Features')
    plt.ylabel('Average Correlation')
    plt.title('Performance vs Number of Features')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plot_file = f'gphybrid/{base_file_name_out}_feature_selection_plot.png'
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    main()