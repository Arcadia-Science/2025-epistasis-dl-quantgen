#!/usr/bin/env python3

import gpatlas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#variables
n_phen=2
n_loci = 200
n_alleles = 2
latent_space_g = 3500
EPS = 1e-15


batch_size = 1000
num_workers = 3

base_file_name = 'test_sim_WF_1kbt_100k_100sites_'
base_file_name_out = 'experiments/test_sim_WF_1kbt_100k_100sites_gplinear_kl01_mse'


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
          kl_weight = 0.01,
          learning_rate=0.001,
          max_epochs=10,
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

            mse_loss = F.mse_loss(output + EPS, phens + EPS)
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
                    mse_loss = F.mse_loss(output + EPS, phens + EPS)
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

def visualize_loss_landscape(model, train_loader, test_loader, loss_fn, device, n_points=31, alpha_range=(-1, 1), beta_range=(-1, 1)):
    """
    Visualize the train and test loss landscapes of a model using filter-normalized random directions.

    Args:
        model: The trained model
        train_loader: DataLoader containing training data
        test_loader: DataLoader containing test data
        loss_fn: Loss function to use for evaluation
        device: Device to run the model on
        n_points: Resolution of the grid (n_points x n_points)
        alpha_range: Range for the first direction
        beta_range: Range for the second direction
    """
    import copy
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Store original model parameters
    theta = [p.data.clone() for p in model.parameters()]

    # Generate two random filter-normalized directions
    def get_random_direction(params):
        return [torch.randn_like(p) for p in params]

    def normalize_direction(direction, params):
        for d, p in zip(direction, params):
            if len(d.shape) > 1:  # For fully connected or conv layers
                # Apply filter-wise normalization
                # For FC layers, each output neuron is treated as a "filter"
                d_norm = torch.norm(d.view(d.shape[0], -1), dim=1)
                p_norm = torch.norm(p.view(p.shape[0], -1), dim=1)

                # Normalize each filter to have same norm as in params
                for i in range(d.shape[0]):
                    if d_norm[i] > 0:
                        d[i] = d[i] * (p_norm[i] / d_norm[i])
        return direction

    # Generate random directions once to use for both train and test
    dir1 = get_random_direction(theta)
    dir2 = get_random_direction(theta)

    # Normalize directions
    dir1 = normalize_direction(dir1, theta)
    dir2 = normalize_direction(dir2, theta)

    # Create a grid for the visualization
    alpha = np.linspace(alpha_range[0], alpha_range[1], n_points)
    beta = np.linspace(beta_range[0], beta_range[1], n_points)

    # Create surfaces for both train and test
    train_loss_surface = np.zeros((n_points, n_points))
    test_loss_surface = np.zeros((n_points, n_points))

    # Create a temporary model for evaluation
    temp_model = copy.deepcopy(model)

    # Function to evaluate loss on a dataloader
    def evaluate_loss(data_loader, max_batches=5):
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for phens, gens in data_loader:
                phens = phens.to(device)
                gens = gens[:, :n_loci * n_alleles].to(device)

                # Forward pass
                outputs = temp_model(gens)

                # Calculate loss
                batch_loss = loss_fn(outputs + EPS, phens + EPS)
                total_loss += batch_loss.item()
                n_batches += 1

                # For speed, limit the number of batches
                if n_batches >= max_batches:
                    break

        return total_loss / n_batches if n_batches > 0 else 0

    # Sample the loss landscape
    print("Sampling train and test loss landscapes...")
    for i, a in enumerate(alpha):
        for j, b in enumerate(beta):
            # Update model parameters with the direction
            for param, t, d1, d2 in zip(temp_model.parameters(), theta, dir1, dir2):
                param.data = t + a * d1 + b * d2

            # Evaluate loss on train data
            temp_model.eval()
            train_loss_surface[i, j] = evaluate_loss(train_loader)

            # Evaluate loss on test data (using the same model parameters)
            test_loss_surface[i, j] = evaluate_loss(test_loader)

            # Update progress
            if (i * n_points + j + 1) % (n_points * 5) == 0:
                print(f"Progress: {(i * n_points + j + 1) / (n_points * n_points) * 100:.1f}%")

    # Create a figure with two 3D subplots side by side
    fig = plt.figure(figsize=(20, 8))

    # Train loss surface - 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(alpha, beta)
    surf1 = ax1.plot_surface(X, Y, train_loss_surface, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax1.set_xlabel('Direction 1')
    ax1.set_ylabel('Direction 2')
    ax1.set_zlabel('Loss')
    ax1.set_title('Train Loss Landscape')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Test loss surface - 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, test_loss_surface, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax2.set_xlabel('Direction 1')
    ax2.set_ylabel('Direction 2')
    ax2.set_zlabel('Loss')
    ax2.set_title('Test Loss Landscape')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # Adjust z-axis limits to be the same for both plots for fair comparison
    z_min = min(train_loss_surface.min(), test_loss_surface.min())
    z_max = max(train_loss_surface.max(), test_loss_surface.max())
    ax1.set_zlim(z_min, z_max)
    ax2.set_zlim(z_min, z_max)

    plt.suptitle('Comparison of Train and Test Loss Landscapes', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{base_file_name_out}_loss_landscape_comparison_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create contour plots for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Train loss contour
    contour1 = ax1.contour(X, Y, train_loss_surface, levels=20)
    fig.colorbar(contour1, ax=ax1)
    ax1.set_xlabel('Direction 1')
    ax1.set_ylabel('Direction 2')
    ax1.set_title('Train Loss Landscape Contour')

    # Test loss contour
    contour2 = ax2.contour(X, Y, test_loss_surface, levels=20)
    fig.colorbar(contour2, ax=ax2)
    ax2.set_xlabel('Direction 1')
    ax2.set_ylabel('Direction 2')
    ax2.set_title('Test Loss Landscape Contour')

    plt.suptitle('Comparison of Train and Test Loss Landscapes (Contour)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{base_file_name_out}_loss_landscape_comparison_contour.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Return the loss surface data for further analysis
    return {
        'alpha': alpha,
        'beta': beta,
        'train_loss_surface': train_loss_surface,
        'test_loss_surface': test_loss_surface
    }

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

    # Create data for the boxplot
    boxplot_data = []
    for i in range(0, n_phen, 1):
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
    #plt.savefig(f'{base_file_name_out}_pheno_corr.png')

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
    #results_df.to_csv(f'{base_file_name_out}_pheno_corr.csv', index=False)


    #generate loss landscape
# Add loss landscape visualization after training
    print("Generating loss landscape visualization...")
    loss_landscape_data = visualize_loss_landscape(
        model=model,
        train_loader=train_loader_gp,
        test_loader=test_loader_gp,
        loss_fn=F.mse_loss,  # Using L1 loss as in your training
        device=device,
        n_points=21,  # Lower resolution for faster computation
        alpha_range=(-1, 1),
        beta_range=(-1, 1)
    )

    return best_loss_gp

#####################################################################################################################
#####################################################################################################################

def main():
    best_loss_gp = run_full_pipeline()
    print(f"Final loss: {best_loss_gp}")

if __name__ == "__main__":
    main()