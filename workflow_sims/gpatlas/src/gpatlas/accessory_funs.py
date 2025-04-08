import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
import torch.nn.functional as F



def visualize_loss_landscape(model, train_loader, test_loader, loss_fn, device, n_points=31, alpha_range=(-1, 1), beta_range=(-1, 1),
                             n_loci=None, n_alleles=None, EPS=None, base_file_name_out=None ):
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
                gens = gens[:, :n_loci].to(device)

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
    plt.savefig(f'{base_file_name_out}_loss_landscape_3d.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'{base_file_name_out}_loss_landscape_contour.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Return the loss surface data for further analysis
    return {
        'alpha': alpha,
        'beta': beta,
        'train_loss_surface': train_loss_surface,
        'test_loss_surface': test_loss_surface
    }
