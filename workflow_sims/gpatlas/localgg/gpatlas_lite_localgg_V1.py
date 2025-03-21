#!/usr/bin/env python3
import gpatlas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset

from pathlib import Path
from typing import cast
import h5py
import time as tm

#variables

n_loci = 100000
n_alleles = 2
window_size = 200
window_stride = 10
glatent = 3000
input_length = n_loci * n_alleles
n_out_channels = 7

n_epochs = 50
batch_size = 128
num_workers = 3

base_file_name = 'gpatlas_input/test_sim_WF_1kbt_10000n_5000000bp_'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################################################################################
#####################################################################################################################

loaders = gpatlas.create_data_loaders(base_file_name, batch_size=128, num_workers=3, shuffle=True)
train_loader_geno = loaders['train_loader_geno']
test_loader_geno = loaders['test_loader_geno']

#####################################################################################################################
#####################################################################################################################

start_time = tm.time()
def train_baseline_model(model, train_loader, test_loader=None,
                         epochs=n_epochs,
                         learning_rate=0.001,
                         weight_decay=1e-5,
                         device=device,
                         patience=6,
                         min_delta=0.001):
    """
    Train the baseline LD-aware autoencoder with no special weighting

    Args:
        model: The autoencoder model
        train_loader: DataLoader with training data
        test_loader: Optional DataLoader with test data
        epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        weight_decay: Weight decay for regularization
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    global start_time
    # Initialize optimizer with proper weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
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
    for epoch in range(epochs):

        # Training
        model.train()
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            # Get data
            if isinstance(data, (list, tuple)):
                            # If it's a tuple or list, take the first element
                            data = data[0]

            data = data.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)

            # Standard BCE loss - NO WEIGHTING
            loss = F.binary_cross_entropy(output, data)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'Epoch: {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}')

        # Validation
        if test_loader is not None:
            model.eval()
            test_loss = 0

            with torch.no_grad():
                for data in test_loader:
                    if isinstance(data, (list, tuple)):
                    # If it's a tuple or list, take the first element
                        data = data[0]
                    data = data.to(device)

                    output = model(data)
                    # Standard BCE loss for evaluation
                    test_loss += F.binary_cross_entropy(output, data).item()

            avg_test_loss = test_loss / len(test_loader)
            history['test_loss'].append(avg_test_loss)
            cur_time = tm.time() - start_time
            start_time = tm.time()

            print(f'Epoch: {epoch+1}/{epochs}, Test Loss: {avg_test_loss:.6f}, Epoch time: {cur_time}')

            # Update learning rate
            scheduler.step(avg_test_loss)

            #early stoppage loop
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

model = gpatlas.LDGroupedAutoencoder(
    input_length=input_length,
    loci_count=n_loci,
    window_size=window_size,
    window_stride=window_stride,
    latent_dim=glatent,
    n_out_channels=n_out_channels)

#train and save full model
model, best_loss, history = train_baseline_model(model, train_loader_geno,test_loader=test_loader_geno, device=device)
torch.save(model.state_dict(), "localgg/localgg_autenc_1kbt_V1_state_dict.pt")
print(f'saved best model with loss: {best_loss}')

#save gg encoder only for G->P
encoder = gpatlas.LDEncoder(model)
torch.save(encoder.state_dict(), "localgg/localgg_enc_1kbt_V1_state_dict.pt")