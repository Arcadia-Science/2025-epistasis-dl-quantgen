#!/usr/bin/env python3

import sys
import time as tm
import argparse
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data.dataset import Dataset
import optuna
from pathlib import Path
from typing import cast
import h5py
import pickle
import json
from datetime import datetime
import traceback
#import matplotlib.pyplot as plt
#import scipy as sc


batch_size = 128
num_workers = 10
num_epochs_final = 25
n_trials_optuna = 30
##########################################################################################
##########################################################################################

n_geno = 100000
n_alleles = 2


##########################################################################################
##########################################################################################


class BaseDataset(Dataset):
    def __init__(self, hdf5_path: Path) -> None:
        self.h5 = h5py.File(hdf5_path, "r")

        self._strain_group = cast(h5py.Group, self.h5["strains"])
        self.strains: list[str] = list(self._strain_group.keys())

    def __len__(self) -> int:
        return len(self._strain_group)

###########

class GenoDataset(BaseDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        strain = self.strains[idx]

        strain_data = cast(Dataset, self._strain_group[strain])

        # Note: genotype is being cast as float32 here, reasons not well understood.
        gens = torch.tensor(strain_data["genotype"][:], dtype=torch.float32).flatten()

        return  gens


##########################################################################################
##########################################################################################

train_data_geno = GenoDataset(snakemake.input['input_train_data'])
test_data_geno = GenoDataset(snakemake.input['input_test_data'])

##########################################################################################
##########################################################################################

train_loader_geno = torch.utils.data.DataLoader(
    dataset=train_data_geno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
test_loader_geno = torch.utils.data.DataLoader(
    dataset=test_data_geno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)

##########################################################################################
##########################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################################
##########################################################################################

# gencoder
class GQ_net(nn.Module):
    def __init__(self, n_loci=None, N=None):
        super().__init__()
        #if N is None:
        #    N = latent_space_g
        if n_loci is None:
            n_loci = n_geno * n_alleles

        batchnorm_momentum = 0.8
        g_latent_dim = N
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=N),
            nn.BatchNorm1d(N, momentum=0.8),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=g_latent_dim),
            nn.BatchNorm1d(g_latent_dim, momentum=0.8),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# gendecoder
class GP_net(nn.Module):
    def __init__(self, n_loci=None, N=None):
        super().__init__()
        #if N is None:
        #    N = latent_space_g
        if n_loci is None:
            n_loci = n_geno * n_alleles

        batchnorm_momentum = 0.8
        g_latent_dim = N
        self.encoder = nn.Sequential(
            nn.Linear(in_features=g_latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=n_loci),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

##########################################################################################
##########################################################################################

def objective(trial: optuna.Trial,
             train_loader,
             test_loader,
             n_geno: int,
             n_alleles: int,
             device: torch.device) -> float:

    # Hyperparameters to optimize
    latent_space_g = trial.suggest_int('latent_space_g', 100, 3500)
    gen_noise = trial.suggest_float('gen_noise', 0.1, 0.8)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 6, 6)
    weights_regularization = trial.suggest_float('weights_regularization', 1e-6, 1e-1, log=True)

    # Constants
    EPS = 1e-15
    adam_b = (0.5, 0.999)
    n_loci = n_geno * n_alleles

    # Initialize models
    GQ = GQ_net(n_loci=n_loci, N=latent_space_g).to(device)
    GP = GP_net(n_loci=n_loci, N=latent_space_g).to(device)

    # Optimizers
    optim_GQ_enc = torch.optim.Adam(GQ.parameters(), lr=learning_rate, betas=adam_b)
    optim_GP_dec = torch.optim.Adam(GP.parameters(), lr=learning_rate, betas=adam_b)

    # Training loop
    for epoch in range(num_epochs):
        GQ.train()
        GP.train()

        epoch_losses = []

        for i, gens in enumerate(train_loader):
            batch_size = gens.shape[0]

            GP.zero_grad()
            GQ.zero_grad()

            # Apply noise
            noise_prob = 1 - gen_noise
            pos_noise = np.random.binomial(1, noise_prob / 2, gens.shape)
            neg_noise = np.random.binomial(1, noise_prob / 2, gens.shape)

            noise_gens = torch.tensor(
                np.where((gens + pos_noise - neg_noise) > 0, 1, 0),
                dtype=torch.float32
            ).to(device)

            gens = gens.to(device)

            z_sample = GQ(noise_gens)
            X_sample = GP(z_sample)

            g_recon_loss = F.binary_cross_entropy(X_sample + EPS, gens + EPS)

            # L1 and L2 regularization
            l1_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 1)
            l2_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 2)

            total_loss = g_recon_loss + l1_reg * weights_regularization + l2_reg * weights_regularization

            total_loss.backward()
            optim_GQ_enc.step()
            optim_GP_dec.step()

            epoch_losses.append(float(total_loss.detach()))

        # Test set evaluation
        if epoch % 1 == 0:
            GQ.eval()
            GP.eval()
            test_losses = []

            with torch.no_grad():
                for gens in test_loader:
                    gens = gens.to(device)

                    z_sample = GQ(gens)
                    X_sample = GP(z_sample)

                    test_loss = F.binary_cross_entropy(X_sample + EPS, gens + EPS)
                    test_losses.append(float(test_loss))

            avg_test_loss = np.mean(test_losses)
            print(f"Epoch {epoch}, Test Loss: {avg_test_loss:.4f}")

    return avg_test_loss

##########################################################################################
##########################################################################################


def train_final_model(best_params, train_loader, test_loader, n_geno, n_alleles, device, timestamp):
    # Set up parameters from best trial
    latent_space_g = best_params['latent_space_g']
    gen_noise = best_params['gen_noise']
    learning_rate = best_params['learning_rate']
    num_epochs = num_epochs_final
    weights_regularization = best_params['weights_regularization']

    # Constants
    EPS = 1e-15
    adam_b = (0.5, 0.999)
    n_loci = n_geno * n_alleles

    # Initialize models
    GQ = GQ_net(n_loci=n_loci, N=latent_space_g).to(device)
    GP = GP_net(n_loci=n_loci, N=latent_space_g).to(device)

    # Optimizers
    optim_GQ_enc = torch.optim.Adam(GQ.parameters(), lr=learning_rate, betas=adam_b)
    optim_GP_dec = torch.optim.Adam(GP.parameters(), lr=learning_rate, betas=adam_b)

    best_test_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        GQ.train()
        GP.train()

        epoch_losses = []

        for i, gens in enumerate(train_loader):
            GP.zero_grad()
            GQ.zero_grad()

            # Apply noise
            noise_prob = 1 - gen_noise
            pos_noise = np.random.binomial(1, noise_prob / 2, gens.shape)
            neg_noise = np.random.binomial(1, noise_prob / 2, gens.shape)

            noise_gens = torch.tensor(
                np.where((gens + pos_noise - neg_noise) > 0, 1, 0),
                dtype=torch.float32
            ).to(device)

            gens = gens.to(device)

            z_sample = GQ(noise_gens)
            X_sample = GP(z_sample)

            g_recon_loss = F.binary_cross_entropy(X_sample + EPS, gens + EPS)

            l1_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 1)
            l2_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 2)

            total_loss = g_recon_loss + l1_reg * weights_regularization + l2_reg * weights_regularization

            total_loss.backward()
            optim_GQ_enc.step()
            optim_GP_dec.step()

            epoch_losses.append(float(total_loss.detach()))

        # Test set evaluation
        GQ.eval()
        GP.eval()
        test_losses = []

        with torch.no_grad():
            for gens in test_loader:
                gens = gens.to(device)
                z_sample = GQ(gens)
                X_sample = GP(z_sample)
                test_loss = F.binary_cross_entropy(X_sample + EPS, gens + EPS)
                test_losses.append(float(test_loss))

        avg_test_loss = np.mean(test_losses)
        print(f"Final Training - Epoch {epoch}, Test Loss: {avg_test_loss:.4f}")

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            #save_path = f'gpatlas/optuna/best_encoder_gg_{timestamp}.pt'
            save_path_enc = snakemake.output['gg_encoder_optimized']
            save_path_dec = snakemake.output['gg_decoder_optimized']
            torch.save({
                'epoch': epoch,
                'model_state_dict': GQ.state_dict(),
                'optimizer_state_dict': optim_GQ_enc.state_dict(),
                'loss': best_test_loss,
                'hyperparameters': best_params
            }, save_path_enc)
            torch.save({
                'epoch': epoch,
                'model_state_dict': GP.state_dict(),
                'optimizer_state_dict': optim_GP_dec.state_dict(),
                'loss': best_test_loss,
                'hyperparameters': best_params
            }, save_path_dec)
            print(f"New best model saved with test loss: {avg_test_loss:.4f}")

    return GQ, GP, best_test_loss

##########################################################################################
##########################################################################################

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    #output_dir = Path('gpatlas/optuna')
    output_dir = Path(snakemake.output['optuna_gg_csv']).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create study
    study = optuna.create_study(direction='minimize')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run optimization
    n_trials = n_trials_optuna

    try:
        # Create a list to store results as we go
        trial_results = []

        study.optimize(
            lambda trial: objective(
                trial=trial,
                train_loader=train_loader_geno,
                test_loader=test_loader_geno,
                n_geno=n_geno,
                n_alleles=n_alleles,
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

        print("\nStudy completed!")
        print(f"Best parameters found: {study.best_params}")
        print(f"Best value achieved: {study.best_value}")

        # Save results to CSV
        results_df = pd.DataFrame(trial_results)
        #results_df.to_csv(f'gpatlas/optuna/optuna_trials_gg_{timestamp}.csv', index=False)
        results_df.to_csv(snakemake.output['optuna_gg_csv'], index=False)

        # Save detailed study information to JSON
        study_info = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': n_trials,
            'datetime': timestamp,
            'all_trials': trial_results
        }

        #with open(f'gpatlas/optuna/optuna_study_gg_{timestamp}.json', 'w') as f:
        with open(snakemake.output['optuna_gg_json'], 'w') as f:
            json.dump(study_info, f, indent=4)

        print(f"\nResults saved to:")
        print(snakemake.output['optuna_gg_csv'])
        print(snakemake.output['optuna_gg_json'])

        # Train final model with best parameters
        if study.best_params is not None:
            print("\nTraining final model with best parameters...")
            GQ, GP, final_loss = train_final_model(
                study.best_params,
                train_loader_geno,
                test_loader_geno,
                n_geno=n_geno,
                n_alleles=n_alleles,
                device=device,
                timestamp=timestamp
            )

            print(f"\nFinal model training completed!")
            print(f"Final test loss: {final_loss:.4f}")

            # Update study info with final model results
            study_info['final_model_loss'] = float(final_loss)
            #with open(f'gpatlas/optuna/optuna_study_gg_{timestamp}.json', 'w') as f:
            with open(snakemake.output['optuna_gg_json'], 'w') as f:
                json.dump(study_info, f, indent=4)

        return study.best_params, study.best_value, trial_results, timestamp

    except Exception as e:
        print(f"Error details: {str(e)}")
        traceback.print_exc()

        # Save error information
        #with open(f'gpatlas/optuna/optuna_error_gg_{timestamp}.txt', 'w') as f:
        error_log = Path(snakemake.output['optuna_gg_csv']).parent / f'optuna_error_gg_{timestamp}.txt'
        with open(error_log, 'w') as f:
            f.write(f"Error occurred: {str(e)}\n")
            traceback.print_exc(file=f)

        return None, None, None, timestamp

if __name__ == "__main__":
    best_params, best_value, trial_results, timestamp = main()

    if best_params is not None:
        print("\nExecution completed successfully!")
        print(f"Best parameters: {best_params}")
        print(f"Best value: {best_value}")
    else:
        print("\nExecution failed. Check error logs.")