#!/usr/bin/env python3
from snakemake.script import snakemake

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from scipy import stats
from itertools import combinations
from sklearn.preprocessing import StandardScaler

# Number of train/test split iterations
n_iterations = 5  # You can adjust this number based on your needs

# Set random seed for reproducibility
np.random.seed(42)


print("Processing input files")
# Load genotype, phenotype, and effect files
geno = pd.read_csv(snakemake.input['input_geno'], index_col=0, sep=" ")
pheno = pd.read_csv(snakemake.input['input_pheno'], index_col=0, sep=" ")
eff = pd.read_csv(snakemake.input['loci_effects'], sep=" ")

print("Running ridge regression fits with multiple train/test splits")

# Convert genotype matrix (n_samples x n_markers)
Z = geno.T.values

# Determine dimensions
n_samples, n_markers = Z.shape
print(f"Original dimensions: {n_samples} samples x {n_markers} markers")

# Calculate number of pairwise interactions
n_interactions = n_markers * (n_markers - 1) // 2
print(f"Number of pairwise interactions: {n_interactions}")
print(f"Total predictors (original + interactions): {n_markers + n_interactions}")

# Define range of lambda values for cross-validation
lambda_grid = np.logspace(-5, 5, 100)  # Search over log scale

# Initialize a dictionary to collect results for each trait across iterations
trait_results = {}

# Loop through each phenotype column
for trait_index in range(pheno.shape[1]):
    print(f"Analyzing trait {trait_index+1}")

    # Extract phenotype column
    y = pheno.iloc[:, trait_index].values

    # List to store results for this trait across iterations
    iteration_results = []

    # Run multiple iterations of train/test split
    for iter_idx in range(n_iterations):
        print(f"  Iteration {iter_idx+1}/{n_iterations}")

        # Parameter setup
        n_train = int(n_samples * 0.85)  # 85% for training

        # Generate random indices for training
        train_indices = np.random.choice(n_samples, size=n_train, replace=False)
        # Test indices are the remaining ones
        test_indices = np.array([i for i in range(n_samples) if i not in train_indices])

        # Split data
        Z_train, Z_test = Z[train_indices, :], Z[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]

        # Mean-center genotypes
        p = np.mean(Z_train, axis=0)
        Z_train_centered = Z_train - p
        Z_test_centered = Z_test - p

        # Standardize (important for interactions)
        scaler = StandardScaler()
        Z_train_scaled = scaler.fit_transform(Z_train_centered)
        Z_test_scaled = scaler.transform(Z_test_centered)

        print("  Creating interaction features...")

        # Create all pairwise interactions at once
        # Get all possible pairs of marker indices
        marker_pairs = list(combinations(range(n_markers), 2))

        # Create interaction terms for training data
        Z_train_interactions = np.zeros((Z_train_scaled.shape[0], n_interactions))
        for idx, (i, j) in enumerate(marker_pairs):
            Z_train_interactions[:, idx] = Z_train_scaled[:, i] * Z_train_scaled[:, j]

        # Create interaction terms for test data
        Z_test_interactions = np.zeros((Z_test_scaled.shape[0], n_interactions))
        for idx, (i, j) in enumerate(marker_pairs):
            Z_test_interactions[:, idx] = Z_test_scaled[:, i] * Z_test_scaled[:, j]

        # Combine main effects and interactions
        Z_train_full = np.hstack([Z_train_scaled, Z_train_interactions])
        Z_test_full = np.hstack([Z_test_scaled, Z_test_interactions])

        print(f"  Combined feature matrix shape: {Z_train_full.shape}")

        # Fit Ridge regression with cross-validation
        print("  Fitting ridge regression model with all features...")
        ridge_cv = RidgeCV(alphas=lambda_grid)
        ridge_cv.fit(Z_train_full, y_train)

        # Get best lambda
        best_lambda = ridge_cv.alpha_
        print(f"  Best lambda: {best_lambda}")

        # Get coefficients
        beta_hat = ridge_cv.coef_

        # Separate coefficients for main effects and interactions
        beta_main = beta_hat[:n_markers]
        beta_interaction = beta_hat[n_markers:]

        # Predict on test set
        u_hat_main = Z_test_scaled @ beta_main
        u_hat_interaction = Z_test_interactions @ beta_interaction
        u_hat = u_hat_main + u_hat_interaction

        # Extract true additive QTL effects for relevant model
        eff_fit = eff[eff['trait'] == trait_index+1]

        # Only compare the main effect coefficients with the true effects
        # since the simulation likely doesn't have true epistatic effects
        pearson_corr_beta, _ = stats.pearsonr(eff_fit['add_eff'], beta_main)

        # Compute Pearson correlation between predicted and true phenotypes
        pearson_corr_pheno, _ = stats.pearsonr(y_test, u_hat)

        # Calculate proportion of variance explained by main effects vs interactions
        var_u_hat = np.var(u_hat)
        explained_variance_main = np.var(u_hat_main) / var_u_hat if var_u_hat > 0 else 0
        explained_variance_interactions = np.var(u_hat_interaction) / var_u_hat if var_u_hat > 0 else 0

        # Store results for this iteration
        iteration_results.append({
            'iteration': iter_idx + 1,
            'pearson_corr_pheno': pearson_corr_pheno,
            'pearson_corr_beta': pearson_corr_beta,
            'best_lambda': best_lambda,
            'var_explained_main': explained_variance_main,
            'var_explained_interactions': explained_variance_interactions
        })

    # Store all iterations for this trait
    trait_results[trait_index + 1] = iteration_results

# Create a summary dataframe with averages and standard deviations
summary_stats = []
for trait, results in trait_results.items():
    # Convert to DataFrame for easy aggregation
    trait_df = pd.DataFrame(results)

    # Calculate mean and std for each metric
    summary_stats.append({
        'trait': trait,
        'pearson_corr_pheno_mean': trait_df['pearson_corr_pheno'].mean(),
        'pearson_corr_pheno_std': trait_df['pearson_corr_pheno'].std(),
        'pearson_corr_beta_mean': trait_df['pearson_corr_beta'].mean(),
        'pearson_corr_beta_std': trait_df['pearson_corr_beta'].std(),
        'best_lambda_mean': trait_df['best_lambda'].mean(),
    })

summary_df = pd.DataFrame(summary_stats)

# Save only the summary stats
summary_df.to_csv(snakemake.output['correlation_summary'], index=False)

# Print summary statistics
print("\nSummary Statistics (averaged across iterations):")
print(summary_df)