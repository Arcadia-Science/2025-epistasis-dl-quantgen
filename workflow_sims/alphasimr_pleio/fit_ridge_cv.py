#!/usr/bin/env python3

from snakemake.script import snakemake
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from scipy import stats

# Number of train/test split iterations
n_iterations = 10  # You can adjust this number based on your needs

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
        n_samples = Z.shape[0]
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

        # Perform Cross-Validation to select best lambda
        ridge_cv = RidgeCV(alphas=lambda_grid, store_cv_results=True)
        ridge_cv.fit(Z_train_centered, y_train)
        best_lambda = ridge_cv.alpha_

        # Predict breeding values (GEBVs)
        beta_hat = ridge_cv.coef_
        u_hat = Z_test_centered @ beta_hat

        # Extract true additive QTL effects for relevant model
        eff_fit = eff[eff['trait'] == trait_index+1]
        pearson_corr_beta, _ = stats.pearsonr(eff_fit['add_eff'], beta_hat)

        # Compute Pearson correlation between predicted and true phenotypes
        pearson_corr_pheno, _ = stats.pearsonr(y_test, u_hat)

        # Store results for this iteration
        iteration_results.append({
            'iteration': iter_idx + 1,
            'pearson_corr_pheno': pearson_corr_pheno,
            'pearson_corr_beta': pearson_corr_beta,
            'best_lambda': best_lambda
        })

    # Store all iterations for this trait
    trait_results[trait_index + 1] = iteration_results

# Create a dataframe with iteration-level results
all_iterations_df = pd.DataFrame([
    {
        'trait': trait,
        'iteration': result['iteration'],
        'pearson_corr_pheno': result['pearson_corr_pheno'],
        'pearson_corr_beta': result['pearson_corr_beta'],
        'best_lambda': result['best_lambda']
    }
    for trait, results in trait_results.items()
    for result in results
])

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
        'best_lambda_std': trait_df['best_lambda'].std(),
    })

summary_df = pd.DataFrame(summary_stats)

# Save results to CSV files
summary_df.to_csv(snakemake.output['correlation_summary'], index=False)

# Print summary statistics
print("\nSummary Statistics (averaged across iterations):")
print(summary_df)
