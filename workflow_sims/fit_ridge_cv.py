#!/usr/bin/env python3

from snakemake.script import snakemake
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from scipy import stats

print("Processing input files")
# Load genotype, phenotype, and effect files
geno = pd.read_csv(snakemake.input['input_geno'], index_col=0, sep=" ")
pheno = pd.read_csv(snakemake.input['input_pheno'], index_col=0, sep=" ")
eff = pd.read_csv(snakemake.input['loci_effects'], sep=" ")

print("Running rr fits")

# Convert genotype matrix (n_samples x n_markers)
Z = geno.T.values

# Train/Test Split (First 90% for training)
n_samples = Z.shape[0]
split_index = int(n_samples * 0.9)  # 90% train, 10% test

Z_train, Z_test = Z[:split_index, :], Z[split_index:, :]

# Compute mean allele frequency for each SNP and mean-center genotypes
p = np.mean(Z_train, axis=0)
Z_train_centered = Z_train - p
Z_test_centered = Z_test - p

# Define range of lambda values for cross-validation
lambda_grid = np.logspace(-3, 5, 50)  # Search over log scale

# Initialize lists to store results
results = []
summary_stats = []

# Loop through each phenotype column
for trait_index in range(pheno.shape[1]):  # Loop over all phenotype columns
    y = pheno.iloc[:, trait_index].values  # Extract phenotype column
    y_train, y_test = y[:split_index], y[split_index:]  # Train/test split

    print("start analysis trait")
    print(trait_index)

    # Perform Cross-Validation to select best lambda
    ridge_cv = RidgeCV(alphas=lambda_grid, store_cv_values=True)
    ridge_cv.fit(Z_train_centered, y_train)
    best_lambda = ridge_cv.alpha_  # Optimal lambda from cross-validation

    print("Best lambda")
    print(best_lambda)

    # Predict breeding values (GEBVs)
    beta_hat = ridge_cv.coef_
    u_hat = Z_test_centered @ beta_hat

    #extract true additive QTL effects for relevant model and calculate correlation with model coefficients
    eff_fit = eff[eff['trait'] == trait_index+1]
    pearson_corr_beta, _ = stats.pearsonr(eff_fit['add_eff'],ridge_cv.coef_)

    # Compute Pearson correlation between predicted and true phenotypes
    pearson_corr_pheno, _ = stats.pearsonr(y_test, u_hat)

    # Store true vs predicted phenotypes in long format
    results.append(pd.DataFrame({
        'trait': trait_index + 1,  # 1-based trait indexing
        'true_pheno': y_test,
        'pred_pheno': u_hat
    }))

    # Store Pearson correlation results and selected lambda
    summary_stats.append({
        'trait': trait_index + 1,
        'pearson_corr_pheno': pearson_corr_pheno,
        'pearson_corr_beta' : pearson_corr_beta,
        'best_lambda': best_lambda
    })

# Combine all trait results into a single long-format dataframe
results_df = pd.concat(results, ignore_index=True)
summary_df = pd.DataFrame(summary_stats)  # Convert correlation results to DataFrame

# Save results to CSV files
results_df.to_csv(snakemake.output['pred_pheno_output'], index=False)
summary_df.to_csv(snakemake.output['correlation_summary'], index=False)

# Print first few rows of results and summary stats
print("Prediction Results:")
print(results_df.head())

print("\nSummary Statistics:")
print(summary_df.head())
