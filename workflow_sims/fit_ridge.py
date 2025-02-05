#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import stats

# Load genotype, phenotype, and effect files
geno = pd.read_csv('alphasimr_output/test_sim_WF_1kbt_10000n_5000000bp_g.txt', index_col=0, sep=" ")
pheno = pd.read_csv('alphasimr_output/test_sim_WF_1kbt_10000n_5000000bp_p.txt', index_col=0, sep=" ")
eff = pd.read_csv('alphasimr_output/test_sim_WF_1kbt_10000n_5000000bp_eff.txt', index_col=0, sep=" ")

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

# Set ridge regression penalty based on heritability
h2 = 0.4
lambda_rrblup = (1 - h2) / h2

# Initialize lists to store results
results = []
summary_stats = []

# Loop through each phenotype column
for trait_index in range(pheno.shape[1]):  # Loop over all phenotype columns
    y = pheno.iloc[:, trait_index].values  # Extract phenotype column
    y_train, y_test = y[:split_index], y[split_index:]  # Train/test split

    # Fit Ridge Regression Model
    ridge_model = Ridge(alpha=lambda_rrblup)
    ridge_model.fit(Z_train_centered, y_train)

    # Predict breeding values (GEBVs)
    beta_hat = ridge_model.coef_
    u_hat = Z_test_centered @ beta_hat

    # Compute Pearson correlation between predicted and true phenotypes
    pearson_corr, _ = stats.pearsonr(y_test, u_hat)

    # Store true vs predicted phenotypes in long format
    results.append(pd.DataFrame({
        'trait': trait_index + 1,  # 1-based trait indexing
        'true_pheno': y_test,
        'pred_pheno': u_hat
    }))

    # Store Pearson correlation results
    summary_stats.append({
        'trait': trait_index + 1,
        'pearson_corr': pearson_corr
    })

# Combine all trait results into a single long-format dataframe
results_df = pd.concat(results, ignore_index=True)
summary_df = pd.DataFrame(summary_stats)  # Convert correlation results to DataFrame

# Save results to CSV files
results_df.to_csv("rrBLUP_output/ridge_predictions_long_format.csv", index=False)
summary_df.to_csv("rrBLUP_output/ridge_summary_stats.csv", index=False)

# Print first few rows of results and summary stats
print("Prediction Results:")
print(results_df.head())

print("\nSummary Statistics:")
print(summary_df.head())

# Plot example trait (e.g., first trait)



plt.scatter(results_df[results_df['trait'] == 1]['true_pheno'],
            results_df[results_df['trait'] == 1]['pred_pheno'])
plt.xlabel("True Phenotype")
plt.ylabel("Predicted Phenotype")
plt.title("Predicted vs True Phenotypes (Trait 1)")
plt.show()
