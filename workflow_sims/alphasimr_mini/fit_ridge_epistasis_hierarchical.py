#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy import stats
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Number of train/test split iterations
n_iterations = 10  # You can adjust this number based on your needs

# Set random seed for reproducibility
np.random.seed(42)

base_file = 'test_sim_qhaplo_2k_100sites_100qtl_Ve0'

print("Processing input files")
# Load genotype, phenotype, and effect files
geno = pd.read_csv(f'{base_file}_g.txt', index_col=0, sep=" ")
pheno = pd.read_csv(f'{base_file}_p.txt', index_col=0, sep=" ")
eff = pd.read_csv(f'{base_file}_eff.txt', sep=" ")

print("Running hierarchical ridge regression fits with multiple train/test splits")

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
lambda_grid_main = np.logspace(-5, 5, 20)  # Main effects regularization
lambda_grid_int = np.logspace(-2, 8, 20)   # Interaction effects regularization - stronger range

# Custom hierarchical ridge regression class
class HierarchicalRidge:
    def __init__(self, alpha_main=1.0, alpha_interaction=10.0):
        """
        Ridge regression with separate regularization for main effects and interactions.

        Parameters:
        -----------
        alpha_main : float
            Regularization strength for main effects
        alpha_interaction : float
            Regularization strength for interaction effects
        """
        self.alpha_main = alpha_main
        self.alpha_interaction = alpha_interaction
        self.coef_main_ = None
        self.coef_interaction_ = None
        self.intercept_ = 0.0
        self.coef_ = None  # Full coefficient vector

    def fit(self, X_main, X_interaction, y):
        """
        Fit the hierarchical ridge model.

        Parameters:
        -----------
        X_main : array, shape (n_samples, n_main_features)
            Main effects feature matrix
        X_interaction : array, shape (n_samples, n_interaction_features)
            Interaction effects feature matrix
        y : array, shape (n_samples,)
            Target values

        Returns:
        --------
        self : object
            Returns self
        """
        n_samples_main, n_features_main = X_main.shape
        n_samples_int, n_features_int = X_interaction.shape

        if n_samples_main != n_samples_int:
            raise ValueError("X_main and X_interaction must have the same number of samples")

        # Center y
        y_mean = np.mean(y)
        y_centered = y - y_mean

        # Create the penalty matrix (diagonal with different alphas)
        # For main effects: alpha_main, for interactions: alpha_interaction
        penalty = np.zeros(n_features_main + n_features_int)
        penalty[:n_features_main] = self.alpha_main
        penalty[n_features_main:] = self.alpha_interaction

        # Combine features
        X_combined = np.hstack([X_main, X_interaction])

        # Solve the penalized least squares problem
        # (X^T X + P)^-1 X^T y where P is the penalty matrix
        XTX = X_combined.T @ X_combined
        penalty_matrix = np.diag(penalty)
        XTy = X_combined.T @ y_centered

        # Solve the system
        beta = np.linalg.solve(XTX + penalty_matrix, XTy)

        # Extract coefficients
        self.coef_main_ = beta[:n_features_main]
        self.coef_interaction_ = beta[n_features_main:]
        self.coef_ = beta
        self.intercept_ = y_mean

        return self

    def predict(self, X_main, X_interaction):
        """
        Predict using the hierarchical ridge model.

        Parameters:
        -----------
        X_main : array, shape (n_samples, n_main_features)
            Main effects feature matrix
        X_interaction : array, shape (n_samples, n_interaction_features)
            Interaction effects feature matrix

        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        y_pred = self.intercept_
        y_pred += X_main @ self.coef_main_
        y_pred += X_interaction @ self.coef_interaction_
        return y_pred

# Custom cross-validation for hierarchical ridge regression
def hierarchical_ridge_cv(X_main, X_interaction, y, alphas_main, alphas_interaction,
                          cv=5, scoring='neg_mean_squared_error'):
    """
    Cross-validation for hierarchical ridge regression.

    Parameters:
    -----------
    X_main : array, shape (n_samples, n_main_features)
        Main effects feature matrix
    X_interaction : array, shape (n_samples, n_interaction_features)
        Interaction effects feature matrix
    y : array, shape (n_samples,)
        Target values
    alphas_main : array, shape (n_alphas_main,)
        Regularization strengths for main effects
    alphas_interaction : array, shape (n_alphas_interaction,)
        Regularization strengths for interaction effects
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring method

    Returns:
    --------
    best_alpha_main : float
        Best regularization strength for main effects
    best_alpha_interaction : float
        Best regularization strength for interaction effects
    """
    n_samples = X_main.shape[0]
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    best_score = -np.inf
    best_alpha_main = None
    best_alpha_interaction = None

    for alpha_main in alphas_main:
        for alpha_interaction in alphas_interaction:
            scores = []

            for train_idx, val_idx in kf.split(X_main):
                # Split data
                X_main_train, X_main_val = X_main[train_idx], X_main[val_idx]
                X_int_train, X_int_val = X_interaction[train_idx], X_interaction[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Fit model
                model = HierarchicalRidge(alpha_main=alpha_main, alpha_interaction=alpha_interaction)
                model.fit(X_main_train, X_int_train, y_train)

                # Predict and score
                y_pred = model.predict(X_main_val, X_int_val)

                # Calculate score based on scoring method
                if scoring == 'neg_mean_squared_error':
                    score = -np.mean((y_val - y_pred) ** 2)
                elif scoring == 'r2':
                    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                    ss_res = np.sum((y_val - y_pred) ** 2)
                    score = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                scores.append(score)

            # Average score across folds
            mean_score = np.mean(scores)

            # Update best if improved
            if mean_score > best_score:
                best_score = mean_score
                best_alpha_main = alpha_main
                best_alpha_interaction = alpha_interaction

    return best_alpha_main, best_alpha_interaction

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

        # Create all pairwise interactions
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

        print(f"  Main effects matrix shape: {Z_train_scaled.shape}")
        print(f"  Interaction matrix shape: {Z_train_interactions.shape}")

        # Find optimal lambda values using cross-validation
        print("  Finding optimal lambda values...")

        # For computational efficiency with large interaction matrices,
        # optionally use a subset of samples for cross-validation
        cv_samples = min(5000, Z_train_scaled.shape[0])
        indices = np.random.choice(Z_train_scaled.shape[0], cv_samples, replace=False)

        best_alpha_main, best_alpha_int = hierarchical_ridge_cv(
            Z_train_scaled[indices],
            Z_train_interactions[indices],
            y_train[indices],
            lambda_grid_main,
            lambda_grid_int,
            cv=5,
            scoring='r2'
        )

        print(f"  Best lambda for main effects: {best_alpha_main}")
        print(f"  Best lambda for interactions: {best_alpha_int}")

        # Fit the hierarchical ridge model with the optimal lambda values
        print("  Fitting hierarchical ridge model...")
        hier_ridge = HierarchicalRidge(alpha_main=best_alpha_main,
                                      alpha_interaction=best_alpha_int)
        hier_ridge.fit(Z_train_scaled, Z_train_interactions, y_train)

        # Get coefficients
        beta_main = hier_ridge.coef_main_
        beta_interaction = hier_ridge.coef_interaction_

        # Predict on test set
        u_hat_main = Z_test_scaled @ beta_main
        u_hat_interaction = Z_test_interactions @ beta_interaction
        u_hat = u_hat_main + u_hat_interaction

        # Extract true additive QTL effects for relevant model
        eff_fit = eff[eff['trait'] == trait_index+1]

        # Compute correlations
        pearson_corr_beta, _ = stats.pearsonr(eff_fit['add_eff'], beta_main)
        pearson_corr_pheno, _ = stats.pearsonr(y_test, u_hat)

        # For comparison, also fit standard ridge regression
        ridge_standard = RidgeCV(alphas=lambda_grid_main)
        ridge_standard.fit(Z_train_scaled, y_train)
        u_hat_standard = ridge_standard.predict(Z_test_scaled)
        pearson_corr_pheno_std_ridge, _ = stats.pearsonr(y_test, u_hat_standard)

        # Calculate variance explained by main effects and interactions
        var_u_hat = np.var(u_hat)
        var_explained_main = np.var(u_hat_main) / var_u_hat if var_u_hat > 0 else 0
        var_explained_interactions = np.var(u_hat_interaction) / var_u_hat if var_u_hat > 0 else 0

        # Store results for this iteration
        iteration_results.append({
            'iteration': iter_idx + 1,
            'pearson_corr_pheno': pearson_corr_pheno,
            'pearson_corr_pheno_std_ridge': pearson_corr_pheno_std_ridge,
            'pearson_corr_beta': pearson_corr_beta,
            'best_lambda_main': best_alpha_main,
            'best_lambda_int': best_alpha_int,
            'var_explained_main': var_explained_main,
            'var_explained_interactions': var_explained_interactions
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
        'pearson_corr_pheno_std_ridge_mean': trait_df['pearson_corr_pheno_std_ridge'].mean(),
        'pearson_corr_pheno_std_ridge_std': trait_df['pearson_corr_pheno_std_ridge_std'].std() if 'pearson_corr_pheno_std_ridge_std' in trait_df else 0,
        'pearson_corr_beta_mean': trait_df['pearson_corr_beta'].mean(),
        'pearson_corr_beta_std': trait_df['pearson_corr_beta'].std(),
        'best_lambda_main_mean': trait_df['best_lambda_main'].mean(),
        'best_lambda_int_mean': trait_df['best_lambda_int'].mean(),
        'var_explained_main_mean': trait_df['var_explained_main'].mean(),
        'var_explained_interactions_mean': trait_df['var_explained_interactions'].mean(),
    })

summary_df = pd.DataFrame(summary_stats)

# Save only the summary stats
summary_df.to_csv(f'{base_file}_hierarchical_ridge_summary.csv', index=False)

# Print summary statistics
print("\nSummary Statistics (averaged across iterations):")
print(summary_df)