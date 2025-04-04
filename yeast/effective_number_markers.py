#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def load_genotype_data(file_path):
    """Load genotype data and convert R/B to 0/1"""
    try:
        df = pd.read_csv(file_path, sep='\t')
    except:
        # Try with whitespace separation if tab fails
        df = pd.read_csv(file_path, delim_whitespace=True)

    print(f"Loaded data with {df.shape[0]} markers and {df.shape[1]-1} samples")

    # Extract marker names
    marker_names = df['marker'].values

    # Create a copy without the marker column
    genotype_matrix = df.drop('marker', axis=1)

    # Convert R to 0, B to 1
    genotype_matrix = genotype_matrix.applymap(lambda x: 0 if x == 'R' else 1 if x == 'B' else np.nan)

    # Convert to numpy array
    return marker_names, genotype_matrix.values

def calculate_ld_matrix_cuda(genotype_matrix, batch_size=1000):
    """
    Calculate LD (r²) matrix using CUDA acceleration with batched processing

    Args:
        genotype_matrix: numpy array with shape (n_markers, n_samples)
        batch_size: number of markers to process in each batch

    Returns:
        ld_matrix: numpy array with shape (n_markers, n_markers)
    """
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get dimensions
    n_markers, n_samples = genotype_matrix.shape

    # Handle NaN values by filling with column mean
    for i in range(n_markers):
        col = genotype_matrix[i, :]
        mask = ~np.isnan(col)
        if mask.sum() > 0:  # If there are non-NaN values
            mean_val = np.mean(col[mask])
            genotype_matrix[i, ~mask] = mean_val

    # Center the data (convert 0/1 to -0.5/0.5)
    centered_matrix = genotype_matrix - 0.5

    # Initialize LD matrix on CPU (will store results here)
    ld_matrix = np.zeros((n_markers, n_markers))

    print(f"Calculating LD matrix for {n_markers} markers in batches of {batch_size}...")
    start_time = time.time()

    # Process in batches to manage memory
    for i in range(0, n_markers, batch_size):
        batch_end = min(i + batch_size, n_markers)
        batch_markers = centered_matrix[i:batch_end]

        # Convert batch to tensor and move to device
        batch_tensor = torch.tensor(batch_markers, dtype=torch.float32).to(device)

        # Process against all other markers in batches
        for j in range(0, n_markers, batch_size):
            j_end = min(j + batch_size, n_markers)
            j_markers = centered_matrix[j:j_end]

            # Convert to tensor and move to device
            j_tensor = torch.tensor(j_markers, dtype=torch.float32).to(device)

            # Calculate correlation matrix between batch_tensor and j_tensor
            # For each pair of markers, we compute their correlation

            # Normalize each marker (subtract mean and divide by std)
            # Note: Since we've already centered the data, mean ≈ 0
            batch_std = torch.std(batch_tensor, dim=1, keepdim=True)
            j_std = torch.std(j_tensor, dim=1, keepdim=True)

            # Replace zero std with 1 to avoid division by zero
            batch_std[batch_std == 0] = 1
            j_std[j_std == 0] = 1

            batch_norm = batch_tensor / batch_std
            j_norm = j_tensor / j_std

            # Calculate correlation using matrix multiplication
            # corr(x,y) = (x·y)/(|x|·|y|)
            corr_matrix = torch.matmul(batch_norm, j_norm.t()) / n_samples

            # Square to get r² values
            r_squared = corr_matrix ** 2

            # Move back to CPU and store in the LD matrix
            r_squared_cpu = r_squared.cpu().numpy()
            ld_matrix[i:batch_end, j:j_end] = r_squared_cpu

        # Report progress
        elapsed = time.time() - start_time
        progress = (i + batch_size) / n_markers
        progress = min(progress, 1.0)  # Cap at 100%
        remaining = elapsed / progress - elapsed if progress > 0 else 0
        print(f"Progress: {progress*100:.1f}% - Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s")

    print(f"LD matrix calculation completed in {time.time() - start_time:.2f} seconds")

    return ld_matrix

def calculate_eigenvalues_cuda(ld_matrix, batch_size=5000):
    """Calculate eigenvalues of the LD matrix using CUDA"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_markers = ld_matrix.shape[0]

    # If the matrix is small enough, process it directly
    if n_markers <= batch_size:
        # Convert to tensor and move to device
        ld_tensor = torch.tensor(ld_matrix, dtype=torch.float32).to(device)

        # Calculate eigenvalues
        print("Calculating eigenvalues...")
        start_time = time.time()
        eigenvalues = torch.linalg.eigvalsh(ld_tensor)
        print(f"Eigenvalue calculation completed in {time.time() - start_time:.2f} seconds")

        # Move back to CPU
        return eigenvalues.cpu().numpy()

    # For larger matrices, we need to use an approximation or batched approach
    # In this case, let's use a random projection method (similar to randomized SVD)
    # This gives an approximation of the largest eigenvalues

    print(f"Matrix too large for direct eigendecomposition ({n_markers}x{n_markers})")
    print("Using randomized method to approximate largest eigenvalues...")

    # Parameters for randomized method
    n_components = min(n_markers // 10, batch_size)  # Number of components to extract
    n_oversamples = min(n_components // 10 + 10, 100)  # Oversampling parameter
    n_iter = 5  # Number of power iterations

    # Convert to tensor and move to device
    ld_tensor = torch.tensor(ld_matrix, dtype=torch.float32).to(device)

    # Generate random matrix for projection
    Q = torch.randn(n_markers, n_components + n_oversamples, device=device)
    Q, _ = torch.linalg.qr(Q)  # Orthogonalize

    # Power iteration to find dominant eigenspace
    for _ in range(n_iter):
        Q = torch.matmul(ld_tensor, Q)
        Q, _ = torch.linalg.qr(Q)

    # Project LD matrix to this subspace
    B = torch.matmul(Q.T, torch.matmul(ld_tensor, Q))

    # Calculate eigenvalues of the smaller matrix
    eigenvalues_approx = torch.linalg.eigvalsh(B)

    # We've approximated the largest eigenvalues
    # The remaining eigenvalues are approximately 1 (for an LD matrix)
    # So we'll create a synthetic eigenvalue array
    all_eigenvalues = torch.ones(n_markers, device=device)

    # Place the approximated eigenvalues at the end (largest values)
    all_eigenvalues[-len(eigenvalues_approx):] = eigenvalues_approx

    # Sort the eigenvalues
    all_eigenvalues, _ = torch.sort(all_eigenvalues)

    return all_eigenvalues.cpu().numpy()

def compute_effective_markers(eigenvalues):
    """Compute effective number of markers using eigenvalue decomposition"""
    # Gao et al. 2008 method
    # M_eff = M - sum(λ_i - 1) where λ_i are eigenvalues > 1
    n_markers = len(eigenvalues)
    m_eff = n_markers - np.sum(eigenvalues[eigenvalues > 1] - 1)

    # Ensure M_eff is not negative or zero (could happen with numerical issues)
    m_eff = max(1, m_eff)

    return m_eff

def compute_effective_markers_li_ji(eigenvalues):
    """
    Compute effective number of markers using Li and Ji's (2005) method

    Args:
        eigenvalues: numpy array of eigenvalues from the LD matrix

    Returns:
        m_eff: effective number of markers
    """
    # Apply Li and Ji's function to each eigenvalue and sum
    m_eff = 0
    for lam in eigenvalues:
        if abs(lam) >= 1:
            # I(x>=1) + (x - floor(x))
            m_eff += 1 + (abs(lam) - np.floor(abs(lam)))
        else:
            # Just x for 0 <= x < 1
            m_eff += abs(lam)

    return m_eff

def run_eigenvalue_analysis(file_path, batch_size=1000):
    """Run the complete eigenvalue-based analysis"""
    # Load data
    marker_names, genotype_matrix = load_genotype_data(file_path)

    # Calculate LD matrix
    ld_matrix = calculate_ld_matrix_cuda(genotype_matrix, batch_size)

    # Calculate eigenvalues
    eigenvalues = calculate_eigenvalues_cuda(ld_matrix, batch_size)

    # Compute effective number of markers
    #m_eff = compute_effective_markers(eigenvalues)
    m_eff = compute_effective_markers_li_ji(eigenvalues)



    # Calculate marker redundancy percentage
    redundancy_percent = (1 - m_eff / len(marker_names)) * 100

    # Print results
    print("\nResults:")
    print(f"Census marker number: {len(marker_names)}")
    print(f"Effective marker number (eigenvalue method): {m_eff:.2f}")
    print(f"Marker redundancy: {redundancy_percent:.2f}%")

    # Visualize LD matrix (subsample if very large)
    max_plot_size = 5000
    if ld_matrix.shape[0] > max_plot_size:
        print(f"LD matrix too large to plot entirely, sampling {max_plot_size} markers...")
        indices = np.linspace(0, ld_matrix.shape[0]-1, max_plot_size).astype(int)
        ld_sample = ld_matrix[indices][:, indices]
    else:
        ld_sample = ld_matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(ld_sample, cmap="viridis", vmin=0, vmax=1)
    plt.title("Linkage Disequilibrium (r²) Matrix")
    plt.tight_layout()
    plt.savefig('ld_heatmap.png', dpi=300)

    # Visualize eigenvalue distribution
    plt.figure(figsize=(10, 6))

    # If there are too many eigenvalues, bin them
    if len(eigenvalues) > 1000:
        plt.hist(eigenvalues, bins=100, alpha=0.7)
    else:
        plt.hist(eigenvalues, bins=30, alpha=0.7)

    plt.axvline(1, color='red', linestyle='--', label='λ = 1 threshold')
    plt.xlabel('Eigenvalues')
    plt.ylabel('Frequency')
    plt.title('Eigenvalue Distribution of LD Matrix')
    plt.legend()
    plt.tight_layout()
    plt.savefig('eigenvalue_distribution.png', dpi=300)

    # Return results
    return {
        'census_markers': len(marker_names),
        'm_eff': m_eff,
        'redundancy_percent': redundancy_percent,
        'ld_matrix': ld_matrix,
        'eigenvalues': eigenvalues
    }

if __name__ == "__main__":
    # Set the file path to your genotype data
    file_path = "BYxRM_GenoData.txt"

    # Run analysis with batch processing to manage memory usage
    # Adjust batch_size based on your GPU memory
    results = run_eigenvalue_analysis(file_path, batch_size=12000)