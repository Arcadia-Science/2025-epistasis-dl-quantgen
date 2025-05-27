"""
Neural network models for genetic data analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#null model fully connected G -> P network
class GP_net(nn.Module):
    """
    Simple fully connected G -> P network

    Args:
        n_loci: #sites * #alleles
        latent_space_g: geno hidden layer size
        n_pheno: number of phenotypes to output/predict
    """
    def __init__(self, n_loci, latent_space_g, n_pheno, latent_space_g1=None):
        super().__init__()
        batchnorm_momentum = 0.8

        if latent_space_g1 == None:
            latent_space_g1 = latent_space_g

        self.gpnet = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=latent_space_g1),
            nn.BatchNorm1d(latent_space_g1, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(in_features=latent_space_g1, out_features=latent_space_g),
            nn.BatchNorm1d(latent_space_g, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(in_features=latent_space_g, out_features=n_pheno),
            nn.BatchNorm1d(n_pheno, momentum=batchnorm_momentum)
        )

    def forward(self, x):
        x = self.gpnet(x)
        return x

#null model fully connected G -> P network with weight pruning
class GP_net_btl(nn.Module):
    """
    Simple fully connected G -> P network with gradual pruning for the first layer

    Args:
        n_loci: #sites * #alleles
        latent_space_g1: first bottleneck layer size
        latent_space_g: geno hidden layer size
        n_pheno: number of phenotypes to output/predict
    """
    def __init__(self, n_loci, latent_space_g1, latent_space_g, n_pheno):
        super().__init__()
        batchnorm_momentum = 0.8

        if latent_space_g1 == None:
            latent_space_g1 = latent_space_g

        # Define layers individually for pruning access
        self.fc1 = nn.Linear(in_features=n_loci, out_features=latent_space_g1)
        self.bn1 = nn.BatchNorm1d(latent_space_g1, momentum=batchnorm_momentum)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.fc2 = nn.Linear(in_features=latent_space_g1, out_features=latent_space_g)
        self.bn2 = nn.BatchNorm1d(latent_space_g, momentum=batchnorm_momentum)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        self.fc3 = nn.Linear(in_features=latent_space_g, out_features=latent_space_g)
        self.bn3 = nn.BatchNorm1d(latent_space_g, momentum=batchnorm_momentum)
        self.act3 = nn.LeakyReLU(0.01, inplace=True)

        self.fc4 = nn.Linear(in_features=latent_space_g, out_features=n_pheno)
        self.bn4 = nn.BatchNorm1d(n_pheno, momentum=batchnorm_momentum)

        # Initialize pruning parameters for the first layer
        self.register_buffer('weight_mask', torch.ones_like(self.fc1.weight))
        self.pruned_so_far = 0

    def forward(self, x):
        # Apply mask to first layer weights
        self.fc1.weight.data = self.fc1.weight.data * self.weight_mask

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x

    def gradual_prune(self, target_sparsity, current_step, total_steps):
        """
        Gradually prunes weights in the first layer according to a schedule
        - target_sparsity: final sparsity level (e.g., 0.8 for 80% pruned)
        - current_step: current training step
        - total_steps: total steps over which to complete pruning
        """
        if current_step > total_steps:
            return self.pruned_so_far

        # Calculate target sparsity for this step (cubic schedule)
        step_ratio = current_step / total_steps
        current_target = target_sparsity * (1.0 - (1.0 - step_ratio)**3)

        # Calculate how many weights to prune this step
        weights_to_prune = int(self.fc1.weight.numel() * (current_target - self.pruned_so_far))

        if weights_to_prune <= 0:
            return self.pruned_so_far

        # Get unpruned weights
        unpruned_weights = self.fc1.weight.data * self.weight_mask

        # Find the threshold for pruning
        abs_weights = unpruned_weights.abs().view(-1)
        non_zero_abs_weights = abs_weights[abs_weights > 0]
        if len(non_zero_abs_weights) <= weights_to_prune:
            return self.pruned_so_far

        threshold = torch.kthvalue(non_zero_abs_weights, weights_to_prune).values

        # Update mask
        new_mask = self.weight_mask.clone()
        new_mask[unpruned_weights.abs() <= threshold] = 0
        self.weight_mask = new_mask

        # Update pruned_so_far
        self.pruned_so_far = 1.0 - (self.weight_mask.sum() / self.weight_mask.numel()).item()

        # Apply mask
        self.fc1.weight.data = self.fc1.weight.data * self.weight_mask

        return self.pruned_so_far

    # Optional: Add method to get sparsity statistics
    def get_sparsity_stats(self):
        total_weights = self.fc1.weight.numel()
        non_zero_weights = self.weight_mask.sum().item()
        sparsity = 1.0 - (non_zero_weights / total_weights)

        # Count weights by QTL (input feature)
        weights_per_qtl = self.weight_mask.sum(dim=0)
        qtl_importance = weights_per_qtl / weights_per_qtl.max()

        return {
            "sparsity": sparsity,
            "active_weights": non_zero_weights,
            "total_weights": total_weights,
            "qtl_importance": qtl_importance
        }

#simple ridge regression equivalent meant to be trained with KL divergence to enforce gaussian prior
class gplinear_kl(nn.Module):
    """
    simple ridge regression equivalent meant to be trained with KL divergence to enforce gaussian prior

    Args:
        n_loci: #sites * #alleles
        n_pheno: number of phenotypes to output/predict
    """
    def __init__(self, n_loci, n_phen):
        super(gplinear_kl, self).__init__()
        self.linear = nn.Linear(n_loci, n_phen)

    def forward(self, x):
        return self.linear(x)
