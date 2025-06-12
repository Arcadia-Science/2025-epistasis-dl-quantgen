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


#simple ridge regression equivalent meant to be trained with KL divergence to enforce gaussian prior
class gplinear_kl(nn.Module):
    """
    simple ridge regression equivalent meant to be trained with KL divergence to enforce gaussian prior on weights

    Args:
        n_loci: #sites * #alleles
        n_pheno: number of phenotypes to output/predict
    """
    def __init__(self, n_loci, n_phen):
        super(gplinear_kl, self).__init__()
        self.linear = nn.Linear(n_loci, n_phen)

    def forward(self, x):
        return self.linear(x)
