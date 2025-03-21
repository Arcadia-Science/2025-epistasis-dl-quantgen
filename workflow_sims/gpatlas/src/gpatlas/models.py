"""
Neural network models for genetic data analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#ld informed genetic autoencoder
class LDGroupedAutoencoder(nn.Module):
    """
    LD-aware autoencoder for genetic data using grouped convolution.
    Each window of loci is processed independently.

    Args:
        input_length: Total length of input tensor (one-hot encoded, so 2 values per locus)
        loci_count: Actual number of genetic loci (half of input_length)
        window_size: Number of loci to group together in local connections
        window_stride: Stride length for the convolution operation
        n_out_channels: Number of output channels per LD window
        latent_dim: Dimension of the latent space
    """
    def __init__(self, input_length, loci_count, window_size, window_stride, n_out_channels, latent_dim):
        super().__init__()

        self.input_length = input_length  # Total values for loci
        self.loci_count = loci_count      # Number of loci
        self.window_size = window_size    # Loci per group
        self.latent_dim = latent_dim      # Latent space dimension
        self.n_out_channels = n_out_channels  # Output channels per LD window

        # Calculate the number of groups
        self.n_groups = loci_count // window_size

        # Encoder layers
        self.encoder_conv = nn.Conv1d(
            in_channels=self.n_groups,           # One channel per window
            out_channels=self.n_groups*n_out_channels,  # Output channels per window
            kernel_size=window_size * 2,         # Cover entire window (2 alleles per locus)
            stride=window_stride,                # Stride length
            groups=self.n_groups,                # Each window processed independently
            bias=True
        )

        self.encoder_act = nn.LeakyReLU(0.1)

        # Fully connected layer to latent space
        self.encoder_fc = nn.Linear(self.n_groups*n_out_channels, latent_dim)
        self.encoder_fc_act = nn.LeakyReLU(0.1)

        # Decoder - mirror of encoder
        self.decoder_fc = nn.Linear(latent_dim, self.n_groups*n_out_channels)
        self.decoder_fc_act = nn.LeakyReLU(0.1)

        # Expand each window back to original size
        self.decoder_conv = nn.ConvTranspose1d(
            in_channels=self.n_out_channels * self.n_groups,
            out_channels=self.n_groups,
            kernel_size=window_size * 2,
            stride=window_stride,
            groups=self.n_groups,
            bias=True
        )

        self.final_act = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the autoencoder

        Args:
            x: Input tensor of shape [batch_size, input_length]

        Returns:
            Reconstructed tensor of shape [batch_size, input_length]
        """
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.n_groups, self.window_size * 2)

        x = self.encoder_conv(x)
        x = self.encoder_act(x)

        x = x.reshape(batch_size, self.n_groups * self.n_out_channels)
        latent = self.encoder_fc(x)
        latent = self.encoder_fc_act(latent)

        # Decode from latent space
        x = self.decoder_fc(latent)
        x = self.decoder_fc_act(x)

        x = x.reshape(batch_size, self.n_groups * self.n_out_channels, 1)

        x = self.decoder_conv(x)
        x = x.reshape(batch_size, self.input_length)

        # Apply sigmoid
        x = self.final_act(x)

        return x

    def encode(self, x):
        """
        Encode data to latent space

        Args:
            x: Input tensor of shape [batch_size, input_length]

        Returns:
            Latent representation of shape [batch_size, latent_dim]
        """
        batch_size = x.size(0)

        # Reshape to group by windows
        x = x.reshape(batch_size, self.n_groups, self.window_size * 2)

        # Apply convolution
        x = self.encoder_conv(x)
        x = self.encoder_act(x)

        # Flatten and map to latent space
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels)
        x = self.encoder_fc(x)
        return self.encoder_fc_act(x)

    def decode(self, z):
        """
        Decode from latent space

        Args:
            z: Latent representation of shape [batch_size, latent_dim]

        Returns:
            Reconstructed tensor of shape [batch_size, input_length]
        """
        batch_size = z.size(0)

        # Map from latent space to window representations
        x = self.decoder_fc(z)
        x =  self.decoder_fc_act(x)

        # Reshape for transposed convolution
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels, 1)

        # Expand each window back to original size
        x = self.decoder_conv(x)

        # Reshape to original format
        x = x.reshape(batch_size, self.input_length)

        # Apply sigmoid
        x = self.final_act(x)

        return x

#LD encoder only for saving

class LDEncoder(nn.Module):
    def __init__(self, autoencoder):
        """
        Extract the encoder part of the trained LDGroupedAutoencoder

        Args:
            autoencoder: Trained LDGroupedAutoencoder model
        """
        super().__init__()

        # Copy all encoder-related parameters from the trained autoencoder
        self.input_length = autoencoder.input_length
        self.loci_count = autoencoder.loci_count
        self.window_size = autoencoder.window_size
        self.latent_dim = autoencoder.latent_dim
        self.n_groups = autoencoder.n_groups
        self.n_out_channels = autoencoder.n_out_channels
        self.n_alleles = autoencoder.n_alleles

        # Copy the encoder layers
        self.allele_conv = autoencoder.allele_conv
        self.allele_conv_act = autoencoder.allele_conv_act
        self.encoder_conv = autoencoder.encoder_conv
        self.encoder_conv_act = autoencoder.encoder_conv_act
        self.encoder_fc = autoencoder.encoder_fc
        self.encoder_fc_act = autoencoder.encoder_fc_act

    def forward(self, x):
        """
        Encode data to latent space

        Args:
            x: Input tensor of shape [batch_size, input_length]

        Returns:
            Latent representation of shape [batch_size, latent_dim]
        """
        batch_size = x.size(0)

        # 1. Reshape to process each locus
        x = x.reshape(batch_size, self.loci_count, self.n_alleles)

        # 2. Apply locus-level convolution
        x = self.allele_conv(x)
        x = self.allele_conv_act(x)
        x = x.squeeze(-1)  # Remove the last dimension

        # 3. Reshape for window-based processing
        x = x.reshape(batch_size, self.n_groups, self.window_size)

        # 4. Apply window-based convolution
        x = self.encoder_conv(x)
        x = self.encoder_conv_act(x)

        # 5. Flatten and map to latent space
        x = x.reshape(batch_size, self.n_groups * self.n_out_channels)
        x = self.encoder_fc(x)
        x = self.encoder_fc_act(x)


#phenotypic encoder/decoder
class Q_net(nn.Module):
    """
    Encoder network for phenotype data.

    Args:
        phen_dim: Dimension of the phenotype data
        N: Hidden layer dimension and latent dimension
    """
    def __init__(self, phen_dim=25, N=1000):
        super().__init__()

        batchnorm_momentum = 0.8
        latent_dim = N
        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class P_net(nn.Module):
    """
    Decoder network for phenotype data.

    Args:
        phen_dim: Dimension of the phenotype data
        N: Hidden layer dimension and latent dimension
    """
    def __init__(self, phen_dim=25, N=1000):
        super().__init__()

        out_phen_dim = phen_dim
        latent_dim = N
        batchnorm_momentum = 0.8

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=out_phen_dim),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

#G->P network
class GQ_to_P_net(nn.Module):
    """
    Network mapping from genotype latent space to phenotype latent space.

    Args:
        N: Output dimension (matching phenotype latent dimension)
        latent_space_g: Input dimension (genotype latent dimension)
        latent_space_gp: Hidden layer dimension
    """
    def __init__(self, N, latent_space_g, latent_space_gp):
        super().__init__()

        batchnorm_momentum = 0.8
        g_latent_dim = latent_space_g
        self.encoder = nn.Sequential(
            nn.Linear(in_features=g_latent_dim, out_features=latent_space_gp),
            nn.BatchNorm1d(latent_space_gp, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=latent_space_gp, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

#vanilla genetic autoencoder
# gencoder
class GQ_net(nn.Module):
    def __init__(self, n_geno=None, n_alleles=2, latent_space_g=None):
        super().__init__()
        n_loci = n_geno * n_alleles

        batchnorm_momentum = 0.8
        g_latent_dim = latent_space_g
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=g_latent_dim),
            nn.BatchNorm1d(g_latent_dim, momentum=0.8),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=g_latent_dim, out_features=g_latent_dim),
            nn.BatchNorm1d(g_latent_dim, momentum=0.8),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# gendecoder
class GP_net(nn.Module):
    def __init__(self, n_geno=None, n_alleles=2, latent_space_g=None):
        super().__init__()
        n_loci = n_geno * n_alleles

        batchnorm_momentum = 0.8
        g_latent_dim = latent_space_g
        self.encoder = nn.Sequential(
            nn.Linear(in_features=g_latent_dim, out_features=g_latent_dim),
            nn.BatchNorm1d(g_latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=g_latent_dim, out_features=n_loci),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


#null model fully connected G -> P network
class GP_net(nn.Module):
    """
    Simple fully connected G -> P network

    Args:
        n_loci: #sites * #alleles
        latent_space_g: geno hidden layer size
        n_pheno: number of phenotypes to output/predict
    """
    def __init__(self, n_loci, latent_space_g, n_pheno):
        super().__init__()
        batchnorm_momentum = 0.8
        self.gpnet = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=latent_space_g),
            nn.BatchNorm1d(latent_space_g, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(in_features=latent_space_g, out_features=latent_space_g),
            nn.BatchNorm1d(latent_space_g, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(in_features=latent_space_g, out_features=n_pheno),
            nn.BatchNorm1d(n_pheno, momentum=batchnorm_momentum)
        )

    def forward(self, x):
        x = self.gpnet(x)
        return x