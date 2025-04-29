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
    def __init__(self, phen_dim=25, N=1000, hidden_dim = None):
        super().__init__()

        batchnorm_momentum = 0.8
        latent_dim = N
        if hidden_dim is None:
            hidden_dim = N
        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=latent_dim),
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
    def __init__(self, phen_dim=25, N=1000, hidden_dim = None):
        super().__init__()

        out_phen_dim = phen_dim
        latent_dim = N
        if hidden_dim is None:
            hidden_dim = N
        batchnorm_momentum = 0.8

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=out_phen_dim),
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

#null model fully connected G -> P network
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

#residual based fully connected G -> P network
class GP_net_combi(nn.Module):
    """
    Combined network using linear model predictions alongside raw genotype data

    Args:
        n_loci: #sites * #alleles
        latent_space_g: geno hidden layer size
        n_pheno: number of phenotypes to output/predict
        linear_model: pre-trained linear model for additive effects
    """
    def __init__(self, n_loci, latent_space_g, n_pheno, linear_model):
        super().__init__()
        batchnorm_momentum = 0.8

        # Store linear model and freeze its parameters
        self.linear_model = linear_model
        for param in self.linear_model.parameters():
            param.requires_grad = False

        # Adjusted input size to include phenotype predictions
        combined_input_size = n_loci + n_pheno

        # Modified network with concatenated input
        self.gpnet = nn.Sequential(
            nn.Linear(in_features=combined_input_size, out_features=latent_space_g),
            nn.BatchNorm1d(latent_space_g, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(in_features=latent_space_g, out_features=latent_space_g),
            nn.BatchNorm1d(latent_space_g, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(in_features=latent_space_g, out_features=n_pheno),
            nn.BatchNorm1d(n_pheno, momentum=batchnorm_momentum)
        )

    def forward(self, x):
        # Get linear model predictions (additive effects)
        with torch.no_grad():
            linear_pred = self.linear_model(x)

        # Concatenate raw genotype data with linear predictions
        combined_input = torch.cat([x, linear_pred], dim=1)

        # Get non-linear component prediction
        nonlinear_pred = self.gpnet(combined_input)

        # Final prediction combines linear and non-linear components
        # This represents: additive effects + non-additive effects (epistasis)
        #return linear_pred + nonlinear_pred
        return nonlinear_pred

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
