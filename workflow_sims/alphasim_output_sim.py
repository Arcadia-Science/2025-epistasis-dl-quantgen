#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
from typing import Tuple, List, Dict

class GeneticSimulation:
    def __init__(
        self,
        genotype_file: str,
        locus_effects_file: str,
        num_offspring: int = 1000,
        heritability: float = 0.8,  # Proportion of variance explained by genetics
        random_seed: int = 42
    ):
        """
        Initialize the genetic simulation.

        Args:
            genotype_file: Path to the genotype file
            locus_effects_file: Path to the locus effects file
            num_offspring: Number of offspring to generate
            heritability: Proportion of phenotypic variance explained by genetics
            random_seed: Seed for random number generation
        """
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.num_offspring = num_offspring
        self.heritability = heritability

        # Load the genotype and locus effects data
        self.genotypes = self._load_genotypes(genotype_file)
        self.locus_effects = self._load_locus_effects(locus_effects_file)

        # Get dimensions
        self.num_markers = self.genotypes.shape[0]
        self.num_individuals = self.genotypes.shape[1] - 1  # Subtract 1 for the marker ID column
        self.num_traits = len(self.locus_effects['trait'].unique())

        # Store marker positions (for recombination)
        self.marker_positions = self._get_marker_positions()

        # Validate data
        self._validate_data()

        print(f"Loaded {self.num_markers} markers for {self.num_individuals} individuals and {self.num_traits} traits")

    def _load_genotypes(self, genotype_file: str) -> pd.DataFrame:
        """Load genotype data from file."""
        # Using r'\s+' with raw string notation to avoid escape sequence warning
        genotypes = pd.read_csv(genotype_file, sep=r'\s+')
        # Make sure the first column is the marker ID
        genotypes.set_index(genotypes.columns[0], inplace=True)
        return genotypes

    def _load_locus_effects(self, locus_effects_file: str) -> pd.DataFrame:
        """Load locus effects data from file."""
        return pd.read_csv(locus_effects_file, sep=r'\s+')

    def _get_marker_positions(self) -> Dict[str, float]:
        """
        Extract marker positions from marker IDs.
        Assumes format like "1_1", "1_2", etc. where first number is chromosome
        and second is position within chromosome.
        """
        positions = {}
        for marker in self.genotypes.index:
            if marker == 'qtl':  # Skip header if present
                continue
            try:
                chrom, pos = marker.split('_')
                if chrom not in positions:
                    positions[chrom] = []
                positions[chrom].append((marker, float(pos)))
            except (ValueError, AttributeError):
                print(f"Warning: Couldn't parse position from marker {marker}")

        # Sort positions within each chromosome
        for chrom in positions:
            positions[chrom] = sorted(positions[chrom], key=lambda x: x[1])

        return positions

    def _validate_data(self):
        """Validate that the locus effects reference valid markers."""
        valid_loci = set(range(1, self.num_markers + 1))
        for _, row in self.locus_effects.iterrows():
            # Convert locus values to integers, handling scientific notation and NaN values
            try:
                locus = int(float(row['locus']))

                # Handle NaN in epi_loc (means no epistatic interaction)
                if pd.isna(row['epi_loc']):
                    epi_loc = 0
                else:
                    epi_loc = int(float(row['epi_loc'])) if row['epi_loc'] != 0 else 0
            except (ValueError, TypeError):
                print(f"Warning: Could not convert locus values to integers: {row['locus']}, {row['epi_loc']}")
                continue

            if locus not in valid_loci:
                print(f"Warning: Locus {locus} in effects file not found in genotype data")

            if epi_loc != 0 and epi_loc not in valid_loci:
                print(f"Warning: Epistatic locus {epi_loc} in effects file not found in genotype data")

    def sample_parents(self, num_pairs: int) -> List[Tuple[int, int]]:
        """
        Sample random pairs of individuals to be parents.

        Args:
            num_pairs: Number of parent pairs to sample

        Returns:
            List of tuples containing indices of parent pairs
        """
        individual_indices = list(range(self.num_individuals))
        parent_pairs = []

        for _ in range(num_pairs):
            # Sample without replacement within each pair
            parents = random.sample(individual_indices, 2)
            parent_pairs.append((parents[0], parents[1]))

        return parent_pairs

    def simulate_recombination(self, parent1_idx: int, parent2_idx: int) -> np.ndarray:
        """
        Simulate recombination between two haploid parents to create a recombinant haploid gamete.

        Args:
            parent1_idx: Index of first parent
            parent2_idx: Index of second parent

        Returns:
            Recombined haploid gamete
        """
        # Convert to 0-based index for the genotype matrix columns
        p1_idx = parent1_idx + 1  # +1 to account for the marker ID column
        p2_idx = parent2_idx + 1

        # Initialize offspring genotype
        offspring_haploid = np.zeros(self.num_markers, dtype=int)

        # Simulate recombination for each chromosome
        for chrom in self.marker_positions:
            markers_on_chrom = self.marker_positions[chrom]

            # Randomly choose initial parent
            current_parent = random.choice([p1_idx, p2_idx])

            # Average 1 recombination event per chromosome (can be adjusted)
            num_recombinations = np.random.poisson(1)

            if num_recombinations > 0 and len(markers_on_chrom) > 1:
                # Choose recombination points
                recomb_points = sorted(random.sample(range(1, len(markers_on_chrom)),
                                                  min(num_recombinations, len(markers_on_chrom)-1)))

                current_segment = 0
                for i, (marker, _) in enumerate(markers_on_chrom):
                    marker_row = self.genotypes.index.get_loc(marker)

                    if current_segment < len(recomb_points) and i >= recomb_points[current_segment]:
                        current_parent = p1_idx if current_parent == p2_idx else p2_idx
                        current_segment += 1

                    # For haploid organisms, directly copy the allele from current parent
                    offspring_haploid[marker_row] = self.genotypes.iloc[marker_row, current_parent]
            else:
                # No recombination, just inherit from one parent
                for marker, _ in markers_on_chrom:
                    marker_row = self.genotypes.index.get_loc(marker)
                    offspring_haploid[marker_row] = self.genotypes.iloc[marker_row, current_parent]

        return offspring_haploid

    def create_offspring(self, parent_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create haploid offspring genotypes from parent pairs.

        Args:
            parent_pairs: List of tuples with parent indices

        Returns:
            Array of haploid offspring genotypes (num_offspring x num_markers)
        """
        offspring_genotypes = np.zeros((self.num_offspring, self.num_markers), dtype=int)

        for i in range(self.num_offspring):
            # Randomly select a parent pair
            parent1_idx, parent2_idx = random.choice(parent_pairs)

            # For haploid organisms, we only need one gamete
            # (rather than combining two gametes as in diploid organisms)
            offspring_genotypes[i] = self.simulate_recombination(parent1_idx, parent2_idx)

        return offspring_genotypes

    def calculate_phenotypes(self, offspring_genotypes: np.ndarray) -> pd.DataFrame:
        """
        Calculate phenotypes for offspring based on their genotypes and locus effects.

        Args:
            offspring_genotypes: Array of offspring genotypes

        Returns:
            DataFrame with phenotype values for each trait for each offspring
        """
        num_offspring = offspring_genotypes.shape[0]
        phenotypes = pd.DataFrame(index=range(num_offspring),
                                columns=[f"trait_{i+1}" for i in range(self.num_traits)])

        # Calculate genetic values for each trait
        for trait in range(1, self.num_traits + 1):
            trait_effects = self.locus_effects[self.locus_effects['trait'] == trait]

            # Initialize genetic values
            genetic_values = np.zeros(num_offspring)

            # Add additive effects
            for _, row in trait_effects.iterrows():
                try:
                    # Convert locus value to integer, handling scientific notation
                    locus = int(float(row['locus'])) - 1  # Convert to 0-based index
                    add_eff = row['add_eff']

                    # For haploid organisms, each locus only has 0 or 1 copies of the allele
                    # (not 0, 1, or 2 as in diploid organisms)
                    genetic_values += offspring_genotypes[:, locus] * add_eff
                except (ValueError, TypeError, IndexError) as e:
                    # Skip this row if there are any conversion errors
                    continue

            # Add epistatic effects (interactions between pairs of loci)
            # Track which interactions we've already processed
            processed_interactions = set()

            for _, row in trait_effects.iterrows():
                # Convert locus values to integers, handling scientific notation and NaN values
                try:
                    locus = int(float(row['locus'])) - 1  # Convert to 0-based index

                    # Handle NaN in epi_loc (means no epistatic interaction)
                    if pd.isna(row['epi_loc']):
                        continue

                    epi_loc = int(float(row['epi_loc'])) if row['epi_loc'] != 0 else 0
                    epi_eff = row['epi_eff']

                    if epi_loc <= 0:  # Skip if no epistatic interaction
                        continue

                    epi_loc -= 1  # Convert to 0-based index

                    # Create a unique identifier for this interaction (order-independent)
                    interaction_pair = tuple(sorted([locus, epi_loc]))

                    # Only process each interaction once
                    if interaction_pair not in processed_interactions:
                        processed_interactions.add(interaction_pair)

                        # For haploid organisms, interaction occurs when both loci have the allele
                        # (value of 1, not just >0 as in diploid organisms)
                        interaction = (offspring_genotypes[:, locus] == 1) & (offspring_genotypes[:, epi_loc] == 1)
                        genetic_values += interaction * epi_eff
                except (ValueError, TypeError, IndexError) as e:
                    # Skip this row if there are any conversion errors
                    continue

            # Calculate genetic variance
            genetic_var = np.var(genetic_values)

            # Calculate environmental variance based on heritability
            if genetic_var > 0:
                env_var = genetic_var * (1 - self.heritability) / self.heritability
            else:
                env_var = 1.0  # Default value if genetic variance is zero

            # Add environmental noise
            env_noise = np.random.normal(0, np.sqrt(env_var), num_offspring)
            final_phenotypes = genetic_values + env_noise

            # Store phenotypes
            phenotypes[f"trait_{trait}"] = final_phenotypes

        return phenotypes

    def run_simulation(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Run the full simulation pipeline.

        Returns:
            Tuple containing offspring genotypes and phenotypes
        """
        # Sample parents
        num_pairs = max(self.num_offspring // 10, 10)  # Ensure reasonable diversity
        parent_pairs = self.sample_parents(num_pairs)

        # Create offspring genotypes
        offspring_genotypes = self.create_offspring(parent_pairs)

        # Calculate phenotypes
        offspring_phenotypes = self.calculate_phenotypes(offspring_genotypes)

        return offspring_genotypes, offspring_phenotypes

    def save_results(self,
                    offspring_genotypes: np.ndarray,
                    offspring_phenotypes: pd.DataFrame,
                    genotype_output: str = "offspring_genotypes.txt",
                    phenotype_output: str = "offspring_phenotypes.txt"):
        """
        Save simulation results to files.

        Args:
            offspring_genotypes: Array of offspring genotypes
            offspring_phenotypes: DataFrame of offspring phenotypes
            genotype_output: Path to save offspring genotypes
            phenotype_output: Path to save offspring phenotypes
        """
        # Save genotypes in the same format as input (markers as rows, individuals as columns)
        # Create a DataFrame with the same structure as the input genotypes
        geno_df = pd.DataFrame(index=self.genotypes.index)

        # Add columns for each offspring
        for i in range(self.num_offspring):
            col_name = f"offspring_{i+1}"
            geno_df[col_name] = offspring_genotypes[i, :]

        # Save with the same separator as input
        geno_df.to_csv(genotype_output, sep='\t')

        # Save phenotypes
        # Create row names that match the offspring columns in genotype file
        offspring_phenotypes.index = [f"offspring_{i+1}" for i in range(self.num_offspring)]
        offspring_phenotypes.to_csv(phenotype_output, sep='\t')

        print(f"Saved offspring genotypes to {genotype_output}")
        print(f"Saved offspring phenotypes to {phenotype_output}")
        print(f"Genotype file format: {geno_df.shape[0]} markers (rows) × {geno_df.shape[1]} individuals (columns)")
        print(f"Phenotype file format: {offspring_phenotypes.shape[0]} individuals (rows) × {offspring_phenotypes.shape[1]} traits (columns)")


# Example usage
if __name__ == "__main__":
    # File paths
    genotype_file = "alphasimr_output/test_sim_WF_1kbt_10000n_5000000bp_g.txt"
    locus_effects_file = "alphasimr_output/test_sim_WF_1kbt_10000n_5000000bp_eff.txt"

    # Initialize simulation
    sim = GeneticSimulation(
        genotype_file=genotype_file,
        locus_effects_file=locus_effects_file,
        num_offspring=10000,
        heritability=0.5,
        random_seed=42
    )

    # Run simulation
    offspring_genotypes, offspring_phenotypes = sim.run_simulation()

    # Save results
    sim.save_results(
        offspring_genotypes=offspring_genotypes,
        offspring_phenotypes=offspring_phenotypes,
        genotype_output="offspring_genotypes.txt",
        phenotype_output="offspring_phenotypes.txt"
    )