"""
GPatlas: Genetic data analysis toolkit with autoencoder models.

This package provides tools for analyzing genetic data using deep learning models,
particularly focused on genotype-phenotype relationships.
"""

__version__ = "0.1.0"

# Import commonly used functions and classes for easier access
from .datasets import *
from .models import *
"""
from .training import (
train_localgg_model,
train_pp_model,
train_gp_model,
focal_loss_for_genetic_data,
)
from .optimization import run_optimization

# Constants that can be overridden by the user
from .config import DEFAULT_CONFIG """