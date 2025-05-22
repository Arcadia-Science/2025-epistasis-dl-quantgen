# Empirical scaling of deep learning models in epistasis prediction

## Purpose
This repository contains all scripts needed to replicate the analyses in this **pub**, reporting results on the ability of a simple deep learning model to recover epistasis on a series of simulated benchmarks.


## Installation and Setup

To directly replicate the environments used to produce the pub, first install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

Then create and activate a new virtual environment. TODO

```bash
conda env create -n snakemake --file envs/snakemake.yml
conda activate snakemake
```

## Data

All input data generated for our experiments is reproduced with the simulation scripts

## Snakemake workflow

To generate simulated data and fit all models, execute the script ```run_snakemake_pipeline.sh```

## Visualize results

Python notebooks
Switch env

### Description of the folder structure

```
sandler_gpatas_data
└── workflow_sims
    ├── alphasimr_mini_snakemake #pipeline for recerating Experiment 1 (Scaling)
    ├── alphasimr_dilution #pipeline for recerating Experiment 2 (Dilution)
    └── alphasimr_pleio #pipeline for recerating Experiment 3 (Genetic correlations + MTL)
 ```
