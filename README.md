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
    ├── alphasimr_scaling #pipeline for recerating Experiment 1 (Scaling)
         ├──Fig_1_scaling.ipynb #notebook for recreating scaling experiment result Fig 1.
         ├──Fig_supplement.ipynb #notebook for comparing analytical and SGD fit lienar models (Sup Fig 1.)
         ├──generate_simulation_reps.ipynb #notebook for generating simulation replicates if new combinations needed, outputs Snakemake_wildcard_config.yaml
         ├──Snakemake_wildcard_config.yaml #config file of all parameter simulation combinations/reps for snakemake pipeline
         └── Snakemake*.smk pipelines for generating sims, fitting linear and MLP models.
    ├── alphasimr_dilution #pipeline for recerating Experiment 2 (Dilution)
    └── alphasimr_pleio #pipeline for recerating Experiment 3 (Genetic correlations + MTL)
 ```
