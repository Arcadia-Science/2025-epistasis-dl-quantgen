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

To generate simulated data and fit all models, first install and load the conda environment in ```workflow_sims\envs\snakemake.yml``` and then execute the script ```run_snakemake_pipeline.sh```.

## Visualize results

The three main figures of the pub are visualized in python notebooks in the 3 experiment directories:
   - ```workflow_sims\alphasimr_scaling\Fig_1_scaling.ipynb```
   - ```workflow_sims\alphasimr_scaling\Fig_2_dilution.ipynb```
   - ```workflow_sims\alphasimr_scaling\Fig_3_pleiotropy.ipynb```

If you would like to re-run these analyses denovo, simply install the conda environment from the file in ```workflow_sims\envs\gpatlas.yml```, load the environment, and run the notebooks. 

### Description of the folder structure

```
sandler_gpatas_data
└── workflow_sims
    ├── alphasimr_scaling #pipeline for recerating Experiment 1 (Scaling)
         ├── Fig_1_scaling.ipynb #notebook for recreating scaling experiment result Fig 1.
         ├── Fig_supplement.ipynb #notebook for comparing analytical and SGD fit lienar models (Sup Fig 1.)
         ├── generate_simulation_reps.ipynb #notebook for writing simulation replicates config
         ├── Snakemake_wildcard_config.yaml #simulation replicates config
         └── Snakemake*.smk pipelines for generating sims, fitting linear, and MLP models.
    ├── alphasimr_dilution #pipeline for recerating Experiment 2 (Dilution)
         ├── Fig_2_dilution.ipynb #notebook for recreating diultion experiment result Fig 2.
         └──  Snakemake*.smk pipelines for generating sims, fitting linear, MLP, and feature seln. models.
    ├── alphasimr_pleio #pipeline for recerating Experiment 3 (Genetic correlations + MTL)
         ├── Fig_3_pleiotropy.ipynb #notebook for recreating pleiotropy experiment result Fig 3.
         └──  Snakemake*.smk pipelines for generating sims, fitting linear, and MLP models.
    ├── envs #conda environments needed for recreating results
    └──gpatlas #python module with geno-pheno modeling functionality
 ```
