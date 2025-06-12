# Empirical scaling of deep learning models in epistasis prediction

## Purpose
This repository contains all scripts needed to replicate the analyses in the pub [TODO: add title and link]. These abalyses test the ability of simple deep learning models to learn epistasic interactions is a series of simulated genotype-phenotype datasets.

Our scripts and analyses are split up into three experiments that correspond to three sections in the pub:

#### Experiment 1: "Scaling"
This set of scripts tests the ability of a simple MLP neural network to capture epistasis in a simulated genotype to phenotype mapping task.
We generate data across a variety of genetic acrhitectures, sample sizes, and QTL numbers to figure out how much data is needed for an MLP to learn epistasis.

The directory for this experiment is: ```workflow_sims/alphasimr_scaling```

#### Experiment 2: "Dilution"
This set of scripts focuses on one base scenario from the scaling experiment (100 QTLs and 10,000 samples) and tests the ability of an MLP to learn epistasis when the causal QTLs are diluted with progressively larger numbers of neutral QTLs.

The directory for this experiment is: ```workflow_sims/alphasimr_dilution```

#### Experiment 3: "Pleiotropy/Genetic correlation"
This set of scripts also focuses on one base scenario of 100 QTLs and 10,000 samples but tests how training on multiple phenotypes that are genetically correlated to varying degrees boosts MLP performance.

The directory for this experiment is: ```workflow_sims/alphasimr_pleio```

## Hardware Requirements

We ran the three experimental pipelines on a a GPU based AWS EC2 instance (g4dn.8xlarge) with 12 vCPUs, 128Gb of RAM, a 1Tb hard drive, and a T4 Tensore Core GPU.
These hardware requirements are only necessary if you wish to replicate the large sample size simulations (10^6 samples) of the scaling experiment. The smaller sample size simulations can be run with 30Gb of RAM (e.g. on a g4dn.2xlarge instance) and take up much less drive space. See the Snakemake workflow instructions below for details on how to avoid replicating the large sample size simulations.

A GPU greatly speeds up model fitting in PyTorch but is not strictly required. However, expect run times to be exceptionally slow when fitting models for simulations with more than 10^3 samples or QTLs.

Runtime for the first scaling simulations is on the order of a week, the dilution and pleiotropy simulations take around 48 hours if run without parallelization. Runtime will be reduced significantly for the scaling simulations if using the pre-generated data (see Uploaded data below) and avoiding the largest sample size simulations.

## Data

All input data required to reproduce the results in the pub is generated with the following simulations scripts:
   - ```workflow_sims/alphasimr_scaling/alphasim_generate.R```
   - ```workflow_sims/alphasimr_dilution/alphasim_generate.R```
   - ```workflow_sims/alphasimr_pleio/alphasim_generate.R```

These scripts are set-up to be run as part of a snakemake pipeline described below.

### Uploaded data
Alternatively we have uploaded the genotype, phenotype and QTL effect size files we generated in our simulations to the following [zenodo repository](https://zenodo.org/records/15644566).

This repo includes 3 files:
   - ```alphasimr_scaling_input.tar.xz``` simulated data for the scaling experiment (except the largest sim reps of 1e06 samples)
   - ```alphasimr_scaling_1e06_input.tar.xz``` simulated data for the scaling experiment from the largest sim reps of 1e06 samples
   - ```alphasimr_dilution_input.tar.xz``` simulated data for the dilution experiment
   - ```alphasimr_pleio_input.tar.xz``` simulated data for the pleiotropy experiment

Simply download these files, and then extract them to the correct input data directory for each experiment

For example: ```tar -xJf alphasimr_pleio_input.tar.xz -C workflow_sims/alphasimr_pleio/alphasimr_output```

The same directory structure is used for all 3 experiments. You will likely have to ```mkdir``` these ```alphasimr_output``` directories before extracting into them.

## Installation and Setup

To directly replicate the environments used to produce the pub, first install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

Then create and activate a new virtual environment:

```bash
conda env create -n snakemake --file workflow_sims/envs/snakemake.yml
conda activate snakemake
```

## Snakemake workflow

To generate simulated data and fit all models, first install and activate the conda environment as described above and then run this command: `bash run_snakemake_pipeline.sh`.

This script will execute all snakemake pipelines sequentially, allowing for parallelizaiton within snakemake workflow if the ```--cores``` parameter is set to more than 1. In principle if hardware allows, these pipeline can be run in parallel by the user with a modified pipeline script.

### Workflow description

For each experiment you will see the following general snakemake files:
   - ```workflow_sims/alphasimr_*/Snakefile_sims.smk``` this pipeline executes the job ```run_sims``` to generate and save simulated data using the R package AlphaSimR
   - ```workflow_sims/alphasimr_*/Snakefile_linear_mod.smk``` this takes the output from the sims and runs ```run_python_rrBLUP``` which fits a ridge-regression model using scikit-learn as follows:
   - ```workflow_sims/alphasimr_*/Snakefile_gpatlas.smk``` this workflow also takes the output from the sims and fits models using PyTorch
      - runs ```generate_input_data``` to generate hdf5 files for input to PyTorch
      - runs ```optimize_fit_gpnet```/```fit_gpnet``` to fit an MLP predicting simulated phenotype from simulated genotypes either with or without hyperparameter optimization depending on the experiment.

Additionally the dilution experiment has another workflow ```workflow_sims/alphasimr_dilution/Snakefile_feat_seln```.
This runs a modified verion of the ```workflow_sims/alphasimr_*/Snakefile_gpatlas.smk``` workflow where feature seleciton is performed using LASSO regression in the rule ```optimize_fit_feat_seln_gpnet```

### Changing simulation parameters
For the first 'scaling' experiment described in the pub you may wish to avoid running the 10^5 and 10^6 sample size simulations due to the hardware requirements.
To do so, you can re-rerun the ```workflow_sims/alphasimr_scaling/generate_simulation_reps.ipynb``` notebook, which generates the config file ```workflow_sims/alphasimr_scaling/Snakemake_wildcard_config.yaml```  which captures the simulation parameters combinations snakemake will execute. Simply edit the ```sample_sizes``` dictionary in the notebook to contain the smaller sample sizes you would like to simulate and run the notebook to generate an updated config file. This notebook requires you to first create a Conda environment from the file `workflow_sims/envs/gpatlas.yml`.

The same process can be done to change the number of replicates for the scaling experiment. Otherwise all parameters for the other simulations are captured in the header of their respective snakemake files and can be edited as needed. Be warned that in order for the pipelines to work properly, all the snakefiles generating results for one experiment must have the same parameter combinations, so their headers must be identical.

## Visualize results

The three main figures in the pub are generated in Jupyter notebooks found in the three experiment directories:
   - ```workflow_sims/alphasimr_scaling/Fig_1_scaling.ipynb```
   - ```workflow_sims/alphasimr_dilution/Fig_2_dilution.ipynb```
   - ```workflow_sims/alphasimr_pleio/Fig_3_pleiotropy.ipynb```

If you would like to re-run these notebooks, create a Conda environment from the file `workflow_sims/envs/gpatlas.yml` and use it to run the notebooks.

### Description of the folder structure

```
sandler_gpatlas_data
├── workflow_sims
│   ├── alphasimr_scaling        #pipeline for recreating Experiment 1 (Scaling)
│   │    ├── Fig_1_scaling.ipynb          #notebook for recreating scaling experiment result Fig 1.
│   │    ├── Fig_supplement.ipynb         #notebook for comparing analytical and SGD fit linear models (Sup Fig 1.)
│   │    ├── generate_simulation_reps.ipynb        #notebook for writing simulation replicates config
│   │    ├── Snakemake_wildcard_config.yaml        #simulation replicates config
│   │    └── Snakemake*.smk         #pipelines for generating sims, fitting linear, and MLP models.
│   ├── alphasimr_dilution          #pipeline for recreating Experiment 2 (Dilution)
│   │    ├── Fig_2_dilution.ipynb         #notebook for recreating dilution experiment result Fig 2.
│   │    └── Snakemake*.smk        #pipelines for generating sims, fitting linear, MLP, and feature seln. models.
│   ├── alphasimr_pleio          #pipeline for recreating Experiment 3 (Genetic correlations + MTL)
│   │    ├── Fig_3_pleiotropy.ipynb          #notebook for recreating pleiotropy experiment result Fig 3.
│   │    └── Snakemake*.smk        #pipelines for generating sims, fitting linear, and MLP models.
│   ├── envs         #conda environments needed for recreating results
│   └── gpatlas       #python module with geno-pheno modeling functionality
└── run_snakemake_pipeline.sh       #master script for running all snakemake pipelines
```
