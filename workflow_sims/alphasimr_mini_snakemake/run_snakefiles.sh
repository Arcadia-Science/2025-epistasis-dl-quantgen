
snakemake -s Snakefile_sims.smk --use-conda --rerun-incomplete --cores 1
snakemake -s Snakefile_linear_mod.smk --use-conda --cores 1
snakemake -s Snakefile_gpatlas.smk --use-conda --cores 1