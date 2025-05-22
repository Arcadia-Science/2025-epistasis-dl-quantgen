

cd workflow_sims/alphasimr_mini_snakemake/
snakemake -s Snakefile_sims.smk --use-conda --cores 1 -n
snakemake -s Snakefile_gpatlas.smk --use-conda --cores 1 -n
snakemake -s Snakefile_linear_mod.smk --use-conda --cores 1 -n

cd ../alphasimr_dilution/
snakemake -s Snakefile_sims.smk --use-conda --cores 1 -n
snakemake -s Snakefile_gpatlas.smk --use-conda --cores 1 -n
snakemake -s Snakefile_linear_mod.smk --use-conda --cores 1 -n

cd ../alphasimr_pleio/
snakemake -s Snakefile_sims.smk --use-conda --cores 1 -n
snakemake -s Snakefile_gpatlas.smk --use-conda --cores 1 -n
snakemake -s Snakefile_linear_mod.smk --use-conda --cores 1 -n
