

cd workflow_sims/alphasimr_scaling/
snakemake -s Snakefile_sims.smk --use-conda --cores 1
snakemake -s Snakefile_gpatlas.smk --use-conda --cores 1
snakemake -s Snakefile_linear_mod.smk --use-conda --cores 1

cd ../alphasimr_dilution/
snakemake -s Snakefile_sims.smk --use-conda --cores 3
snakemake -s Snakefile_gpatlas.smk --use-conda --cores 3
snakemake -s Snakefile_linear_mod.smk --use-conda --cores 5
snakemake -s Snakefile_feat_seln.smk --use-conda --cores 4

cd ../alphasimr_pleio/
snakemake -s Snakefile_sims.smk --use-conda --cores 1
snakemake -s Snakefile_gpatlas.smk --use-conda --cores 1
snakemake -s Snakefile_linear_mod.smk --use-conda --cores 1

cd ../..
