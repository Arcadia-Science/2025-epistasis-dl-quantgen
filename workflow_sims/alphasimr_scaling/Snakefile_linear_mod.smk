
configfile: "Snakemake_wildcard_config.yaml"

# only run workflow on small sample sizes just to compare to stochastic linear model
VALID_COMBINATIONS = [
    {"sample_size": p["sample_size"], "qtl_n": p["qtl_n"], "rep": p["rep"]}
    for p in config["parameter_sets"]
    if p["sample_size"] <= 10000
]

# Function to generate outputs based on a pattern
def get_valid_outputs(pattern):
    return [pattern.format(qtl_n=combo["qtl_n"], sample_size=combo["sample_size"], rep=combo["rep"])
            for combo in VALID_COMBINATIONS]

onstart:
    shell("mkdir -p linear_model")

rule all:
    input:
        get_valid_outputs('linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_scklrr_corr_summary.txt')

#fit rrBLUP approximation through scikit-learn ridge regression (cross validated)
rule run_python_rrBLUP:
    conda:
        '../envs/gpatlas.yml'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_eff.txt'
    output:
        correlation_summary = 'linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_scklrr_corr_summary.txt',
    resources:
        mem_mb=25000
    script:
        'fit_ridge_cv.py'
