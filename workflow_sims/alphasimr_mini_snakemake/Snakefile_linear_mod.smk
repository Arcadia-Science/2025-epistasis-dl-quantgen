
configfile: "Snakemake_wildcard_config.yaml"

# Convert config parameter sets to the format your function expects
VALID_COMBINATIONS = [
    {"sample_size": p["sample_size"], "qtl_n": p["qtl_n"], "rep": p["rep"]}
    for p in config["parameter_sets"]
]

# Function to generate outputs based on a pattern
def get_valid_outputs(pattern):
    return [pattern.format(qtl_n=combo["qtl_n"], sample_size=combo["sample_size"], rep=combo["rep"])
            for combo in VALID_COMBINATIONS]



rule all:
    input:
        get_valid_outputs('linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_scklrr_corr_summary.txt'),
        #get_valid_outputs('linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_rrBLUP_corr_summary.txt'),
        #get_valid_outputs('linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_scklrr_epi_corr_summary.txt')

#fit rrBLUP approximation through sci-kit learn ridge regression (cross validated)
rule run_python_rrBLUP:
    conda: 'gpatlas'
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

#fit rrBLUP approximation through sci-kit learn ridge regression (cross validated)
rule run_python_rrBLUP_epistatic:
    conda: 'gpatlas'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_eff.txt'
    output:
        correlation_summary = 'linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_scklrr_epi_corr_summary.txt',
    resources:
        mem_mb=25000
    script:
        'fit_ridge_epistasis.py'


"""
#fit rrBLUP (deterministically)
rule run_rrBLUP:
    conda: 'blup'
    input:
        input_pheno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rep{rep}_p.txt',
        input_geno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rep{rep}_g.txt',
        loci_effects = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rep{rep}_eff.txt'
    output:
        correlation_summary = 'rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rep{rep}_rrBLUP_corr_summary.txt',
        pred_pheno_output = 'rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rep{rep}_rrBLUP_pheno_pred.txt',
        pred_pheno_output_plot = 'rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rep{rep}_rrBLUP_pheno_pred.png'
    script:
        'rrBLUP.R'
"""