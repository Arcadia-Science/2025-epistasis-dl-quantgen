
QTL_N = [10, 20, 30, 40, 50, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000]
SAMPLE_SIZE = [1000, 10000, 100000, 1000000]

VALID_COMBINATIONS = [
    # combinations
    {"qtl_n": 50, "sample_size": 10000},
    {"qtl_n": 80, "sample_size": 10000},
    {"qtl_n": 100, "sample_size": 10000},
    {"qtl_n": 200, "sample_size": 10000},
    {"qtl_n": 300, "sample_size": 10000},
    {"qtl_n": 400, "sample_size": 10000},
    {"qtl_n": 500, "sample_size": 10000},
    {"qtl_n": 600, "sample_size": 10000},

    {"qtl_n": 200, "sample_size": 100000},
    {"qtl_n": 300, "sample_size": 100000},
    {"qtl_n": 400, "sample_size": 100000},
    {"qtl_n": 500, "sample_size": 100000},
    {"qtl_n": 600, "sample_size": 100000},
    {"qtl_n": 700, "sample_size": 100000},
    {"qtl_n": 800, "sample_size": 100000},
    {"qtl_n": 1000, "sample_size": 100000},
    {"qtl_n": 2000, "sample_size": 100000},

    #{"qtl_n": 500, "sample_size": 1000000},
    #{"qtl_n": 600, "sample_size": 1000000},
    #{"qtl_n": 700, "sample_size": 1000000},
    #{"qtl_n": 800, "sample_size": 1000000},
    #{"qtl_n": 900, "sample_size": 1000000},
    #{"qtl_n": 1000, "sample_size": 1000000},
    #{"qtl_n": 2000, "sample_size": 1000000},
    #{"qtl_n": 5000, "sample_size": 1000000},

    {"qtl_n": 10, "sample_size": 1000},
    {"qtl_n": 20, "sample_size": 1000},
    {"qtl_n": 30, "sample_size": 1000},
    {"qtl_n": 40, "sample_size": 1000},
    {"qtl_n": 50, "sample_size": 1000},
    {"qtl_n": 100, "sample_size": 1000},
    {"qtl_n": 200, "sample_size": 1000},
    {"qtl_n": 300, "sample_size": 1000},

    ]

def get_valid_outputs(pattern):
    return [pattern.format(qtl_n=combo["qtl_n"], sample_size=combo["sample_size"])
            for combo in VALID_COMBINATIONS]


rule all:
   input:
        get_valid_outputs('linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_scklrr_corr_summary.txt'),
        #get_valid_outputs('linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_rrBLUP_corr_summary.txt'),
        #get_valid_outputs('linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_scklrr_epi_corr_summary.txt')

#fit rrBLUP approximation through sci-kit learn ridge regression (cross validated)
rule run_python_rrBLUP:
    conda: 'gpatlas'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_eff.txt'
    output:
        correlation_summary = 'linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_scklrr_corr_summary.txt',
    resources:
        mem_mb=25000
    script:
        'fit_ridge_cv.py'


#fit rrBLUP approximation through sci-kit learn ridge regression (cross validated)
rule run_python_rrBLUP_epistatic:
    conda: 'gpatlas'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_eff.txt'
    output:
        correlation_summary = 'linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_scklrr_epi_corr_summary.txt',
    resources:
        mem_mb=25000
    script:
        'fit_ridge_epistasis.py'


"""
#fit rrBLUP (deterministically)
rule run_rrBLUP:
    conda: 'blup'
    input:
        input_pheno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_p.txt',
        input_geno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_g.txt',
        loci_effects = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_eff.txt'
    output:
        correlation_summary = 'rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rrBLUP_corr_summary.txt',
        pred_pheno_output = 'rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rrBLUP_pheno_pred.txt',
        pred_pheno_output_plot = 'rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rrBLUP_pheno_pred.png'
    script:
        'rrBLUP.R'
"""