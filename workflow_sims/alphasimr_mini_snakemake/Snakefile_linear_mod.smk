
QTL_N = [100, 1000]
SAMPLE_SIZE = [1000, 10000, 100000]


rule all:
   input:
        expand("linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_scklrr_corr_summary.txt", qtl_n=QTL_N, sample_size = SAMPLE_SIZE),

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