#coalescent pop scenarios to use
SIM_SCENARIO = ['sim_WF_null', 'sim_WF_10kbt', 'sim_WF_1kbt']
BP_LEN = [5000000]
SAMPLE_SIZE = [10000]


rule all:
   input:
        expand("rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_scklrr_corr_summary.txt", sim_scenario = SIM_SCENARIO, bp_len=BP_LEN, sample_size = SAMPLE_SIZE),
        #expand("rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_rrBLUP_corr_summary.txt", sim_scenario = SIM_SCENARIO, bp_len=BP_LEN, sample_size = SAMPLE_SIZE)


#fit rrBLUP approximation through sci-kit learn ridge regression (cross validated)
rule run_python_rrBLUP:
    conda: 'gpatlas'
    input:
        input_pheno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_p.txt',
        input_geno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_g.txt',
        loci_effects = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_eff.txt'
    output:
        correlation_summary = 'rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_scklrr_corr_summary.txt',
        pred_pheno_output = 'rrBLUP_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_scklrr_pheno_pred.txt'
    resources:
        mem_mb=25000
    script:
        'fit_ridge_cv.py'


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