#coalescent pop scenarios to use
SIM_SCENARIO = ['sim_WF_null', 'sim_WF_10kbt', 'sim_WF_1kbt']
BP_LEN = [5000000]
SAMPLE_SIZE = [10000]


rule all:
   input:
        expand("alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}_BLUP.txt", sim_scenario = SIM_SCENARIO, bp_len=BP_LEN, sample_size = SAMPLE_SIZE),
        expand("alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}_pheno_pred.png", sim_scenario = SIM_SCENARIO, bp_len=BP_LEN, sample_size = SAMPLE_SIZE)


#generate simulated data
rule run_rrBLUP:
    conda: 'blup'
    input:
        input_pheno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_p.txt',
        input_geno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_g.txt',
        loci_effects = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_eff.txt'
    output:
        correlation_summary = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}_BLUP.txt',
        rrBLUP_pheno_pred_plot = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}_pheno_pred.png'
    params:
        sim_scenario = "{sim_scenario}",
        bp_len = "{bp_len}",
        sample_size = "{sample_size}"
    script:
        'rrBLUP.R'
