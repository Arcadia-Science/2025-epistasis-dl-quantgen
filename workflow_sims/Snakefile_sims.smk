

#coalescent pop scenarios to use
SIM_SCENARIO = ['sim_WF_null', 'sim_WF_10kbt', 'sim_WF_1kbt']
BP_LEN = [5000000]
SAMPLE_SIZE = [10000]


rule all:
   input:
        expand("alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_p.txt", sim_scenario = SIM_SCENARIO, bp_len=BP_LEN, sample_size = SAMPLE_SIZE),
        expand("alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_g.txt", sim_scenario = SIM_SCENARIO, bp_len=BP_LEN, sample_size = SAMPLE_SIZE)




#generate simulated data
rule run_sims:
    conda: 'R'
    input:
        'population_setup.R'
    output:
        output_pheno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_p.txt',
        output_geno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_g.txt',
        loci_effects = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_eff.txt',
        sim_summary = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_summary.txt',
        trait_architecture = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_trait_architecture.RDS',
        sim_LD_plot = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_LD.png',
    params:
        sim_scenario = "{sim_scenario}",
        bp_len = "{bp_len}",
        sample_size = "{sample_size}"
    script:
        'generate_sims.R'


#run linear model
##

#run GPatlas
##
