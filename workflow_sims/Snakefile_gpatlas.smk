#coalescent pop scenarios to use
#SIM_SCENARIO = [ 'sim_WF_10kbt']
SIM_SCENARIO = ['sim_WF_null', 'sim_WF_10kbt', 'sim_WF_1kbt']
BP_LEN = [5000000]
SAMPLE_SIZE = [10000]


rule all:
   input:
        expand("gpatlas/gpatlas_input/test_{sim_scenario}_{sample_size}n_{bp_len}_test.hdf5", sim_scenario = SIM_SCENARIO, bp_len=BP_LEN, sample_size = SAMPLE_SIZE),
        expand("gpatlas/optuna/test_{sim_scenario}_{sample_size}n_{bp_len}_gg_encoder.pt", sim_scenario = SIM_SCENARIO, bp_len=BP_LEN, sample_size = SAMPLE_SIZE),
        expand("gpatlas/optuna/test_{sim_scenario}_{sample_size}n_{bp_len}_gp_network.pt", sim_scenario = SIM_SCENARIO, bp_len=BP_LEN, sample_size = SAMPLE_SIZE)


#create hdf5 files for input to gpatlas
rule generate_input_data:
    conda: 'gpatlas'
    input:
        input_pheno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_p.txt',
        input_geno = 'alphasimr_output/test_{sim_scenario}_{sample_size}n_{bp_len}bp_g.txt',
    output:
        train_data_input = 'gpatlas/gpatlas_input/test_{sim_scenario}_{sample_size}n_{bp_len}_train.hdf5',
        test_data_input = 'gpatlas/gpatlas_input/test_{sim_scenario}_{sample_size}n_{bp_len}_test.hdf5'
    resources:
        mem_mb=25000
    script:
        'gpatlas/prep_data_h5.py'


rule optimize_gg_encoder:
    conda: "envs/gpatlas.yml"
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        gg_encoder_optimized = 'gpatlas/optuna/test_{sim_scenario}_{sample_size}n_{bp_len}_gg_encoder.pt',
        optuna_gg_csv = 'gpatlas/optuna/test_{sim_scenario}_{sample_size}n_{bp_len}_gg_optuna.csv',
        optuna_gg_json = 'gpatlas/optuna/test_{sim_scenario}_{sample_size}n_{bp_len}_gg_optuna.json'
    resources:
        mem_mb=25000
    script:
        "gpatlas/gpatlas_lite_optuna_gg_snakemake.py"



    #shell:
        #"""
        #pip install torch scikit-learn h5py optuna
        #python gpatlas/gpatlas_lite_optuna_gg_snakemake.py
        #"""
rule optimize_gp_network:
    conda: "envs/gpatlas.yml"
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
        gg_encoder_optimized = rules.optimize_gg_encoder.output.gg_encoder_optimized
    output:
        pp_encoder_optimized = 'gpatlas/optuna/test_{sim_scenario}_{sample_size}n_{bp_len}_pp_encoder.pt',
        gp_encoder_optimized = 'gpatlas/optuna/test_{sim_scenario}_{sample_size}n_{bp_len}_gp_network.pt',
        optuna_gp_csv = 'gpatlas/optuna/test_{sim_scenario}_{sample_size}n_{bp_len}_gp_optuna.csv',
        optuna_gp_json = 'gpatlas/optuna/test_{sim_scenario}_{sample_size}n_{bp_len}_gp_optuna.json'
    resources:
        mem_mb=25000
    script:
        "gpatlas/gpatlas_lite_optuna_gp_snakemake.py"