
QTL_N = [100, 1000, 10000]
SAMPLE_SIZE = [1000, 10000, 100000, 1000000]

VALID_COMBINATIONS = [
    # Original combinations
    {"qtl_n": 100, "sample_size": 1000},
    {"qtl_n": 100, "sample_size": 10000},
    {"qtl_n": 100, "sample_size": 100000},
    {"qtl_n": 1000, "sample_size": 1000},
    {"qtl_n": 1000, "sample_size": 10000},
    {"qtl_n": 1000, "sample_size": 100000},

    # New combinations with 10000 QTLs
    {"qtl_n": 10000, "sample_size": 1000},
    {"qtl_n": 10000, "sample_size": 10000},
    {"qtl_n": 10000, "sample_size": 100000},
    #{"qtl_n": 10000, "sample_size": 1000000},
    ]

def get_valid_outputs(pattern):
    return [pattern.format(qtl_n=combo["qtl_n"], sample_size=combo["sample_size"])
            for combo in VALID_COMBINATIONS]

get_valid_outputs("alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_p.txt")
rule all:
   input:
        get_valid_outputs('gpnet/input_data/qhaplo_{qtl_n}qtl_{sample_size}n_train.hdf5'),
        get_valid_outputs('gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations2.csv'),
        get_valid_outputs('gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations_untuned.csv'),
        get_valid_outputs("gplinear/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations_untuned.csv")


#create hdf5 files for input to gpatlas
rule generate_input_data:
    conda: 'gpatlas'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_g.txt',
    output:
        train_data_input = 'gpnet/input_data/qhaplo_{qtl_n}qtl_{sample_size}n_train.hdf5',
        test_data_input = 'gpnet/input_data/qhaplo_{qtl_n}qtl_{sample_size}n_test.hdf5'
    resources:
        mem_mb=25000,
    script:
        'prep_data.py'


rule optimize_fit_gpnet:
    #conda: "envs/gpatlas.yml"
    conda: 'gpatlas'
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        pheno_corrs = 'gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations2.csv'
    resources:
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}"
    threads: 12
    script:
        "gpnet.py"

rule optimize_fit_gpnet_untuned:
    #conda: "envs/gpatlas.yml"
    conda: 'gpatlas'
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        pheno_corrs = 'gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations_untuned.csv'
    resources:
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}"
    threads: 12
    script:
        "gpnet_untuned.py"

rule optimize_fit_gplinear_untuned:
    #conda: "envs/gpatlas.yml"
    conda: 'gpatlas'
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        pheno_corrs = 'gplinear/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations_untuned.csv'
    resources:
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}"
    threads: 12
    script:
        "gplinear_untuned.py"
