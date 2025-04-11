
QTL_N = [100, 1000]
SAMPLE_SIZE = [1000, 10000, 100000]


rule all:
   input:
        expand('gpnet/input_data/qhaplo_{qtl_n}qtl_{sample_size}n_train.hdf5', qtl_n = QTL_N, sample_size = SAMPLE_SIZE),
        expand('gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations.csv', qtl_n = QTL_N, sample_size = SAMPLE_SIZE),
        expand('gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations_untuned.csv', qtl_n = QTL_N, sample_size = SAMPLE_SIZE),
        expand("gplinear/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations_untuned.csv", qtl_n=QTL_N, sample_size = SAMPLE_SIZE)


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
        pheno_corrs = 'gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_phenotype_correlations.csv'
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
        "gpliner_untuned.py"

