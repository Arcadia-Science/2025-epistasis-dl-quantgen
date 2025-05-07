QTL_N = [100,200,300,500,1000]
SAMPLE_SIZE = [10000]
PLEIO_STRENGTH = [0, 0.25, 0.5, 0.75, 0.95]
TRAIT_N = [10, 100]


rule all:
    input:
        expand("input_data/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_train.hdf5", qtl_n=QTL_N,
                                                                                                            sample_size=SAMPLE_SIZE,
                                                                                                            pleio_strength=PLEIO_STRENGTH,
                                                                                                            trait_n=TRAIT_N),

        expand("gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_phenotype_correlations_untuned.csv", qtl_n=QTL_N,
                                                                                                            sample_size=SAMPLE_SIZE,
                                                                                                            pleio_strength=PLEIO_STRENGTH,
                                                                                                            trait_n=TRAIT_N),

        expand("gplinear/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_phenotype_correlations_untuned.csv", qtl_n=QTL_N,
                                                                                                            sample_size=SAMPLE_SIZE,
                                                                                                            pleio_strength=PLEIO_STRENGTH,
                                                                                                            trait_n=TRAIT_N)



#create hdf5 files for input to gpatlas
rule generate_input_data:
    conda: 'gpatlas'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_g.txt',
    output:
        train_data_input = 'input_data/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_train.hdf5',
        test_data_input = 'input_data/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_test.hdf5'
    resources:
        mem_mb=25000,
    script:
        'prep_data.py'


rule optimize_fit_gpnet_untuned:
    #conda: "envs/gpatlas.yml"
    conda: 'gpatlas'
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        pheno_corrs = 'gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_phenotype_correlations_untuned.csv'
    resources:
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        pleio_strength = "{pleio_strength}",
        trait_n = "{trait_n}"
    threads: 12
    script:
        "gpnet.py"

rule optimize_fit_gplinear_untuned:
    #conda: "envs/gpatlas.yml"
    conda: 'gpatlas'
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        pheno_corrs = 'gplinear/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_phenotype_correlations_untuned.csv'
    resources:
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        pleio_strength = "{pleio_strength}",
        trait_n = "{trait_n}"
    threads: 12
    script:
        "gplinear.py"
