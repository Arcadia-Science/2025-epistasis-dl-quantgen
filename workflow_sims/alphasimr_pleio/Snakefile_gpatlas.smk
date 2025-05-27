QTL_N = [100, 200, 300, 500, 1000]
SAMPLE_SIZE = [10000]
PLEIO_STRENGTH = [0, 0.25, 0.5, 0.75, 0.95]
TRAIT_N = [10, 100]
REP = list(range(1, 6))  # 5 replicates, adjust as needed

rule all:
    input:
        expand("input_data/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_train.hdf5",
               qtl_n=QTL_N,
               sample_size=SAMPLE_SIZE,
               pleio_strength=PLEIO_STRENGTH,
               trait_n=TRAIT_N,
               rep=REP),

        expand("gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_phenotype_correlations.csv",
               qtl_n=QTL_N,
               sample_size=SAMPLE_SIZE,
               pleio_strength=PLEIO_STRENGTH,
               trait_n=TRAIT_N,
               rep=REP)

# Create hdf5 files for input to gpatlas
rule generate_input_data:
    conda:
        '../envs/gpatlas.yml'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_g.txt',
    output:
        train_data_input = 'input_data/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_train.hdf5',
        test_data_input = 'input_data/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_test.hdf5'
    resources:
        mem_mb=25000,
    params:
        rep = "{rep}"
    script:
        'prep_data.py'

rule optimize_fit_gpnet:
    conda:
        '../envs/gpatlas.yml'
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        pheno_corrs = 'gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_phenotype_correlations.csv'
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        pleio_strength = "{pleio_strength}",
        trait_n = "{trait_n}",
        rep = "{rep}"
    threads: 12
    script:
        "fit_mlp.py"
