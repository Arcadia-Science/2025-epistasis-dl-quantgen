configfile: "Snakemake_wildcard_config.yaml"

# Convert config parameter sets to the format your function expects
VALID_COMBINATIONS = [
    {"sample_size": p["sample_size"], "qtl_n": p["qtl_n"], "rep": p["rep"]}
    for p in config["parameter_sets"]
]

# Function to generate outputs based on a pattern
def get_valid_outputs(pattern):
    return [pattern.format(qtl_n=combo["qtl_n"], sample_size=combo["sample_size"], rep=combo["rep"])
            for combo in VALID_COMBINATIONS]

rule all:
   input:
        get_valid_outputs('gpnet/input_data/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_train.hdf5'),
        get_valid_outputs('gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_phenotype_correlations_untuned.csv'),
        get_valid_outputs("gplinear/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_phenotype_correlations_untuned.csv"),


#create hdf5 files for input to gpatlas
rule generate_input_data:
    conda:
        '../envs/gpatlas.yml'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_g.txt',
    output:
        train_data_input = 'gpnet/input_data/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_train.hdf5',
        test_data_input = 'gpnet/input_data/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_test.hdf5'
    script:
        'prep_data.py'

rule optimize_fit_gpnet_untuned:
    conda:
        '../envs/gpatlas.yml'
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        pheno_corrs = 'gpnet/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_phenotype_correlations_untuned.csv'
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        rep = "{rep}"
    threads: 12
    script:
        "gpnet_untuned.py"

rule optimize_fit_gplinear_untuned:
    conda:
        '../envs/gpatlas.yml'
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        pheno_corrs = 'gplinear/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_phenotype_correlations_untuned.csv'
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        rep = "{rep}"
    threads: 12
    script:
        "gplinear_untuned.py"
