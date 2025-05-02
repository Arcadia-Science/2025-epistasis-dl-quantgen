
configfile: "Snakemake_wildcard_config.yaml"

# Convert config parameter sets to the format your function expects
VALID_COMBINATIONS = [
    {"sample_size": p["sample_size"], "qtl_n": p["qtl_n"]}
    for p in config["parameter_sets"]
]

# Function to generate outputs based on a pattern
def get_valid_outputs(pattern):
    return [pattern.format(qtl_n=combo["qtl_n"], sample_size=combo["sample_size"])
            for combo in VALID_COMBINATIONS]

rule all:
    input:
        get_valid_outputs("alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_p.txt"),
        get_valid_outputs("alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_g.txt")


#generate simulated data
rule run_sims:
    conda: 'R'
    output:
        output_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_p.txt',
        output_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_eff.txt'
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}"
    script:
        'alphasim_generate.R'
