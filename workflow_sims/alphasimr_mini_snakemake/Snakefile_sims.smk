
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
        get_valid_outputs("alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_p.txt"),
        get_valid_outputs("alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_g.txt")

#generate simulated data
rule run_sims:
    conda:
        '../envs/R.yml'
    output:
        output_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_p.txt',
        output_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_rep{rep}_eff.txt'
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        rep = "{rep}"
    script:
        'alphasim_generate.R'
