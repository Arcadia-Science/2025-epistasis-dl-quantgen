
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
    {"qtl_n": 1000, "sample_size": 1000000},
    {"qtl_n": 10000, "sample_size": 1000000}
    ]

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


#run linear model
##

#run GPatlas
##
