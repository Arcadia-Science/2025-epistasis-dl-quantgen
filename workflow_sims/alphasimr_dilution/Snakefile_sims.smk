
MARKER_N = [100, 250, 500, 750, 1000, 2500, 5000, 10000]
QTL_N = [100]
SAMPLE_SIZE = [10000]


rule all:
    input:
        expand("alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_p.txt", qtl_n=QTL_N, marker_n=MARKER_N, sample_size=SAMPLE_SIZE),
        expand("alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_g.txt", qtl_n=QTL_N, marker_n=MARKER_N, sample_size=SAMPLE_SIZE)


#generate simulated data
rule run_sims:
    conda: 'R'
    output:
        output_pheno = "alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_p.txt",
        output_geno = "alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_g.txt",
        loci_effects = "alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_eff.txt"
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        marker_n = "{marker_n}"
    script:
        'alphasim_generate.R'


#run linear model
##

#run GPatlas
##
