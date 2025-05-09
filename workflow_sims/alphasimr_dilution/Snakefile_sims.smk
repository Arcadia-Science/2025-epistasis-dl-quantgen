
MARKER_N = [100, 250, 500, 750, 1000, 2500, 5000, 10000]
QTL_N = [100]
SAMPLE_SIZE = [10000]
REP = list(range(1, 6))  # 5 replicates, adjust as needed

rule all:
    input:
        expand("alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_p.txt",
               qtl_n=QTL_N, marker_n=MARKER_N, sample_size=SAMPLE_SIZE, rep=REP),
        expand("alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_g.txt",
               qtl_n=QTL_N, marker_n=MARKER_N, sample_size=SAMPLE_SIZE, rep=REP)

#generate simulated data
rule run_sims:
    conda: 'R'
    output:
        output_pheno = "alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_p.txt",
        output_geno = "alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_g.txt",
        loci_effects = "alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_eff.txt"
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        marker_n = "{marker_n}",
        rep = "{rep}"
    script:
        'alphasim_generate.R'


#run linear model
##

#run GPatlas
##
