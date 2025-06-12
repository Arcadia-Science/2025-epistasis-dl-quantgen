
MARKER_N = [100, 250, 500, 750, 1000, 2500, 5000, 10000]
QTL_N = [100]
SAMPLE_SIZE = [10000]
REP = list(range(1, 6))  # 5 replicates, adjust as needed

onstart:
    shell("mkdir -p linear_model")

rule all:
   input:
        expand("linear_model/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_scklrr_corr_summary.txt",
               qtl_n=QTL_N, marker_n=MARKER_N, sample_size=SAMPLE_SIZE, rep=REP)

#fit rrBLUP approximation through sci-kit learn ridge regression (cross validated)
rule run_python_rrBLUP:
    conda:
        '../envs/gpatlas.yml'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_eff.txt'
    output:
        correlation_summary = 'linear_model/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_scklrr_corr_summary.txt',
    resources:
        mem_mb=25000
    params:
        rep = "{rep}"
    script:
        '../common_scripts/fit_ridge_cv.py'
