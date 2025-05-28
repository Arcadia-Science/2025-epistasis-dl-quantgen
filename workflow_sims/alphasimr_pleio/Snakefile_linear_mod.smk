QTL_N = [100, 200, 300, 500, 1000]
SAMPLE_SIZE = [10000]
PLEIO_STRENGTH = [0, 0.25, 0.5, 0.75, 0.95]
TRAIT_N = [10, 100]
REP = list(range(1, 6))  # 5 replicates, adjust as needed

onstart:
    shell("mkdir -p linear_model")

rule all:
    input:
        expand("linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_scklrr_corr_summary.txt",
               qtl_n=QTL_N,
               sample_size=SAMPLE_SIZE,
               pleio_strength=PLEIO_STRENGTH,
               trait_n=TRAIT_N,
               rep=REP)

#fit rrBLUP approximation through sci-kit learn ridge regression (cross validated)
rule run_python_rrBLUP:
    conda:
        '../envs/gpatlas.yml'
    input:
        input_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_p.txt',
        input_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_eff.txt'
    output:
        correlation_summary = 'linear_model/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_scklrr_corr_summary.txt',
    resources:
        mem_mb=25000
    script:
        'fit_ridge_cv.py'
