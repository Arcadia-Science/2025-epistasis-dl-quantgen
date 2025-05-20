QTL_N = [100, 200, 300, 500, 1000]
SAMPLE_SIZE = [10000]
PLEIO_STRENGTH = [0, 0.25, 0.5, 0.75, 0.95]
TRAIT_N = [10, 100]
REP = list(range(1, 6))  # 5 replicates, adjust as needed

rule all:
    input:
        expand("alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_p.txt",
               qtl_n=QTL_N,
               sample_size=SAMPLE_SIZE,
               pleio_strength=PLEIO_STRENGTH,
               trait_n=TRAIT_N,
               rep=REP),
        expand("alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_g.txt",
               qtl_n=QTL_N,
               sample_size=SAMPLE_SIZE,
               pleio_strength=PLEIO_STRENGTH,
               trait_n=TRAIT_N,
               rep=REP)

# Generate simulated data
rule run_sims:
    conda: 'R'
    output:
        output_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_p.txt',
        output_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_rep{rep}_eff.txt'
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        pleio_strength = "{pleio_strength}",
        trait_n = "{trait_n}",
        rep = "{rep}"  # Add rep parameter
    script:
        'alphasim_generate_pleio.R'
