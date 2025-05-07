QTL_N = [100,200,300,500,1000]
SAMPLE_SIZE = [10000]
PLEIO_STRENGTH = [0, 0.25, 0.5, 0.75, 0.95]
TRAIT_N = [10, 100]

rule all:
    input:
        expand("alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_p.txt", qtl_n=QTL_N,
                                                                                                            sample_size=SAMPLE_SIZE,
                                                                                                            pleio_strength=PLEIO_STRENGTH,
                                                                                                            trait_n=TRAIT_N),
        expand("alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_g.txt", qtl_n=QTL_N,
                                                                                                            sample_size=SAMPLE_SIZE,
                                                                                                            pleio_strength=PLEIO_STRENGTH,
                                                                                                            trait_n=TRAIT_N),


#generate simulated data
rule run_sims:
    conda: 'R'
    output:
        output_pheno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_p.txt',
        output_geno = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_g.txt',
        loci_effects = 'alphasimr_output/qhaplo_{qtl_n}qtl_{sample_size}n_{pleio_strength}pleio_{trait_n}trait_eff.txt'
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        pleio_strength = "{pleio_strength}",
        trait_n = "{trait_n}"
    script:
        'alphasim_generate_pleio.R'
