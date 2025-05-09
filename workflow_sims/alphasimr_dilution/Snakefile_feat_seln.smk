
MARKER_N = [100, 250, 500, 750, 1000, 2500, 5000, 10000]
QTL_N = [100]
SAMPLE_SIZE = [10000]
REP = list(range(1, 6))  # 5 replicates, adjust as needed

rule all:
   input:
        expand('gphybrid/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_mlp_pruned_correlations.csv',
               qtl_n=QTL_N, marker_n=MARKER_N, sample_size=SAMPLE_SIZE, rep=REP)
        #expand('gpnet/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_phenotype_correlations_untuned.csv',
        #       qtl_n=QTL_N, marker_n=MARKER_N, sample_size=SAMPLE_SIZE, rep=REP)

#create hdf5 files for input to gpatlas
rule generate_input_data:
    conda: 'gpatlas'
    input:
        input_pheno = "alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_p.txt",
        input_geno = "alphasimr_output/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_g.txt"
    output:
        train_data_input = 'gpnet/input_data/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_train.hdf5',
        test_data_input = 'gpnet/input_data/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_test.hdf5'
    resources:
        mem_mb=25000,
    params:
        rep = "{rep}"
    script:
        'prep_data.py'

rule optimize_fit_gpnet:
    #conda: "envs/gpatlas.yml"
    conda: 'gpatlas'
    input:
        input_train_data = rules.generate_input_data.output.train_data_input,
        input_test_data = rules.generate_input_data.output.test_data_input,
    output:
        pheno_corrs = 'gphybrid/qhaplo_{qtl_n}qtl_{marker_n}marker_{sample_size}n_rep{rep}_mlp_pruned_correlations.csv'
    resources:
    params:
        sample_size = "{sample_size}",
        qtl_n = "{qtl_n}",
        marker_n = "{marker_n}",
        rep = "{rep}"
    threads: 12
    script:
        "gplinear_feat_set_optuna.py"
