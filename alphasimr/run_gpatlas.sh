#!bin/bash

#####LTEE data
src/g_p_atlas_full.py \
--dataset_path ../sandler_gpatlas_data/alphasimr/gpatlas_out/ \
--train_suffix ../sim_2trait_jan03_train.pk \
--test_suffix ../sim_2trait_jan03_test.pk \
--n_phens 2 \
--n_phens_to_predict 2 \
--n_phens_to_analyze 2 \
--e_hidden_dim 2000 \
--d_hidden_dim 2000 \
--latent_dim 2000 \
--sd_noise 0.01 \
--n_locs 4297 \
--n_alleles 2 \
--n_loci_measured 4200 \
--ge_hidden_dim 4000 \
--g_latent_dim 4000 \
--n_epochs_gen 3 \
--gq_to_p_hidden_dim 2000 \
--n_epochs_gen_phen 3 \
--n_env 1 \
--batch_size 200 \
--n_epochs 1 \
--n_cpu 5




#second set of 50/50 SFS traits
#####LTEE data
src/g_p_atlas_full_test.py \
--dataset_path ../sandler_gpatlas_data/alphasimr/gpatlas_out/ \
--train_suffix ../sim_2trait_dec20_train.pk \
--test_suffix ../sim_2trait_dec20_test.pk \
--n_phens 2 \
--n_phens_to_predict 2 \
--n_phens_to_analyze 2 \
--e_hidden_dim 7 \
--d_hidden_dim 7 \
--latent_dim 7 \
--sd_noise 0.01 \
--n_locs 4200 \
--n_alleles 2 \
--n_loci_measured 4200 \
--ge_hidden_dim 3000 \
--g_latent_dim 3000 \
--n_epochs_gen 50 \
--gq_to_p_hidden_dim 7 \
--n_epochs_gen_phen 50 \
--n_env 1 \
--batch_size 50 \
--n_epochs 10 \
--n_cpu 5


#second set of realistic SFS traits with no MSE reduction for captum
#####LTEE data
src/g_p_atlas_full_test.py \
--dataset_path ../sandler_gpatlas_data/alphasimr/gpatlas_out/ \
--train_suffix ../sim_2trait_jan02_train.pk \
--test_suffix ../sim_2trait_jan02_test.pk \
--n_phens 2 \
--n_phens_to_predict 2 \
--n_phens_to_analyze 2 \
--e_hidden_dim 200 \
--d_hidden_dim 200 \
--latent_dim 200 \
--sd_noise 0.01 \
--n_locs 4207 \
--n_alleles 2 \
--n_loci_measured 4207 \
--ge_hidden_dim 5000 \
--g_latent_dim 5000 \
--n_epochs_gen 40 \
--gq_to_p_hidden_dim 200 \
--n_epochs_gen_phen 50 \
--n_env 1 \
--batch_size 50 \
--n_epochs 10 \
--n_cpu 5