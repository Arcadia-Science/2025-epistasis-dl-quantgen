#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pickle as pk
import h5py

base_file = 'BYxRM'
######################################################################################
######################################################################################

def save_to_hdf5(data_input: dict, hdf5_path: Path, gzip: bool = True) -> Path:
    data = data_input
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(hdf5_path, "w") as h5f:
        metadata_group = h5f.create_group("metadata")

        loci_array = np.array(data["loci"], dtype=str_dt)
        metadata_group.create_dataset("loci", data=loci_array)

        pheno_names_array = np.array(data["phenotype_names"], dtype=str_dt)
        metadata_group.create_dataset("phenotype_names", data=pheno_names_array)

        strains_group = h5f.create_group("strains")

        for idx, strain_id in enumerate(data["strain_names"]):
            strain_grp = strains_group.create_group(strain_id)

            pheno = np.array(data["phenotypes"][idx], dtype=np.float64)
            strain_grp.create_dataset("phenotype", data=pheno)

            pheno_bv = np.array(data["phenotypes_bv"][idx], dtype=np.float64)
            strain_grp.create_dataset("phenotypes_bv", data=pheno_bv)

            genotype = np.array(data["genotypes"][idx], dtype=np.int8)
            strain_grp.create_dataset(
                "genotype",
                data=genotype,
                chunks=True,
                compression="gzip" if gzip else None,
            )

        print(f"{hdf5_path} generated from {data_input}.")

    return hdf5_path
out_dict={}

######################################################################################
######################################################################################

#file_prefix = 'test_sim_WF_1kbt_10000n_5000000bp'
phen_file = open(f'{base_file}_PhenoData.txt' , 'r')

phens = phen_file.read().split('\n')
phens = [x.split() for x in phens]

out_dict['phenotype_names'] = phens[0][1:] #extract header of pheno names from first row
#dict(list(out_dict.items())[2:3])


out_dict['strain_names'] = [x[0] for x in phens[1:-1]] #strain names extracted from first colun skipping one row
out_dict['phenotypes'] = [x[1:] for x in phens[1:-1]]
out_dict['phenotypes'] = [[float(y)  if y!= 'NA' else 0 for y in x[1:]] for x in phens[1:-1]] #convert pheno to float, dealing with NA


# Scale phenotypes to have mean 0 and standard deviation 1
phenotypes_array = np.array(out_dict['phenotypes'], dtype=np.float64)
phenotypes_transposed = phenotypes_array.T  # Transpose to get each phenotype as a row
scaled_phenotypes = []

# Store transformation parameters
means = []
stds = []

for phenotype_values in phenotypes_transposed:
    mean = np.mean(phenotype_values)
    std = np.std(phenotype_values)
    means.append(mean)
    stds.append(std)

    if std == 0:  # Avoid division by zero
        scaled = phenotype_values - mean  # Just center if all values are the same
    else:
        scaled = (phenotype_values - mean) / std
    scaled_phenotypes.append(scaled)

# Save transformation parameters for back-transformation
out_dict['phenotype_scaling'] = {
    'means': means,
    'stds': stds,
    'phenotype_names': out_dict['phenotype_names']
}

# Also save transformation parameters separately for easy access
scaling_params = {
    'means': means,
    'stds': stds,
    'phenotype_names': out_dict['phenotype_names']
}
with open(f'{base_file}_phenotype_scaling.pk', 'wb') as f:
    pk.dump(scaling_params, f)

# Transpose back and convert to list
scaled_phenotypes = np.array(scaled_phenotypes).T.tolist()
out_dict['phenotypes'] = scaled_phenotypes


#########################
#additive breeding values
bv_file = open(f'{base_file}_breeding_vals_scaled.csv' , 'r')

bv = bv_file.read().split('\n')
bv = [x.split(",") for x in bv]

out_dict['phenotypes_bv'] = bv[1:]
out_dict['phenotypes_bv'] = [[float(y) if y != 'NA' else 0 for y in x] for x in bv[1:-1]]

#########################

genotype_file = open(f'{base_file}_GenoData.txt' , 'r')

gens = genotype_file.read().split('\n')
gens = [x.split() for x in gens]

out_dict['loci'] = [x[0] for x in gens[1:-1]]
new_coding_dict = {'B':[1,0],'R':[0,1]}
out_dict['genotypes'] = [[new_coding_dict[x] for x in [gens[y][n] for y in range(len(gens))[1:-1]]] for n in range(len(gens[0]))[1:]]

#dump full dataset
#pk.dump(out_dict, open('gpatlas/' + file_prefix + '.pk','wb'))


#################################################################################
#################################################################################
#split test train

in_data = out_dict

out_dict_test = {}
out_dict_train = {}

categories_to_stratefy = ['phenotypes', 'phenotypes_bv', 'genotypes', 'strain_names']
categories_to_copy = [x for x in in_data.keys() if x not in categories_to_stratefy]

train_length = round(len(in_data['strain_names'])*0.85)

#train set
for x in categories_to_copy:
 out_dict_train[x] = in_data[x]

for x in categories_to_stratefy:
 out_dict_train[x] = in_data[x][:train_length]

#pk.dump(out_dict_train, open('gpatlas/' + file_prefix + '_train.pk','wb'))
save_to_hdf5(out_dict_train, f'{base_file}_train.hdf5' ,)

del(out_dict_train)

#test set
for x in categories_to_copy:
 out_dict_test[x] = in_data[x]

for x in categories_to_stratefy:
 out_dict_test[x] = in_data[x][train_length:]

#pk.dump(out_dict_test, open('gpatlas/' + file_prefix + '_test.pk','wb'))
save_to_hdf5(out_dict_test, f'{base_file}_test.hdf5')

del(out_dict_test)
