import os
from pathlib import Path
import numpy as np
import pickle as pk
import h5py


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
# Read phenotype file
phen_file = open('test_sim_WF_1kbt_100kups_5mb_p.txt', 'r')
phens = phen_file.read().split('\n')
phens = [x.split() for x in phens if x]  # Skip empty lines

# Extract phenotype information
out_dict = {}
out_dict['phenotype_names'] = phens[0][1:]  # Extract header of pheno names from first row
out_dict['strain_names'] = [x[0] for x in phens[1:] if x]  # Strain names from first column
out_dict['phenotypes'] = []

# Convert phenotypes to float, handling NA values
for x in phens[1:]:
    if not x:  # Skip empty lines
        continue
    row_phenos = []
    for y in x[1:]:
        if y == 'NA':
            row_phenos.append(0)  # Or use None/np.nan if preferred
        else:
            row_phenos.append(float(y))
    out_dict['phenotypes'].append(row_phenos)

# Read genotype file - now just a matrix of genotypes
genotype_file = open('genotype.txt', 'r')
gens = genotype_file.read().split('\n')
gens = [x.split() for x in gens if x]  # Skip empty lines

# Since there are no locus IDs, generate them
loci_count = len(gens[0])  # Number of columns = number of loci
out_dict['loci'] = [f"locus_{i}" for i in range(loci_count)]

# Process genotypes - pure matrix format
new_coding_dict = {'0': [1, 0], '1': [0, 1]}
out_dict['genotypes'] = []

for row in gens:
    if not row:  # Skip empty lines
        continue

    # Process all genotypes for this individual
    ind_genotypes = []
    for geno in row:
        if geno in new_coding_dict:
            ind_genotypes.append(new_coding_dict[geno])
        else:
            ind_genotypes.append([0, 0])

    out_dict['genotypes'].append(ind_genotypes)



#################################################################################
#################################################################################
#split test train

in_data = out_dict

out_dict_test = {}
out_dict_train = {}

categories_to_stratefy = ['phenotypes', 'genotypes', 'strain_names']
categories_to_copy = [x for x in in_data.keys() if x not in categories_to_stratefy]

train_length = round(len(in_data['strain_names'])*0.85)

#train set
for x in categories_to_copy:
 out_dict_train[x] = in_data[x]

for x in categories_to_stratefy:
 out_dict_train[x] = in_data[x][:train_length]

#pk.dump(out_dict_train, open('gpatlas/' + file_prefix + '_train.pk','wb'))
save_to_hdf5(out_dict_train, 'test_sim_WF_1kbt_100kups_5mb_train.h5')

del(out_dict_train)

#test set
for x in categories_to_copy:
 out_dict_test[x] = in_data[x]

for x in categories_to_stratefy:
 out_dict_test[x] = in_data[x][train_length:]

#pk.dump(out_dict_test, open('gpatlas/' + file_prefix + '_test.pk','wb'))
save_to_hdf5(out_dict_test, 'test_sim_WF_1kbt_100kups_5mb_test.h5')

del(out_dict_test)
