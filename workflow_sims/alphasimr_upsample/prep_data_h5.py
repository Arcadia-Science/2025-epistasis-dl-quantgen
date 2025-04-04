#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pickle as pk
import h5py
import gc


######################################################################################
######################################################################################

def save_to_hdf5_with_progress(data_input: dict, hdf5_path: Path, gzip: bool = True) -> Path:
    data = data_input
    str_dt = h5py.string_dtype(encoding="utf-8")
    total_strains = len(data["strain_names"])

    with h5py.File(hdf5_path, "w") as h5f:
        # Create metadata group and save basic info
        print("Saving metadata...")
        metadata_group = h5f.create_group("metadata")

        loci_array = np.array(data["loci"], dtype=str_dt)
        metadata_group.create_dataset("loci", data=loci_array)

        pheno_names_array = np.array(data["phenotype_names"], dtype=str_dt)
        metadata_group.create_dataset("phenotype_names", data=pheno_names_array)

        # Flush metadata to disk
        h5f.flush()
        print("Metadata saved successfully")

        # Create strains group
        strains_group = h5f.create_group("strains")
        h5f.flush()

        # Track progress and flush regularly
        print(f"Starting to save {total_strains} strains...")
        progress_interval = max(1, total_strains // 100)  # Report progress every 1%
        flush_interval = max(1, total_strains // 20)      # Flush to disk every 5%

        for idx, strain_id in enumerate(data["strain_names"]):
            # Create strain group and datasets
            try:
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

                # Report progress
                if (idx + 1) % progress_interval == 0 or idx == 0 or idx == total_strains - 1:
                    percent = (idx + 1) / total_strains * 100
                    print(f"Progress: {idx + 1}/{total_strains} strains processed ({percent:.1f}%)")

                # Flush to disk periodically
                if (idx + 1) % flush_interval == 0:
                    h5f.flush()
                    print(f"Flushed data to disk at {(idx + 1) / total_strains * 100:.1f}% completion")

            except Exception as e:
                print(f"Error processing strain {strain_id} (index {idx}): {str(e)}")
                raise  # Re-raise the exception after logging

        # Final flush to ensure all data is written
        h5f.flush()
        print(f"All {total_strains} strains saved successfully")
        print(f"{hdf5_path} generated.")

    return hdf5_path
out_dict={}

######################################################################################
######################################################################################

def save_to_hdf5_with_slicing(data_input, hdf5_path, slice_start=None, slice_end=None, gzip=True):
    data = data_input
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(hdf5_path, "w") as h5f:
        metadata_group = h5f.create_group("metadata")

        # These aren't sliced
        loci_array = np.array(data["loci"], dtype=str_dt)
        metadata_group.create_dataset("loci", data=loci_array)

        pheno_names_array = np.array(data["phenotype_names"], dtype=str_dt)
        metadata_group.create_dataset("phenotype_names", data=pheno_names_array)

        strains_group = h5f.create_group("strains")

        # Apply slicing to stratified data
        if slice_start is not None and slice_end is not None:
            strain_range = range(slice_start, slice_end)
        elif slice_start is not None:
            strain_range = range(slice_start, len(data["strain_names"]))
        elif slice_end is not None:
            strain_range = range(0, slice_end)
        else:
            strain_range = range(len(data["strain_names"]))

        # Process only the specified range of strains
        for idx in strain_range:
            strain_id = data["strain_names"][idx]
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

        print(f"{hdf5_path} generated.")

    return hdf5_path
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

phen_file.close()
del phens
gc.collect()

print('pheno extraction done')

len(out_dict['phenotypes'][1])


# Read genotype file - CSV format with headers
genotype_file = open('test_sim_WF_1kbt_100kups_5mb_g.txt', 'r')

#Read and process header line for locus names
header = genotype_file.readline().strip()
out_dict['loci'] = header.split(',')

# Create a temporary directory for chunk files
temp_dir = "temp_genotype_chunks"
os.makedirs(temp_dir, exist_ok=True)

# Process genotypes in chunks
new_coding_dict = {'0': [1, 0], '1': [0, 1]}
chunk_size = 10000  # Adjust based on your memory constraints
chunk_files = []
current_chunk = []
chunk_counter = 0

for line in genotype_file:
    line = line.strip()
    if not line:
        continue

    parts = line.split(',')
    if len(parts) <= 1:
        continue

    # Process genotypes for this individual
    ind_genotypes = []
    for geno in parts[1:]:
        if geno.strip() in new_coding_dict:
            ind_genotypes.append(new_coding_dict[geno.strip()])
        else:
            ind_genotypes.append([0, 0])

    current_chunk.append(ind_genotypes)

    # If chunk is full, write to disk and clear memory
    if len(current_chunk) >= chunk_size:
        chunk_filename = os.path.join(temp_dir, f"genotype_chunk_{chunk_counter}.pk")
        with open(chunk_filename, 'wb') as f:
            pk.dump(current_chunk, f)

        chunk_files.append(chunk_filename)
        chunk_counter += 1
        current_chunk = []
        gc.collect()  # Force garbage collection

# Save last chunk if not empty
if current_chunk:
    chunk_filename = os.path.join(temp_dir, f"genotype_chunk_{chunk_counter}.pk")
    with open(chunk_filename, 'wb') as f:
        pk.dump(current_chunk, f)
    chunk_files.append(chunk_filename)

# Close genotype file to free resources
genotype_file.close()
print('genotype chunking done, combining')

# Now combine all chunks for final output
out_dict['genotypes'] = []
for chunk_file in chunk_files:
    with open(chunk_file, 'rb') as f:
        chunk_data = pk.load(f)
        out_dict['genotypes'].extend(chunk_data)

    # Delete chunk file after reading to free disk space
    os.remove(chunk_file)

# Remove temporary directory
os.rmdir(temp_dir)

print('saving full data')
save_to_hdf5_with_progress(out_dict, 'test_sim_WF_1kbt_100kups_5mb_full.h5')
print('saving done')

"""""
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

print('test train file saving')
"""""
#in_data = out_dict

#train_length = round(len(in_data['strain_names'])*0.85)

# Save train data directly (0 to train_length)
#save_to_hdf5_with_slicing(in_data, 'test_sim_WF_1kbt_100kups_5mb_train.h5',
#                          slice_start=0, slice_end=train_length)

# Save test data directly (train_length to end)
#save_to_hdf5_with_slicing(in_data, 'test_sim_WF_1kbt_100kups_5mb_test.h5',
#                          slice_start=train_length, slice_end=None)