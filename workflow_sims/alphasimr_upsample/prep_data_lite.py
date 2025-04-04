#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pickle as pk
import h5py
import gc
import sys

######################################################################################
# HDF5 saving function with progress tracking
######################################################################################

def save_to_hdf5_from_chunks(data_dict, chunk_files, output_path, gzip=True):
    """Save data to HDF5 directly from chunk files to minimize memory usage"""
    str_dt = h5py.string_dtype(encoding="utf-8")
    total_strains = len(data_dict["strain_names"])

    with h5py.File(output_path, "w") as h5f:
        # Save metadata first
        print("Saving metadata...")
        metadata_group = h5f.create_group("metadata")

        loci_array = np.array(data_dict["loci"], dtype=str_dt)
        metadata_group.create_dataset("loci", data=loci_array)

        pheno_names_array = np.array(data_dict["phenotype_names"], dtype=str_dt)
        metadata_group.create_dataset("phenotype_names", data=pheno_names_array)

        h5f.flush()
        print("Metadata saved successfully")

        # Create strains group
        strains_group = h5f.create_group("strains")
        h5f.flush()

        # Process chunks one at a time
        strain_idx = 0
        for chunk_idx, chunk_file in enumerate(chunk_files):
            print(f"Processing chunk {chunk_idx+1}/{len(chunk_files)}")

            with open(chunk_file, 'rb') as f:
                chunk_data = pk.load(f)

            # Process each strain in this chunk
            for genotype in chunk_data:
                if strain_idx >= total_strains:
                    print("Warning: More genotypes than strain names, stopping early")
                    break

                strain_id = data_dict["strain_names"][strain_idx]
                try:
                    strain_grp = strains_group.create_group(strain_id)

                    pheno = np.array(data_dict["phenotypes"][strain_idx], dtype=np.float64)
                    strain_grp.create_dataset("phenotype", data=pheno)

                    # Convert to numpy array but maintain one-hot encoding format
                    genotype_array = np.array(genotype, dtype=np.int8)
                    strain_grp.create_dataset(
                        "genotype",
                        data=genotype_array,
                        chunks=True,
                        compression="gzip" if gzip else None,
                    )

                    # Report progress periodically
                    if (strain_idx + 1) % 1000 == 0 or strain_idx == 0 or strain_idx == total_strains - 1:
                        percent = (strain_idx + 1) / total_strains * 100
                        print(f"Progress: {strain_idx + 1}/{total_strains} strains processed ({percent:.1f}%)")
                        h5f.flush()
                        print(f"Flushed data to disk at {percent:.1f}% completion")

                except Exception as e:
                    print(f"Error processing strain {strain_id} (index {strain_idx}): {str(e)}")
                    # Continue with next strain

                strain_idx += 1

            # Free memory after processing each chunk
            del chunk_data
            gc.collect()

            # Delete chunk file after processing
            try:
                os.remove(chunk_file)
                print(f"Removed processed chunk file: {chunk_file}")
            except Exception as e:
                print(f"Warning: Failed to remove chunk file {chunk_file}: {str(e)}")

        h5f.flush()
        print(f"All {strain_idx}/{total_strains} strains processed")
        print(f"{output_path} generated.")

######################################################################################
# Train/Test Split Function with Low Memory Usage
######################################################################################

def save_train_test_split(data_dict, chunk_files, train_output, test_output, train_ratio=0.85, gzip=True):
    """Split data into train/test sets and save directly without loading all data into memory"""
    str_dt = h5py.string_dtype(encoding="utf-8")
    total_strains = len(data_dict["strain_names"])
    train_size = round(total_strains * train_ratio)

    print(f"Creating train set ({train_size} strains) and test set ({total_strains - train_size} strains)")

    # Process train set
    with h5py.File(train_output, "w") as train_h5f:
        # Save metadata
        metadata_group = train_h5f.create_group("metadata")
        loci_array = np.array(data_dict["loci"], dtype=str_dt)
        metadata_group.create_dataset("loci", data=loci_array)
        pheno_names_array = np.array(data_dict["phenotype_names"], dtype=str_dt)
        metadata_group.create_dataset("phenotype_names", data=pheno_names_array)
        train_h5f.flush()

        # Create strains group
        strains_group = train_h5f.create_group("strains")
        train_h5f.flush()

        # Process test set
        with h5py.File(test_output, "w") as test_h5f:
            # Save metadata
            metadata_group = test_h5f.create_group("metadata")
            metadata_group.create_dataset("loci", data=loci_array)
            metadata_group.create_dataset("phenotype_names", data=pheno_names_array)
            test_h5f.flush()

            # Create strains group
            test_strains_group = test_h5f.create_group("strains")
            test_h5f.flush()

            # Process chunks one at a time
            strain_idx = 0
            for chunk_idx, chunk_file in enumerate(chunk_files):
                print(f"Processing chunk {chunk_idx+1}/{len(chunk_files)} for train/test split")

                with open(chunk_file, 'rb') as f:
                    chunk_data = pk.load(f)

                # Process each strain in this chunk
                for genotype in chunk_data:
                    if strain_idx >= total_strains:
                        print("Warning: More genotypes than strain names, stopping early")
                        break

                    strain_id = data_dict["strain_names"][strain_idx]
                    pheno = np.array(data_dict["phenotypes"][strain_idx], dtype=np.float64)
                    genotype_array = np.array(genotype, dtype=np.int8)

                    try:
                        # Determine if this strain belongs to train or test set
                        if strain_idx < train_size:
                            # Add to train set
                            strain_grp = strains_group.create_group(strain_id)
                            strain_grp.create_dataset("phenotype", data=pheno)
                            strain_grp.create_dataset(
                                "genotype",
                                data=genotype_array,
                                chunks=True,
                                compression="gzip" if gzip else None,
                            )
                        else:
                            # Add to test set
                            strain_grp = test_strains_group.create_group(strain_id)
                            strain_grp.create_dataset("phenotype", data=pheno)
                            strain_grp.create_dataset(
                                "genotype",
                                data=genotype_array,
                                chunks=True,
                                compression="gzip" if gzip else None,
                            )

                        # Report progress periodically
                        if (strain_idx + 1) % 1000 == 0 or strain_idx == 0 or strain_idx == total_strains - 1:
                            percent = (strain_idx + 1) / total_strains * 100
                            print(f"Progress: {strain_idx + 1}/{total_strains} strains processed ({percent:.1f}%)")

                            if strain_idx < train_size:
                                train_h5f.flush()
                            else:
                                test_h5f.flush()

                            print(f"Flushed data to disk at {percent:.1f}% completion")

                    except Exception as e:
                        print(f"Error processing strain {strain_id} (index {strain_idx}): {str(e)}")

                    strain_idx += 1

                # Free memory after processing each chunk
                del chunk_data
                gc.collect()

                # Delete chunk file after processing
                try:
                    os.remove(chunk_file)
                    print(f"Removed processed chunk file: {chunk_file}")
                except Exception as e:
                    print(f"Warning: Failed to remove chunk file {chunk_file}: {str(e)}")

            # Final flush for test set
            test_h5f.flush()
            print(f"Test set saved to {test_output}")

        # Final flush for train set
        train_h5f.flush()
        print(f"Train set saved to {train_output}")

    print(f"Train/test split complete: {train_size} train samples, {total_strains - train_size} test samples")

######################################################################################
# Main Script
######################################################################################

try:
    # Create a dictionary to store data
    out_dict = {}

    # Read phenotype file
    print("Reading phenotype file...")
    phen_file = open('test_sim_WF_1kbt_100kups_5mb_p.txt', 'r')
    phens = phen_file.read().split('\n')
    phens = [x.split() for x in phens if x]  # Skip empty lines

    # Extract phenotype information
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

    print('Phenotype extraction done')
    print(f"Number of strains: {len(out_dict['strain_names'])}")
    print(f"Number of phenotypes per strain: {len(out_dict['phenotypes'][0])}")

    # Create a temporary directory for chunk files
    temp_dir = "temp_genotype_chunks"
    os.makedirs(temp_dir, exist_ok=True)

    # Read genotype file in chunks
    print("Reading genotype file in chunks...")
    genotype_file = open('test_sim_WF_1kbt_100kups_5mb_g.txt', 'r')

    # Read and process header line for locus names
    header = genotype_file.readline().strip()
    out_dict['loci'] = header.split(',')
    print(f"Number of loci: {len(out_dict['loci'])}")

    # Process genotypes in chunks
    new_coding_dict = {'0': [1, 0], '1': [0, 1]}  # Maintain one-hot encoding
    chunk_size = 1000  # Adjust based on your memory constraints
    chunk_files = []
    current_chunk = []
    chunk_counter = 0
    line_counter = 0

    print("Processing genotypes in chunks...")
    for line in genotype_file:
        line_counter += 1
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
            print(f"Saved chunk {chunk_counter} with {len(current_chunk)} genotypes")
            chunk_counter += 1
            current_chunk = []
            gc.collect()  # Force garbage collection

        # Report progress periodically
        if line_counter % 10000 == 0:
            print(f"Processed {line_counter} lines from genotype file")

    # Save last chunk if not empty
    if current_chunk:
        chunk_filename = os.path.join(temp_dir, f"genotype_chunk_{chunk_counter}.pk")
        with open(chunk_filename, 'wb') as f:
            pk.dump(current_chunk, f)
        chunk_files.append(chunk_filename)
        print(f"Saved final chunk {chunk_counter} with {len(current_chunk)} genotypes")

    # Close genotype file to free resources
    genotype_file.close()
    print(f'Genotype chunking done: {line_counter} lines processed, {len(chunk_files)} chunks created')

    # Check if we have the expected number of genotypes
    expected_genotypes = len(out_dict['strain_names'])
    if line_counter < expected_genotypes:
        print(f"Warning: Found fewer genotype lines ({line_counter}) than strain names ({expected_genotypes})")

    # Save full dataset
    print('Saving full dataset directly from chunks...')
    save_to_hdf5_from_chunks(out_dict, chunk_files.copy(), 'test_sim_WF_1kbt_100kups_5mb_full.h5')

    # Since saving the full dataset consumed the chunk files, we need to regenerate them
    print("Regenerating genotype chunks for train/test split...")
    genotype_file = open('test_sim_WF_1kbt_100kups_5mb_g.txt', 'r')

    # Skip header
    genotype_file.readline()

    # Re-process genotypes in chunks
    chunk_files = []
    current_chunk = []
    chunk_counter = 0
    line_counter = 0

    for line in genotype_file:
        line_counter += 1
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
            gc.collect()

    # Save last chunk if not empty
    if current_chunk:
        chunk_filename = os.path.join(temp_dir, f"genotype_chunk_{chunk_counter}.pk")
        with open(chunk_filename, 'wb') as f:
            pk.dump(current_chunk, f)
        chunk_files.append(chunk_filename)

    genotype_file.close()

    # Create train/test split
    print('Creating and saving train/test split directly from chunks...')
    save_train_test_split(
        out_dict,
        chunk_files,
        'test_sim_WF_1kbt_100kups_5mb_train.h5',
        'test_sim_WF_1kbt_100kups_5mb_test.h5'
    )

    # Cleanup
    try:
        os.rmdir(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Failed to remove temp directory {temp_dir}: {str(e)}")

    print('All processing complete!')

except Exception as e:
    print(f"Error during processing: {str(e)}")
    print(f"Stack trace: {sys.exc_info()}")

    # Don't delete temp files in case of error
    print("Process terminated with error. Temporary files preserved for debugging.")
    sys.exit(1)