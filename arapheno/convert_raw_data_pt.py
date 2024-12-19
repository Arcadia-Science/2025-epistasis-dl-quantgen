import os
import pickle as pk

out_dict={}

phen_file = open('arapheno/arapheno_scaled_study3_4_5_highNA.txt' , 'r')

phens = phen_file.read().split('\n')

phens = [x.split() for x in phens]

out_dict['phenotype_names'] = phens[0][1:] #extract header of pheno names from first row
#dict(list(out_dict.items())[2:3])


out_dict['strain_names'] = [x[0] for x in phens[1:-1]] #strain names extracted from first colun skipping one row

out_dict['phenotypes'] = [x[1:] for x in phens[1:-1]]

out_dict['phenotypes'] = [[float(y)  if y!= 'NA' else 0 for y in x[1:]] for x in phens[1:-1]] #convert pheno to float, dealing with NA



genotype_file = open('annotated_output_presence_absence_only.txt' , 'r')

gens = genotype_file.read().split('\n')

gens = [x.split() for x in gens]

out_dict['loci'] = [x[0] for x in gens[1:-1]]

#new_coding_dict = {0:[0,0,1],1:[1,0,0], 2:[0,1,0]}
new_coding_dict = {'0':[1,0,0],'1':[0,1,0], '2':[0,0,1]}

out_dict['genotypes'] = [[new_coding_dict[x] for x in [gens[y][n] for y in range(len(gens))[1:-1]]] for n in range(len(gens[0]))[1:]]


pk.dump(out_dict, open('arapheno/arapheno_scaled_study3_4_5_highNA_pheno.pk','wb'))