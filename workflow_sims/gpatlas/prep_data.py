import os
import pickle as pk

out_dict={}

file_prefix = 'test_sim_WF_1kbt_10000n_5000000bp'
phen_file = open('alphasimr_output/' + file_prefix + '_p.txt' , 'r')

phens = phen_file.read().split('\n')

phens = [x.split() for x in phens]

out_dict['phenotype_names'] = phens[0][1:] #extract header of pheno names from first row
#dict(list(out_dict.items())[2:3])


out_dict['strain_names'] = [x[0] for x in phens[1:-1]] #strain names extracted from first colun skipping one row

out_dict['phenotypes'] = [x[1:] for x in phens[1:-1]]

out_dict['phenotypes'] = [[float(y)  if y!= 'NA' else 0 for y in x[1:]] for x in phens[1:-1]] #convert pheno to float, dealing with NA



genotype_file = open('alphasimr_output/' + file_prefix + '_g.txt' , 'r')

gens = genotype_file.read().split('\n')

gens = [x.split() for x in gens]

out_dict['loci'] = [x[0] for x in gens[1:-1]]

new_coding_dict = {'0':[1,0],'1':[0,1]}

out_dict['genotypes'] = [[new_coding_dict[x] for x in [gens[y][n] for y in range(len(gens))[1:-1]]] for n in range(len(gens[0]))[1:]]

#dump full dataset
pk.dump(out_dict, open('gpatlas/' + file_prefix + '.pk','wb'))


#################################################################################
#################################################################################
#split test train



in_data = pk.load(open('gpatlas/' + file_prefix + '.pk','rb'))

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

pk.dump(out_dict_train, open('gpatlas/' + file_prefix + '_train.pk','wb'))

del(out_dict_train)

#test set
for x in categories_to_copy:
 out_dict_test[x] = in_data[x]

for x in categories_to_stratefy:
 out_dict_test[x] = in_data[x][train_length:]

pk.dump(out_dict_test, open('gpatlas/' + file_prefix + '_test.pk','wb'))

del(out_dict_test)