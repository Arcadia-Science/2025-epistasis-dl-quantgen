#!/usr/bin/env python3

import pickle as pk
import sys

#in_file = sys.argv[1]

#in_data = pk.load(open(in_file,'rb'))


in_data = pk.load(open('arapheno/arapheno_scaled_study3_4_5_lowNA_pheno.pk','rb'))


out_dict_test = {}

out_dict_train = {}

categories_to_stratefy = ['phenotypes', 'strain_names']

categories_to_copy = [x for x in in_data.keys() if x not in categories_to_stratefy]

train_length = round(len(in_data['strain_names'])*0.8)

#train set
for x in categories_to_copy:
 out_dict_train[x] = in_data[x]

for x in categories_to_stratefy:
 out_dict_train[x] = in_data[x][:train_length]

pk.dump(out_dict_train, open('arapheno/arapheno_scaled_study3_4_5_lowNA_pheno_train.pk','wb'))

del(out_dict_train)

#test set
for x in categories_to_copy:
 out_dict_test[x] = in_data[x]

for x in categories_to_stratefy:
 out_dict_test[x] = in_data[x][train_length:]

pk.dump(out_dict_test, open('arapheno/arapheno_scaled_study3_4_5_lowNA_pheno_test.pk','wb'))

del(out_dict_test)