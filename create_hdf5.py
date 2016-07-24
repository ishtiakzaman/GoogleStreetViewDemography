#!/usr/bin/python

import h5py
import numpy
import os

rows = 43100
cols = 1

file = h5py.File('train_db.h5', 'w')

dataset_income = file.create_dataset('label_income',(rows, cols))
dataset_poverty = file.create_dataset('label_poverty',(rows, cols))
dataset_crime = file.create_dataset('label_crime',(rows, cols))

data_income = numpy.zeros((rows, cols))
data_poverty = numpy.zeros((rows, cols))
data_crime = numpy.zeros((rows, cols))
f = open('train_value.txt', 'r')
i = 0
for lines in f:
	v = lines.split()
	
	data_income[i][0] = v[0]	
	data_poverty[i][0] = v[1]
	data_crime[i][0] = v[2]

	i = i + 1
dataset_income[...] = data_income
dataset_poverty[...] = data_poverty
dataset_crime[...] = data_crime

file.close ()

os.system('echo "train_db.h5" > train_db.txt')
