#!/usr/bin/python

import sys

file = open(sys.argv[1], 'r')
count = 0
max = 0.0
min = 999999.9

for line in file:
	line = line.rstrip('\n')
	val = float(line)
	if val > max:
		max = val
	if val < min:
		min = val
		

file.seek(0,0)

for line in file:
	line = line.rstrip('\n')		
	val = (float(line)-min)/(max-min)
	val = val * 10.0
	print("%.2f" % val)

#print min # 1.0
#print max # 1032.0

