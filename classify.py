#!/usr/bin/python

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import sys
import os
caffe_root = '../../../caffe_python/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()


# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

model_def = 'deploy.prototxt'
model_weights= 'snapshot_iter_75000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

data_total = 0
income_correct = 0
poverty_correct = 0
crime_correct = 0

income_table = [[0 for x in range(7)] for y in range(7)] 
poverty_table = [[0 for x in range(7)] for y in range(7)]
crime_table = [[0 for x in range(7)] for y in range(7)]

file = open('classify.txt', 'r')
count = 0
for line in file:
	line = line.rstrip('\n')
	words = line.split()	
	image_link = words[0]
	income_truth_cat = int(words[1])
	poverty_truth_cat = int(words[2])
	crime_truth_cat = int(words[3])
	
	#print image_link, income_truth, poverty_truth, crime_truth
	
	image = caffe.io.load_image('../images/' + image_link)
	transformed_image = transformer.preprocess('data', image)
	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image
	### perform classification
	output = net.forward()

	#print output['final_crime']
	#print np.argmax(output['final_crime'][0])
	#print

	income_cat = np.argmax(output['final_income'][0])
	poverty_cat = np.argmax(output['final_poverty'][0])
	crime_cat = np.argmax(output['final_crime'][0])	

	#print 'Income:', income, ', truth:', income_truth
	#print 'Poverty:', poverty, ', truth:', poverty_truth


	count = count + 1
	#if count == 10:
	#	break
	#continue

	data_total = data_total + 1

	if income_cat == income_truth_cat:
		income_correct = income_correct + 1	
	income_table[income_truth_cat][income_cat] = income_table[income_truth_cat][income_cat] + 1

	if poverty_cat == poverty_truth_cat:
		poverty_correct = poverty_correct + 1	
	poverty_table[poverty_truth_cat][poverty_cat] = poverty_table[poverty_truth_cat][poverty_cat] + 1


	if crime_cat == crime_truth_cat:
		crime_correct = crime_correct + 1	
	crime_table[crime_truth_cat][crime_cat] = crime_table[crime_truth_cat][crime_cat] + 1

	
	if count % 200 == 0:		
		print 'Income'
		print 'Total:', data_total, ', Correct:', income_correct, ', incorrect:', (data_total-income_correct), 
		print ', accuracy: ', str(float(income_correct*100.0/data_total))+'%'
		print 'Poverty'
		print 'Total:', data_total, ', Correct:', poverty_correct, ', incorrect:', (data_total-poverty_correct),
		print ', accuracy: ', str(float(poverty_correct*100.0/data_total))+'%'
		print 'Crime'
		print 'Total:', data_total, ', Correct:', crime_correct, ', incorrect:', (data_total-crime_correct),
		print ', accuracy: ', str(float(crime_correct*100.0/data_total))+'%'
		#break
		

print 'Income'
print 'Total:', data_total, ', Correct:', income_correct, ', incorrect:', (data_total-income_correct), 
print ', accuracy: ', str(float(income_correct*100.0/data_total))+'%'
print 'Poverty'
print 'Total:', data_total, ', Correct:', poverty_correct, ', incorrect:', (data_total-poverty_correct),
print ', accuracy: ', str(float(poverty_correct*100.0/data_total))+'%'
print 'Crime'
print 'Total:', data_total, ', Correct:', crime_correct, ', incorrect:', (data_total-crime_correct),
print ', accuracy: ', str(float(crime_correct*100.0/data_total))+'%'

print 'Income Confusion Table'
print '     ',
for i in range(0,7):
	print '  (' + str(i+1) + ')',
print
for i in range(0,7):
	print '['+str(i+1)+']', '',
	for j in range(0,7):
		print '%5d' % income_table[i][j],
	print
print

print 'Poverty Confusion Table'
print '     ',
for i in range(0,7):
	print '  (' + str(i+1) + ')',
print
for i in range(0,7):
	print '['+str(i+1)+']', '',
	for j in range(0,7):
		print '%5d' % poverty_table[i][j],
	print
print

print 'Crime Confusion Table'
print '     ',
for i in range(0,7):
	print '  (' + str(i+1) + ')',
print
for i in range(0,7):
	print '['+str(i+1)+']', '',
	for j in range(0,7):
		print '%5d' % crime_table[i][j],
	print
print

file.close()
