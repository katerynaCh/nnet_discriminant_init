import numpy as np
import sys
from utils_h5py import get_weights
import time
from sklearn.cluster import MiniBatchKMeans
import tensorflow as tf
from tensorflow.keras.layers import Input, LeakyReLU, Dense, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.compat.v1.keras.backend import set_session
import gc
import os
import pickle
from tensorflow.keras import activations
import random

partial = None #how many samples from each class to use. select None to use all samples
activation_functions = ['relu'] #'relu', 'tanh'
meanX = 1 #perform mean centering or not
num_clusters = 16 #how many clusters to start with. n_clusters*n_classes - 1 should correspond to the number of filters in the first layer. The number of clusters is reduced by 2 times at each new layer.
dataset_name = 'cifar'

if dataset_name == 'mnist':
	channels = 1
	datafile = './mnist.pkl'
elif dataset_name == 'cifar':
	channels = 3	
	datafile = './cifar.pkl'
	
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True	# dynamically grow the memory used on the GPU
config.log_device_placement = True	# to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

for activation_function in activation_functions:
	n_clusters = num_clusters
	with open(datafile, 'rb') as f:
		dataset = pickle.load(f)

	y_train = np.asarray(dataset['training_labels'])
	X_train = dataset['training_images'].astype('float32')
	
	if dataset == 'mnist':
		X_train = (X_train - np.mean(X_train, axis=0)) 
	elif dataset == 'cifar':
		X_train = (X_train - np.mean(X_train, axis=0)) /(np.std(X_train, axis=0) + 1e-5)
		
	del dataset

	C = len(np.unique(y_train))

	if partial is None:
		idxs = []
		for i in range(C):
			idxs += list(np.where(y_train==i)[0])
		patch_labels = y_train[idxs]
		patch_data = X_train[idxs,:]
	else:	
		patch_data = np.expand_dims(X_train[0,:], axis=0)
		labels = []
		for i in range(C):
			idxs = np.where(y_train==i)[0]
			random.shuffle(idxs)
			idxs = idxs[:partial]
			labels += list(y_train[idxs])
			patch_data = np.concatenate([patch_data, X_train[idxs,:]])
	
		patch_labels = np.asarray(labels)
		patch_data = np.asarray(patch_data[1:,:])
	
	del X_train
	print('starting clustering')
	start_time = time.time()
	weights = {}

	act = activation_function
	layer = 0
	time_fold_start = time.time()
	while n_clusters > 1:
		if meanX:	
			patch_data = (patch_data - np.mean(patch_data,0)) /(np.std(patch_data, axis=0) + 1e-5)
		
		sortedX = np.zeros(np.shape(patch_data))	   
		#step 1. cluster
		final_clusters_prev = []
		start=0
		finish=0
		for cl in range(C):
			X_curr_class = patch_data[patch_labels == cl, :]
			
			kmeans = MiniBatchKMeans(n_clusters, batch_size=2048).fit(X_curr_class)
			clst_lbls = kmeans.labels_
	
			for i in range(n_clusters):
				temp =	X_curr_class[clst_lbls==i,:]
				finish = start + np.shape(temp)[0]
				sortedX[start:finish,:] = temp
				start = start + np.shape(temp)[0] 
				final_clusters_prev = np.concatenate([final_clusters_prev, i*np.ones(sum(clst_lbls == i))])

		del patch_data
		clst_class_lbls1 = np.tile(1, (1, C))
		W1 = get_weights(sortedX, patch_labels, n_clusters, final_clusters_prev, 0.001, C*n_clusters-1)	
		weights[layer] = W1
		sortedX = np.dot(W1.T, sortedX.T).T		 
		#encapsulate in keras model for efficient and convenient batch-wise processing
		gc.collect()
		input = Input(shape=np.shape(sortedX))
		if act == 'tanh':
			x00_act = Activation(activations.tanh)
		elif act == 'relu':
			x00_act = Activation(activations.relu)
		elif act == 'leakyrelu':
			x00_act = LeakyReLU(alpha=0.3)
		output = x00_act(input)		
		model = Model(inputs=input, outputs=output)
		sortedX = np.expand_dims(sortedX, axis = 0)
		patch_data = model.predict(sortedX)	
		patch_data = np.squeeze(patch_data)
		del model
		del input
		n_clusters = int(n_clusters/2)
		layer+=1

	
	times = time.time() - start_time
	print('Time taken: ', times)
	with open('{}_{}_{}_{}_mlp.pkl'.format(dataset_name, num_clusters, str(partial), activation_function), 'wb') as f:
		pickle.dump(weights, f)

