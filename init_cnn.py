import numpy as np
import sys
from utils_h5py import get_weights
from tensorflow.keras.utils import to_categorical 
import mat73
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, LeakyReLU, Conv2D, MaxPool2D, Dense, Activation, Flatten, BatchNormalization, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.compat.v1.keras.backend import set_session
import gc
import os
import pickle
from tensorflow.keras import activations
from fasterVectorBatchNorm import CustomBatchNormalization
import random

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True	# dynamically grow the memory used on the GPU
config.log_device_placement = True	# to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

partial = 500 #how many samples from each class to use. select None to use all samples
activation_functions = ['leakyrelu'] #'relu', 'tanh'
meanX = 1 #perform mean centering or not
num_clusters = 16 #how many clusters to start with. n_clusters*n_classes - 1 should correspond to the number of filters in the first layer. The number of clusters is reduced by 2 times at each new layer.
dataset_name = 'cifar' #'cifar'


filtersize=5
batch_size = 256
	
if dataset_name == 'mnist':
	channels = 1
	datafile = './mnist.pkl'
	data_size = (28,28,1)
elif dataset_name == 'cifar':
	channels = 3	
	datafile = './cifar.pkl'
	data_size = (32,32,3)

def extract(image):
	image = tf.image.extract_patches(image, [1,filtersize,filtersize,1], strides=[1,filtersize,filtersize,1], rates = [1,1,1,1], padding='VALID')			 
	return image
	

for activation_function in activation_functions:
	with open(datafile, 'rb') as f:
		dataset = pickle.load(f)
	
	y_train = np.asarray(dataset['training_labels'])
	X_train_temp = dataset['training_images'].astype('float32')
	X_train = np.reshape(X_train_temp, (np.shape(X_train_temp)[0], data_size[0], data_size[1], data_size[2]))

	del dataset	
	del X_train_temp
	
	
	X_train[:,:,:,0] = (X_train[:,:,:,0] - np.mean(np.mean(np.mean(X_train[:,:,:,0]))))	
	if dataset_name == 'cifar':
		X_train[:,:,:,1] = (X_train[:,:,:,1] - np.mean(X_train[:,:,:,1]))	  
		X_train[:,:,:,2] = (X_train[:,:,:,2] - np.mean(X_train[:,:,:,2]))	  
		
	X_train = X_train / 255	

	C = len(np.unique(y_train))
	
	if partial is None:
		idxs = []
		for i in range(C):
			idxs += list(np.where(y_train==i)[0])
		y_train = y_train[idxs]
		images = X_train[idxs,:,:,:]
	else:	
		images = np.expand_dims(X_train[0,:], axis=0)
		labels = []
		for i in range(C):
			idxs = np.where(y_train==i)[0]
			random.shuffle(idxs)
			idxs = idxs[:partial]
			labels += list(y_train[idxs])
			images = np.concatenate([images, X_train[idxs,:,:,:]])
		y_train = np.asarray(labels)
		images = images[1:,:,:,:]

	del X_train

	print('starting clustering')
	start_time = time.time()
	weights = {}
	act = activation_function

	layer = 0
	n_clusters = num_clusters

	while n_clusters > 1:
		
		
		
		input = Input(shape=(np.shape(images)[1], np.shape(images)[2], np.shape(images)[3]))
		output1 = CustomBatchNormalization(filtersize=filtersize)(input, training=True)
		output2 = Lambda(extract)(output1)
		model = Model(inputs=input, outputs=output2)			
		
		patch_data = model.predict(images, batch_size=batch_size)
		channels = np.shape(images)[-1]
		
		patch_data = np.reshape(patch_data, [-1,channels*(filtersize**2)])


		n_patches_per_im = int(np.shape(patch_data)[0] / np.shape(images)[0])
		patch_labels = np.repeat(y_train, n_patches_per_im);
		
		
		final_clusters_prev = [] 
		patch_labels = np.asarray(patch_labels)
		
		resorted_indices = []
		start=0
		finish=0
		
		for cl in range(C):
			indices = np.where(patch_labels == cl)[0]
			X_curr_class = patch_data[indices,:]
			
			kmeans = MiniBatchKMeans(n_clusters, batch_size=batch_size)
			kmeans.fit(X_curr_class)
			clst_lbls = kmeans.labels_
			
			for i in range(n_clusters):
				temp =	X_curr_class[clst_lbls==i,:]
				finish = start + np.shape(temp)[0]
				start = start + np.shape(temp)[0] 
				final_clusters_prev = np.concatenate([final_clusters_prev, i*np.ones(sum(clst_lbls == i))])
				resorted_indices = np.concatenate([resorted_indices, indices[clst_lbls==i]])	

		resorted_indices = resorted_indices.astype(int)
		patch_data = patch_data[resorted_indices,:]

		clst_class_lbls1 = np.tile(1, (1, C))
		start_h = time.time()
		W1 = get_weights(patch_data, patch_labels, n_clusters, final_clusters_prev, 0.001, C*n_clusters-1)	
		
		weights[layer] = W1
		time_hh = time.time()
		
		filters = np.zeros(( filtersize,filtersize,channels,C*n_clusters-1))
		for i in range(C*n_clusters-2):
			filters[:,:,:,i] = np.reshape(W1[:,i], (filtersize,filtersize,channels))
			
		input = Input(shape=(np.shape(images)[1],np.shape(images)[2],np.shape(images)[3]))
		x00_bn = CustomBatchNormalization(filtersize=filtersize)
		x00 = Conv2D(C*n_clusters-1, (filtersize, filtersize), padding = 'same', use_bias=False, weights=[filters])
		x00_pool = MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')
		if act == 'tanh':
			x00_act = Activation(activations.tanh)
		elif act == 'relu':
			x00_act = Activation(activations.relu)
		elif act == 'leakyrelu':
			x00_act = LeakyReLU(alpha=0.3)
					
		x = x00_bn(input, training=True)
		x = x00(x)
		x = x00_pool(x)
		output = x00_act(x)	
		n_its = int(np.floor(np.shape(images)[0]/batch_size))
		K.clear_session()
		model = Model(inputs=input, outputs=output)
		K.clear_session()
		images = model.predict(images, batch_size=batch_size)	

		del model
		del input
		K.clear_session()
		gc.collect()
		tf.compat.v1.reset_default_graph()
		
		n_clusters = int(n_clusters/2)
		layer+=1

	input = Input(shape=(np.shape(images)[1],np.shape(images)[2],np.shape(images)[3]))
				
	
	output = Flatten()(input)
	model = Model(inputs=input, outputs=output)
	K.clear_session()
	patch_data = model.predict(images)
	
	if meanX:
		patch_data = (patch_data - np.mean(patch_data,axis=0))/(np.std(patch_data,axis=0) + 1e-5)
	
	del images
	n_clusters=13	
	final_clusters_prev = []
	sortedX = np.zeros(np.shape(patch_data))
	start=0
	finish=0
	for cl in range(C):
		X_curr_class = patch_data[y_train == cl, :]
		
		kmeans = MiniBatchKMeans(n_clusters, batch_size=1500).fit(X_curr_class)
		clst_lbls = kmeans.labels_
		
	
		for i in range(n_clusters):
			
			temp =	X_curr_class[clst_lbls==i,:]
			finish = start + np.shape(temp)[0]
			sortedX[start:finish,:] = temp
			start = start + np.shape(temp)[0] 
			
			final_clusters_prev = np.concatenate([final_clusters_prev, i*np.ones(sum(clst_lbls == i))])

	del patch_data

	clst_class_lbls1 = np.tile(1, (1, C))
	start_h = time.time()
	W1 = get_weights(sortedX, y_train, n_clusters, final_clusters_prev, 0.001, 128)	
	W1 = W1[:,:128]
		
	weights[layer] = W1

	
	times_all = time.time() - start_time

	with open('{}_{}_{}_{}_cnn.pkl'.format(dataset_name, num_clusters, str(partial), activation_function), 'wb') as f:
		pickle.dump(weights, f)





