# -*- coding: utf-8 -*-
"""
Created on Sun Mar	1 21:22:40 2020

@author: chumache
"""

import numpy as np
import scipy
import numpy.linalg as linalg
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.compat.v1.keras.backend import set_session
import gc
import tensorflow.keras.backend as K
import time
import h5py

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True	# dynamically grow the memory used on the GPU
config.log_device_placement = True	# to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)


def calculate_targets_singleview(class_lbls, n_clusters, clst_lbls):
	Label = np.unique(class_lbls)
	T = []
	N = len(class_lbls)
	nClasses = len(np.unique(class_lbls))
	nClusters = len(np.unique(clst_lbls))
	Y = np.random.rand(nClasses,nClasses-1); 
	Z = np.zeros([N,nClasses-1])
	num2class = {};
	for cl in range(nClasses):
		l = sum(class_lbls == cl)
		if l in num2class.keys():
			num2class[l].append(cl)
		else:
			num2class[l] = [cl]

	keys = list(num2class.keys())
	keys = np.sort(keys)

	for ii in range(nClasses):
		ind_i = np.where(class_lbls==Label[ii])	   
		Z[ind_i,:] = np.tile(Y[ii,:], (len(ind_i),1))
	
	T = Z

	for l in keys:
		classes = num2class[l]
		rep = len(classes)	 

		RandVals = np.random.rand(rep*nClusters, rep*(nClusters - 1))
		i = 0
		Tgen = np.zeros([N,rep*(nClusters - 1)])
		for c in classes:
			for clust in range(n_clusters):
				idxs = np.where((class_lbls == c) & (clst_lbls == clust))
				length = len(idxs)
				Tgen[idxs, :] = np.tile(RandVals[i,:], (length, 1))
				i = i+1
			
		
		T = np.concatenate([T,Tgen],-1)
	
		
	T = np.concatenate([np.ones((N,1)), T],-1)
	Y,_ = scipy.linalg.qr(T, mode='economic')
	#Y = Y[:,:nClusters*nClasses] #python version returns full square matrix, so limit to match matlab #using economic mode that does the same
	Y= Y[:,1:]
	T = np.transpose(Y)
	return T


def get_weights(X_train_sorted, y_train, n_clusters, clst_lbls,alpha, dim):
	X_train_sorted = np.transpose(X_train_sorted)
	T = calculate_targets_singleview(y_train, n_clusters, clst_lbls)
	if np.shape(X_train_sorted)[0] < np.shape(X_train_sorted)[1]:
		ss = np.dot(X_train_sorted,np.transpose(X_train_sorted))
		succ = 0
		while not succ:

			try:
				L = scipy.linalg.cholesky(ss)
				succ=1
			except:
				for i in range(np.shape(ss)[0]):
					ss[i,i] += alpha

		
		tt = np.linalg.pinv(L)
		W = tt.dot(tt.T).dot(X_train_sorted).dot(T.T)
	else:
		ss = np.dot(np.transpose(X_train_sorted),X_train_sorted)
		succ = 0

		while not succ:
			try:
				L = scipy.linalg.cholesky(ss)
				succ=1
			except:
				for i in range(np.shape(ss)[0]):
					ss[i,i] += alpha
				
		tt = np.linalg.pinv(L)
		W = X_train_sorted.dot(tt.T.dot(tt)).dot(T.T)

	tmpNorm = np.sqrt(np.diag(np.dot(W.T,W)))
	W = np.divide(W, np.tile(tmpNorm.T,(np.shape(W)[0],1)))
	return W
	

	
def extract_patches(inputs_nopad, filtersize):
	variance_epsilon = 1e-5
	N = inputs_nopad.shape[0]
	H = inputs_nopad.shape[1]
	remain = tf.math.floormod(H,filtersize);
	C = inputs_nopad.shape[-1]	
	right_side = tf.math.floordiv(remain,2)
	leftover = tf.math.floormod(remain,2)
	left_side = tf.add(right_side,leftover)

	pad_left = tf.zeros((N,left_side,H,C))
	pad_right = tf.zeros((N,right_side, H,C))
	pad_top = tf.zeros((N,tf.add(tf.add(H,left_side),right_side), left_side, C))
	pad_bottom = tf.zeros((N, tf.add(tf.add(H,left_side),right_side), right_side, C))

	inputs = tf.concat([pad_left, inputs_nopad], axis=1)
	del inputs_nopad
	inputs = tf.concat([inputs, pad_right], axis=1)
	inputs = tf.concat([pad_top, inputs], axis=2)
	inputs = tf.concat([inputs, pad_bottom], axis=2)

	patches = tf.image.extract_patches(inputs, [1,filtersize,filtersize,1], strides=[1,filtersize,filtersize,1], rates = [1,1,1,1], padding='VALID')
		
	patches = tf.reshape(patches, [-1,(filtersize**2)*C])
	mean, variance = tf.nn.moments(patches,0)
	inv = math_ops.rsqrt(variance + variance_epsilon)
	H_pad = inputs.shape[1]
	del inputs
	dilated_h = tf.math.multiply(filtersize,tf.math.floordiv(H_pad,filtersize))
	patches = tf.reshape(patches, [-1,tf.math.multiply(tf.math.square(filtersize),C)])
		  
	patches_standardized = tf.subtract(tf.math.multiply(patches,inv),tf.math.multiply(mean,inv))
	del patches
	axes_1_2_size = tf.cast(tf.sqrt(tf.math.divide(tf.math.square(dilated_h),tf.math.square(filtersize))), tf.int32)
	reconstruct = tf.reshape(patches_standardized, (N, axes_1_2_size, axes_1_2_size, filtersize, filtersize, C)) 
	reconstruct = tf.transpose(reconstruct, (0, 1, 3, 2, 4, 5))
	# Reshape back
	reconstruct = tf.reshape(reconstruct, (N, dilated_h, dilated_h, C))
	del N, H, remain, C, right_side, leftover, left_side, pad_left, pad_right, pad_top, pad_bottom, mean, variance, inv, H_pad, dilated_h, axes_1_2_size
	gc.collect()
	return reconstruct, patches_standardized
	
