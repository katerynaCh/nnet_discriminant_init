
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:33:12 2020

@author: chumache
"""



seed_value= 42
#1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}

from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.layers.core import Dense, Activation, Flatten
from tensorflow.keras.layers import Input, Dropout, LeakyReLU, Conv2D, MaxPool2D, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import activations
import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold 
import pickle
from tensorflow.keras.initializers import RandomNormal, RandomUniform
import time
from fasterVectorBatchNorm import CustomBatchNormalization
import mat73


weight_files = ['./mnist_16_500_leakyrelu_cnn.pkl']
dataset = 'mnist'
batchnorm = 1
filtersize=5

if dataset == 'cifar':
	datapath = './cifar.pkl'
	data_shape = (32,32,3)
elif dataset == 'mnist':
	datapath = './mnist.pkl'	
	data_shape = (28,28,1)
channels = data_shape[-1]
	
for file in weight_files:
	act = file.split('_')[-2]

	save_dir_zerobias = file.split('.pkl')[0].split('/')[-1]+ '_zerobias_' + str(batchnorm) + '/'
	if not os.path.exists(save_dir_zerobias):
		os.mkdir(save_dir_zerobias)
	
	with open(file, 'rb') as f:
		weights = pickle.load(f)
	
	with open(datapath, 'rb') as f:
		dataset = pickle.load(f)
					
	
	accuracies_zerobias = []
	times_zerobias = []
	
	
	y_train = dataset['training_labels']
	X_train = dataset['training_images'].astype('float32')

	y_val = dataset['val_labels']
	X_val = dataset['val_images'].astype('float32')

	X_test = dataset['test_images'].astype('float32')
	y_test = dataset['test_labels'].copy()
		
	X_train = np.reshape(X_train, (np.shape(X_train)[0], data_shape[0], data_shape[1], data_shape[2]))
	X_val = np.reshape(X_val, (np.shape(X_val)[0], data_shape[0], data_shape[1], data_shape[2]))
	X_test = np.reshape(X_test, (np.shape(X_test)[0], data_shape[0], data_shape[1], data_shape[2]))

	for ch in range(channels):
		X_val[:,:,:,ch] = (X_val[:,:,:,ch] - np.mean(np.mean(np.mean(X_train[:,:,:,ch])))) 
		X_test[:,:,:,ch] = (X_test[:,:,:,ch] - np.mean(np.mean(np.mean(X_train[:,:,:,ch]))))
		X_train[:,:,:,ch] = (X_train[:,:,:,ch] - np.mean(np.mean(np.mean(X_train[:,:,:,ch]))))	 
		
	X_train = X_train/255
	X_val = X_val/255
	X_test = X_test/255

	weights_m = []
	n_clusters=16

	C=10

	for kk in weights:
		W1 = weights[kk]
		if kk == 4:
			weights_m.append(W1)
			continue

		filters = np.zeros(( filtersize,filtersize,channels,C*n_clusters-1))
		for i in range(C*n_clusters-2):
			filters[:,:,:,i] = np.reshape(W1[:,i], (filtersize,filtersize,channels))
		weights_m.append(filters)
		channels = np.shape(W1)[1]
		n_clusters = int(n_clusters/2)
	
	W00 = weights_m[0]
	W0 = weights_m[1]
	W1 = weights_m[2]
	W2 = weights_m[3]
	W3 = weights_m[4]
	
	input_dim = X_train.shape
	
	y_train = to_categorical(y_train) 
	y_val = to_categorical(y_val) 
	nb_classes = y_train.shape[1]
	
	
	input = Input(shape=data_shape)		
	x00 = Conv2D(159, (5,5), padding = 'same', use_bias=False, weights=[W00])
	x00_pool = MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')
	if act == 'tanh':
		x00_act = Activation(activations.tanh)
	elif act == 'relu':
		x00_act = Activation(activations.relu)
	elif act == 'shiftedrelu':
		x00_act = Activation(activations.relu)
	elif act == 'leakyrelu':
		x00_act = LeakyReLU(alpha=0.3)
	x0 = Conv2D(79, (5,5), padding = 'same', use_bias=False, weights=[W0])
	x0_pool = MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')
	if act == 'tanh':
		x0_act = Activation(activations.tanh)
	elif act == 'relu':
		x0_act = Activation(activations.relu)
	elif act == 'shiftedrelu':
		x0_act = Activation(activations.relu)
	elif act == 'leakyrelu':
		x0_act = LeakyReLU(alpha=0.3)
	x1 = Conv2D(39, (5,5), padding = 'same', use_bias=False, weights=[W1])
	x1_pool = MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')
	if act == 'tanh':
		x1_act = Activation(activations.tanh)
	elif act == 'relu':
		x1_act = Activation(activations.relu)
	elif act == 'shiftedrelu':
		x1_act = Activation(activations.relu)
	elif act == 'leakyrelu':
		x1_act = LeakyReLU(alpha=0.3)
	x2 = Conv2D(19, (5,5), padding = 'same', use_bias=False, weights=[W2])
	x2_pool = MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')
	if act == 'tanh':
		x2_act = Activation(activations.tanh)
	elif act == 'relu':
		x2_act = Activation(activations.relu)
	elif act == 'shiftedrelu':
		x2_act = Activation(activations.relu)
	elif act == 'leakyrelu':
		x2_act = LeakyReLU(alpha=0.3)
		
	x3 = Dense(128, weights = [W3, np.zeros(128)])
	if act == 'tanh':
		x3_act = Activation(activations.tanh)
	elif act == 'relu':
		x3_act = Activation(activations.relu)
	elif act == 'shiftedrelu':
		x3_act = Activation(activations.relu)
	elif act == 'leakyrelu':
		x3_act = LeakyReLU(alpha=0.3)
	x4 = Dense(10, kernel_initializer=RandomNormal(mean=0., stddev=0.05), activation='softmax') #weights = [W4, np.zeros(10)], activation='softmax')
	

	if batchnorm == 1: 
		x = CustomBatchNormalization(filtersize=filtersize)(input)
		x = x00(x)
	else:
		x = x00(input)
	x = x00_pool(x)
	x = x00_act(x)	
   
	if batchnorm == 1: 
		x = CustomBatchNormalization(filtersize=filtersize)(x)
	x = x0(x)
	x = x0_pool(x)
	x = x0_act(x)	
	if batchnorm == 1: 
		x = CustomBatchNormalization(filtersize=filtersize)(x)
	x = x1(x)
	x = x1_pool(x)
	x = x1_act(x)	
	if batchnorm == 1: 
		x = CustomBatchNormalization(filtersize=filtersize)(x)
	x = x2(x)
	x = x2_pool(x)
	x = x2_act(x)	
	x = Flatten()(x)
	if batchnorm == 1:
		x = BatchNormalization()(x)
	x = x3(x)
	x = x3_act(x)
	output = x4(x)
		
	model_our = Model(inputs=input, outputs=output)
	model_our.summary()
	sgd = SGD(lr=0.001)
	
	model_our.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	
	callback = EarlyStopping(monitor='val_accuracy', patience=10,restore_best_weights=True)
	filepath = save_dir_zerobias + "f"+str('_')+"-our-zerobias-{epoch:02d}-{val_accuracy:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=12)	
	start_zerobias = time.process_time()			
	historyour = model_our.fit(X_train, y_train, epochs=300, batch_size=32, validation_data = (X_val, y_val), verbose=1, callbacks = [callback, checkpoint])
	times_zerobias.append(time.process_time() - start_zerobias)
	model_our.save(save_dir_zerobias + 'model_zerobias_'+str('_')+'.h5')
	with open(save_dir_zerobias + 'zerobias_f'+str('_'), 'wb') as file_pi:
		pickle.dump(historyour.history, file_pi)
	preds = model_our.predict(X_test, verbose=0)
	preds = np.argmax(preds,axis=1)
	acc = accuracy_score(y_test, preds)
	accuracies_zerobias.append(acc)
	
	plt.plot(historyour.history['accuracy'])
	plt.plot(historyour.history['val_accuracy'])
	plt.title('Model accuracy' + str('_'))
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	
	plt.savefig(save_dir_zerobias+str('_') + '_our')
	plt.show()
	plt.clf()
	with open(save_dir_zerobias+str('_')+'acc.txt', 'a') as f:
		f.write(str(acc))	
		
	with open(save_dir + 'result.txt', 'a') as f:
		
		f.write('Mean accuracy, zerobias: '+ str(np.mean(accuracies_zerobias)) + '\n')
		f.write('Std accuracy, zerobias: '+ str(np.std(accuracies_zerobias)) + '\n')
		f.write('Mean time, zerobias: '+ str(np.mean(times_zerobias)) + '\n')
		
