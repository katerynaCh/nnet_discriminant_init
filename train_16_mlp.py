
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, LeakyReLU
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

#import os
# Read data
import mat73

weight_files = ['./cifar_16_None_relu_mlp.pkl']
dataset = 'cifar'
batchnorm = 1

if dataset == 'cifar':
	datapath = './cifar.pkl'
elif dataset == 'mnist':
	datapath = './mnist.pkl'	
	
for file in weight_files:
	act = file.split('_')[-2]  
	save_dir_zerobias = file.split('.pkl')[0].split('/')[-1] + '_zerobias' + str(batchnorm) + '/'

	if not os.path.exists(save_dir_zerobias):
		os.mkdir(save_dir_zerobias)

	with open(file, 'rb') as f:
		weights = pickle.load(f)
	
	with open(datapath, 'rb') as f:
		dataset = pickle.load(f)

	
	y_train = dataset['training_labels']
	X_train = dataset['training_images'].astype('float32')

	y_val = dataset['val_labels']
	X_val = dataset['val_images'].astype('float32')

	X_test = dataset['test_images'].astype('float32')
	y_test = dataset['test_labels'].copy()
	
	accuracies_zerobias = []
	times_zerobias = []
	
	if dataset == 'mnist':
		X_val = (X_val - np.mean(X_train,axis=0)) 
		X_test = (X_test - np.mean(X_train, axis=0)) 
		X_train = (X_train - np.mean(X_train, axis=0)) 
	elif dataset == 'cifar':
		X_val = (X_val - np.mean(X_train,axis=0)) /(np.std(X_train, axis=0) + 1e-5)
		X_test = (X_test - np.mean(X_train, axis=0)) /(np.std(X_train, axis=0) + 1e-5)
		X_train = (X_train - np.mean(X_train, axis=0)) /(np.std(X_train, axis=0) + 1e-5)
   
	W00 = weights[0]
	W0 = weights[1]
	W1 = weights[2]
	W2 = weights[3]
	
	input_dim = X_train.shape[1]
	
	y_train = to_categorical(y_train) 
	y_val = to_categorical(y_val) 
	nb_classes = y_train.shape[1]
	
	
	input = Input(shape=(input_dim,))
	x00 = Dense(159, weights=[W00,np.zeros(159)])#(input)
	if act == 'tanh':
		x00_act = Activation(activations.tanh)
	elif act == 'relu':
		x00_act = Activation(activations.relu)
	elif act == 'shiftedrelu':
		x00_act = Activation(activations.relu)
	elif act == 'leakyrelu':
		x00_act = LeakyReLU(alpha=0.3)
	x0 = Dense(79, weights=[W0,np.zeros(79)])#(input)
	if act == 'tanh':
		x0_act = Activation(activations.tanh)
	elif act == 'relu':
		x0_act = Activation(activations.relu)
	elif act == 'shiftedrelu':
		x0_act = Activation(activations.relu)
	elif act == 'leakyrelu':
		x0_act = LeakyReLU(alpha=0.3)
	x1 = Dense(39, weights=[W1,np.zeros(39)])
	if act == 'tanh':
		x1_act = Activation(activations.tanh)
	elif act == 'relu':
		x1_act = Activation(activations.relu)
	elif act == 'shiftedrelu':
		x1_act = Activation(activations.relu)
	elif act == 'leakyrelu':
		x1_act = LeakyReLU(alpha=0.3)
	x2 = Dense(19, weights=[W2,np.zeros(19)])
	if act == 'tanh':
		x2_act = Activation(activations.tanh)
	elif act == 'relu':
		x2_act = Activation(activations.relu)
	elif act == 'shiftedrelu':
		x2_act = Activation(activations.relu)
	elif act == 'leakyrelu':
		x2_act = LeakyReLU(alpha=0.3)
	
	x3 = Dense(10, activation='softmax', kernel_initializer=RandomNormal(mean=0., stddev=0.05)) 
		
	x = x00(input)
	x = x00_act(x)	
	if batchnorm == 1: 
		x = BatchNormalization()(x)
	x = x0(x)
	
	x = x0_act(x)	
	if batchnorm == 1: 
		x = BatchNormalization()(x)
	x = x1(x)
	
	x = x1_act(x)	
	if batchnorm == 1: 
		x = BatchNormalization()(x)
	x = x2(x)
	
	x = x2_act(x)	

	
	output = x3(x)
	
		
	model_our = Model(inputs=input, outputs=output)
	model_our.summary()
	sgd = SGD(lr=0.001)
	
	model_our.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
	
	callback = EarlyStopping(monitor='val_acc', patience=10,restore_best_weights=True)
	filepath = save_dir_zerobias + "f"+str('_')+"-our-zerobias-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=12)	
	start_zerobias = time.process_time()			
	historyour = model_our.fit(X_train, y_train, epochs=300, batch_size=32, validation_data = (X_val, y_val), verbose=1, callbacks = [callback, checkpoint])
	times_zerobias.append(time.process_time() - start_zerobias)
	model_our.save(save_dir_zerobias + 'model_zerobias_'+str('_')+'.h5')
	with open(save_dir_zerobias + 'zerobias_f'+str('_'), 'wb') as file_pi:
		pickle.dump(historyour.history, file_pi)
	preds = model_our.predict(X_test, verbose=0)
	preds = np.argmax(preds,axis=1)
	acc = accuracy_score(y_test, preds)
	print('_', acc)
	accuracies_zerobias.append(acc)
	
	plt.plot(historyour.history['acc'])
	plt.plot(historyour.history['val_acc'])
	plt.title('Model accuracy' + str('_'))
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	
	plt.savefig(save_dir_zerobias+str('_') + '_our')
	plt.show()
	plt.clf()
	
	
	with open(os.path.join(save_dir_zerobias,'result.txt'), 'a') as f:

		f.write('Mean accuracy, zerobias: '+ str(np.mean(accuracies_zerobias)) + '\n')
		f.write('Std accuracy, zerobias: '+ str(np.std(accuracies_zerobias)) + '\n')
		
		f.write('Mean time, zerobias: '+ str(np.mean(times_zerobias)) + '\n')
		
			
		
