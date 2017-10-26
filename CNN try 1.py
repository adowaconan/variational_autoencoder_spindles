# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:25:47 2017

@author: ning
"""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from scipy import stats
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,Reshape
from keras.layers import Input
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import losses
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.convolutional import MaxPooling2D,MaxPooling1D
from keras.layers import UpSampling2D,UpSampling1D
from keras.utils import np_utils
from keras import backend as K

from sklearn.model_selection import train_test_split

working_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\'
def generate_(n=3):
    temp_ = {}
    for ii in range(n):
        temp_[ii] = np.load(working_dir+'data%d.npy'%(ii+1),)
    temp_list = [temp_[ii] for ii in range(n)]
    data = np.concatenate(temp_list)
    return data
data = generate_(6)
data = data.astype('float32')
data = data * 1e5 # rescale 

# some preprocessing the outlier values (extreme values)
Trimmed_data = stats.trimboth(data.flatten(),0.05)
idx_max = data > Trimmed_data.max()
idx_min = data < Trimmed_data.min()
data[idx_max] = Trimmed_data.max()
data[idx_min] = Trimmed_data.min()


# leave out test data
X_train,X_test = train_test_split(data,test_size=0.05)
# set up some hyper parameters
batch_size = 250
n_filters = 40
length_filters = 50
pool_size = 10
length_strides = 1
n_output = 25
file_path = working_dir+'weights.1D.best.hdf5'
checkPoint = ModelCheckpoint(file_path,monitor='val_loss',save_best_only=True,mode='min',period=1,verbose=1)
callback_list = [checkPoint]

layer_ = {}
model = Sequential()
model.add(Conv1D(n_filters,length_filters,strides=length_strides,padding='same',activation='relu',input_shape=(2000,61)))
model.add(MaxPooling1D(pool_size))
model.add(Conv1D(int(n_filters/2),int(length_filters/2),strides=length_strides,padding='same',activation='relu',))
model.add(MaxPooling1D(pool_size))
#layer_[1] = model.output_shape[1:]
#model.add(Flatten())
#layer_[0] = model.output_shape[1]
#model.add(Dense(n_output,activation='linear'))
#model.add(Dense(layer_[0],activation='linear'))
#model.add(Reshape(layer_[1]))

model.add(Conv1D(int(n_filters/4),int(length_filters/4),strides=length_strides,padding='same',activation='relu',))
model.add(UpSampling1D(pool_size))
model.add(Conv1D(n_filters,length_filters,strides=length_strides,padding='same',activation='relu',))
model.add(UpSampling1D(pool_size))
model.add(Conv1D(61,length_filters,strides=length_strides,padding='same',activation='relu'))


model.compile(optimizer='sgd',loss=losses.binary_crossentropy,metrics=['mae'])
model.summary()


model.fit(X_train,X_train,batch_size=batch_size,epochs=500,validation_split=0.2,callbacks=callback_list,verbose=0)

X_pred = model.predict(X_test)
idx = np.random.choice(np.arange(len(X_test)),size=1,)
fig,ax=plt.subplots(nrows=2)
ax[0].plot(X_test[idx].reshape(2000,61))
ax[1].plot(X_pred[idx].reshape(2000,61))
