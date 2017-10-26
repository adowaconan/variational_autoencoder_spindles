# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:06:17 2017

@author: ning
"""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras

from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,Reshape
from keras.layers import Input,Permute
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
data = generate_(2)
data = data.astype('float32')
data = data * 1e6 # rescale to micro voltage
temp_ = []
for k in data:
    k = k.T
    temp_.append(k)
data = np.array(temp_)

# leave out test data
X_train,X_test = train_test_split(data)
# set up some hyper parameters
batch_size = 100
n_filters = {1:40,2:20,3:10}
length_filters = {1:(40,1),2:(1,20),3:(1,10)}
pool_size=(1,16)
length_strides = {1:(1,1),2:(1,1),3:(1,16),4:(1,1),5:(1,16)}
n_output = 25
input_shape = (61,2000)
file_path = 'weights.22D.best.hdf5'
checkPoint = ModelCheckpoint(file_path,monitor='val_loss',save_best_only=True,mode='min',period=5)
callback_list = [checkPoint]

model = Sequential()
model.add(Reshape((61,2000,1),input_shape=input_shape))
model.add(Conv2D(n_filters[1],length_filters[1],strides=length_strides[1],activation='linear',data_format='channels_last'))
model.add(Permute((3,2,1)))
model.add(Conv2D(n_filters[2],length_filters[2],strides=length_strides[2],padding='same',activation='relu',))
model.add(MaxPooling2D(pool_size,strides=length_strides[3]))
model.add(Conv2D(n_filters[3],length_filters[3],strides=length_strides[4],padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size,strides=length_strides[5]))

#model.add()
model.summary()
