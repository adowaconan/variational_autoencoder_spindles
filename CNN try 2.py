# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:06:17 2017

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
data = data * 1e5 # rescale 
temp_ = []
for k in data:
    k = k.T
    temp_.append(k)
data = np.array(temp_)

# some preprocessing the outlier values (extreme values)
Trimmed_data = stats.trimboth(data.flatten(),0.05)
idx_max = data > Trimmed_data.max()
idx_min = data < Trimmed_data.min()
data[idx_max] = Trimmed_data.max()
data[idx_min] = Trimmed_data.min()


# leave out test data
X_train,X_test = train_test_split(data)
# set up some hyper parameters
batch_size = 100
n_filters = {1:50,2:10,3:10,4:61,5:61}
length_filters = {1:(61,1),2:(10,25),3:(5,10),4:(10,20),5:(1,1)}
pool_size={1:(2,2),2:(2,2),3:(1,2),4:(1,10)}
length_strides = {1:(1,1),2:(5,10),3:(1,10),4:(2,2),5:(1,1),6:(1,1),7:(1,1)}
n_output = 25
input_shape = (61,2000)
file_path = 'weights.2D.best.hdf5'
checkPoint = ModelCheckpoint(file_path,monitor='val_loss',save_best_only=True,mode='min',period=5)
callback_list = [checkPoint]

model = Sequential()
model.add(Reshape((61,2000,1),input_shape=input_shape))
model.add(Conv2D(n_filters[1],length_filters[1],strides=length_strides[1],padding='same',activation='relu',data_format='channels_first'))
#model.add(Permute((3,2,1)))
model.add(MaxPooling2D(pool_size[1],strides=length_strides[2],padding='same'))
model.add(Conv2D(n_filters[2],length_filters[2],strides=length_strides[3],padding='same',activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size[2],strides=length_strides[4],padding='same',data_format='channels_first'))

model.add(Conv2D(n_filters[3],length_filters[3],strides=length_strides[5],padding='same',activation='relu',data_format='channels_first'))
model.add(UpSampling2D(pool_size[3]))
model.add(Conv2D(n_filters[4],length_filters[4],strides=length_strides[6],padding='same',activation='relu',data_format='channels_first'))
model.add(UpSampling2D(pool_size[4]))
#model.add(Conv2D(n_filters[5],length_filters[5],strides=length_strides[7],padding='same',activation='relu',data_format='channels_first'))
model.summary()

model.compile(optimizer='sgd',loss=losses.binary_crossentropy,metrics=['mae'])
model.fit(X_train,X_train,batch_size=batch_size,epochs=500,validation_split=0.2,callbacks=callback_list,verbose=0)

X_pred = model.predict(X_test)
idx = np.random.choice(np.arange(len(X_test)),size=1,)
fig,ax=plt.subplots(nrows=2)
ax[0].plot(X_test[idx].reshape(2000,61))
ax[1].plot(X_pred[idx].reshape(2000,61))
