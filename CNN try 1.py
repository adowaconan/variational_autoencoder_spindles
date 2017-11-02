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
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import losses,metrics
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
data = data * 1e6 # rescale 


# some preprocessing the outlier values (extreme values)
#Trimmed_data = stats.trimboth(data.flatten(),0.05)
#idx_max = data > Trimmed_data.max()
#idx_min = data < Trimmed_data.min()
#data[idx_max] = Trimmed_data.max()
#data[idx_min] = Trimmed_data.min()

normal_data = (data - data.mean()) / data.std()
# leave out test data
X_train,X_test = train_test_split(normal_data,test_size=0.01)
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
model.add(Conv1D(n_filters,length_filters,strides=length_strides,padding='same',activation='sigmoid',input_shape=(2000,61)))
model.add(MaxPooling1D(pool_size))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(int(n_filters/2),int(length_filters/2),strides=length_strides,padding='same',activation='sigmoid',))
model.add(MaxPooling1D(int(pool_size/5)))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(int(n_filters/4),int(length_filters/4),strides=length_strides,padding='same',activation='sigmoid',))
model.add(MaxPooling1D(int(pool_size/2)))
#model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Conv1D(int(n_filters/4),int(length_filters/4),strides=length_strides,padding='same',activation='sigmoid',))
model.add(UpSampling1D(int(pool_size/5)))
#model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Conv1D(int(n_filters/2),int(length_filters/2),strides=length_strides,padding='same',activation='sigmoid',))
model.add(UpSampling1D(int(pool_size/2)))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(n_filters,length_filters,strides=length_strides,padding='same',activation='sigmoid',))
model.add(UpSampling1D(pool_size))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(61,length_filters,strides=length_strides,padding='same',activation='sigmoid'))


model.compile(optimizer='sgd',loss=losses.mean_squared_error,metrics=[metrics.mean_squared_error,
                                                                       metrics.kullback_leibler_divergence,
                                                                       metrics.logcosh])
model.summary()


model.fit(X_train,X_train,batch_size=batch_size,epochs=3000,validation_split=0.2,callbacks=callback_list,verbose=1,)

X_pred = model.predict(X_test)
idx = np.random.choice(np.arange(len(X_test)),size=1,)
fig,ax=plt.subplots(nrows=2)
ax[0].plot(X_test[idx])
ax[1].plot(X_pred[idx])
