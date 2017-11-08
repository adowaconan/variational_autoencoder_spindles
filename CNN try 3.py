# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:05:43 2017

@author: ning

we will add mu and std to the latent layer and reduce the complexity of the autoencoder

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

working_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\'
def generate_(n=3):
    temp_ = {}
    for ii in range(n):
        temp_[ii] = np.load(working_dir+'data%d.npy'%(ii+1),)
    temp_list = [temp_[ii] for ii in range(n)]
    data = np.concatenate(temp_list)
    return data
data = generate_(7)
data = data.astype('float32')
data = data * 1e6 # rescale 

scaler = MinMaxScaler(feature_range=(-1,1))
normal_data = scaler.fit_transform(data.reshape(data.shape[0],-1))
normal_data = normal_data.reshape(data.shape)
# leave out test data
X_train,X_test = train_test_split(normal_data,test_size=0.01)

input_height = 1
input_width = 2000
num_channels = 32

batch_size = 50
kernel_size = 50
depth = 60

learning_rate = 0.0001
training_epochs = 5000

total_batches = X_train.shape[0] // batch_size

X = tf.placeholder(tf.float32,shape=[None,input_width,num_channels])

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1,)
    return tf.Variable(initial,name = name)
def bias_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name = name)
def apply_1D(input_,output_size,width,stride,name):
    inputSize = input_.get_shape()[1:]
    input_ = tf.expand_dims(input_,axis=1)
    filter_ = tf.get_variable("conv_filter",shape=[1,width,inputSize,output_size])
    convolved = tf.nn.conv2d(input_,filter=filter_,strides=[1,1,stride,1],padding='same',name=name)
    return tf.squeeze(convolved,axis=1)
def add_bias(convolved,name):
    return tf.nn.relu(convolved,name=name)
def apply_max_pool(x,kernel_size,stride_size,name):
    
    x = tf.expand_dims(x,axis=1)
    return tf.nn.max_pool(x,ksize=[1,1,kernel_size,1],strides=[1,1,stride_size,1],padding='same',name=name)
        
layer_1 = apply_1D(X,output_size=[None,])


