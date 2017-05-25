# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:59:04 2017
https://github.com/y0ast/VAE-TensorFlow/blob/master/main.py
https://github.com/pkmital/tensorflow_tutorials/blob/master/python/11_variational_autoencoder.py
@author:  ning
"""

import tensorflow as tf
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('D:\\NING - spindle\\training set\\')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# preparing data: train, validation, test
raw_files = [f for f in os.listdir(os.getcwd()) if ('.fif' in f)]
raw_files = np.random.choice(raw_files,size=1,replace=False)
epochs = []
for ii,raw_name in enumerate(raw_files):
    raw = mne.io.read_raw_fif(raw_name,)
    duration = 5
    events = mne.make_fixed_length_events(raw,id=1,duration=duration,start=300,stop=raw.times[-1]-300)
    epoch=mne.Epochs(raw,events,tmin=0,tmax=duration,preload=True,
                            proj=False).resample(128)
    epoch=epoch.pick_channels(epoch.ch_names[:-2])
    if ii == 0:
        epochs = epoch.get_data()
    else:
        epochs = np.concatenate([epochs,epoch.get_data()])
    raw.close()
print(epochs.shape)
plot_x,plot_y = epochs.shape[1],epochs.shape[2]
epochs = epochs.astype('float32').reshape(epochs.shape[0],-1)
scaler = MinMaxScaler()
scaler.fit(epochs)
x_train,x_test = train_test_split(epochs)
x_validation,x_test = train_test_split(x_test)
x_train = scaler.transform(x_train)
x_validation= scaler.transform(x_validation)
x_test = scaler.transform(x_test)

input_dim = x_train.shape[1]

