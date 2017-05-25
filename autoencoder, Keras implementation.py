# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:50:45 2017

@author: ning
"""

# keras autoencoder

import mne
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers
from keras.layers import Dense, Input
from keras.models import Model
import matplotlib.pyplot as plt

os.chdir('D:\\NING - spindle\\training set\\')

raw_files = [f for f in os.listdir(os.getcwd()) if ('.fif' in f)]
raw_files = np.random.choice(raw_files,size=1,replace=False)
epochs = []
for ii,raw_name in enumerate(raw_files):
    raw = mne.io.read_raw_fif(raw_name,)
    duration = 5
    events = mne.make_fixed_length_events(raw,id=1,duration=duration,start=300,stop=raw.times[-1]-300)
    epoch=mne.Epochs(raw,events,tmin=0,tmax=duration,preload=True,
                            proj=False).resample(100)
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

input_shape = x_train.shape[1]

# this is the size of our encoded representations
#encoding_dim = 64  

input_signal = Input(shape=(input_shape,))
encoded = Dense(500, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_signal)
encoded = Dense(100, activation='relu')(encoded)
encoded = Dense(20, activation='relu')(encoded)

decoded = Dense(100, activation='relu')(encoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(input_shape, activation='sigmoid')(decoded)

autoencoder = Model(input=input_signal, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_signal, output=encoded)

# create a placeholder for an encoded (X-dimensional) input
encoded_input_1 = Input(shape=(20,))
encoded_input_2 = Input(shape=(100,))
encoded_input_3 = Input(shape=(500,))

# retrieve the last layer of the autoencoder model
decoder_layer_1 = autoencoder.layers[-3]
decoder_layer_2 = autoencoder.layers[-2]
decoder_layer_3 = autoencoder.layers[-1]

# create the decoder model
decoder_1 = Model(input = encoded_input_1, output = decoder_layer_1(encoded_input_1))
decoder_2 = Model(input = encoded_input_2, output = decoder_layer_2(encoded_input_2))
decoder_3 = Model(input = encoded_input_3, output = decoder_layer_3(encoded_input_3))
# training
autoencoder.compile(optimizer='Adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=int(50),
                batch_size=1024,
                shuffle=True,
                validation_data=(x_validation, x_validation))
# visualization
a=autoencoder.predict(x_test)
a = scaler.inverse_transform(a)
x_test = scaler.inverse_transform(x_test)
ns = np.random.choice(np.arange(x_test.shape[0]),size=3,replace=False)
for n in ns:
    fig, ax = plt.subplots(nrows=2,figsize=(8,8))
    _=ax[0].plot(x_test[n].reshape(plot_x,plot_y).T)
    _=ax[1].plot(a[n].reshape(plot_x,plot_y).T)
    #fig.savefig('D:/NING - spindle/variational_autoencoder_spindles/test results/simple_encoder%d.png'%n)