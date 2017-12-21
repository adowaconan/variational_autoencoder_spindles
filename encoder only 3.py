# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:31:54 2017

@author: ning
"""

import os
import mne
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mne.decoding import Vectorizer
from sklearn import metrics
import pandas as pd

os.chdir('D:/Ning - spindle/training set')

working_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_20_2017\\'
saving_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\CNN vae\\'
saving_dir_weight = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\'
def cos_similarity(x,y):
    x = Vectorizer().fit_transform(x)
    y = Vectorizer().fit_transform(y)
    metrics_ = np.mean(metrics.pairwise.cosine_similarity(x,y))
    return metrics_
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

seed = np.random.seed(12345)
random_idx = np.random.choice(np.arange(41),size=30,replace=False)

labels = []
for e in np.array([f for f in os.listdir(working_dir) if ('-epo.fif' in f)])[random_idx]:
    
    temp_epochs = mne.read_epochs(working_dir + e,preload=True)
#        temp_data = temp_epochs.get_data()
#        data.append(temp_data)
    labels.append(temp_epochs.events[:,-1])
    
    del temp_epochs

#data = np.concatenate(data,0)
labels = np.concatenate(labels)

data = []

for tf in np.array([f for f in os.listdir(working_dir) if ('-tfr.h5' in f)])[random_idx]:
    tfcs = mne.time_frequency.read_tfrs(working_dir+tf)[0]
    data_ = tfcs.data
    scaler = MinMaxScaler(feature_range=(0,1))
    vectorizer = Vectorizer()
    data_vec = vectorizer.fit_transform(data_)
    data_scaled = scaler.fit_transform(data_vec)
    data_scaled = vectorizer.inverse_transform(data_scaled)
    del tfcs
    data.append(data_scaled)
    del data_, data_scaled, data_vec
data = np.concatenate(data,axis=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=12345)
X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size=0.2,random_state=12345)
del data


    

#########################################################
############## U-net model ##############################
#########################################################
from keras.layers import Input, concatenate, Conv2D, MaxPool2D, UpSampling2D, Dropout
from keras.models import Model
import keras
from keras.callbacks import ModelCheckpoint
inputs = Input(shape=(32, 16, 192), batch_shape=(None,32,16,192),name='input',dtype='float64')
conv1 = Conv2D(64, (3,3), activation='relu',padding='same',kernel_initializer='he_normal',
               data_format='channels_first',dilation_rate=2,use_bias=True)(inputs)
print('conv1 shape:',conv1.shape)
#crop1 = Cropping2D(cropping=((3,3),(3,3)),data_format='channels_first')(conv1)
#print('crop1 shape:', crop1.shape)
conv1_2 = Conv2D(64, (3,3), activation='relu',padding='same',kernel_initializer='he_normal',
               data_format='channels_first',dilation_rate=2,use_bias=True)(conv1)
print('conv1 shape:',conv1_2.shape)
drop1 = Dropout(0.5)(conv1_2)
pool1 = MaxPool2D(pool_size=(2,2),data_format='channels_first')(drop1)
print('pool1 shape:',pool1.shape)

conv2 = Conv2D(128, (3,3), activation='relu',padding='same',kernel_initializer='he_normal',
               data_format='channels_first',dilation_rate=2,use_bias=True)(pool1)
print('conv2 shape:', conv2.shape)
conv2_2 = Conv2D(128, (3,3), activation='relu',padding='same',kernel_initializer='he_normal',
               data_format='channels_first',dilation_rate=2,use_bias=True)(conv2)
print('conv2_2 shape:', conv2_2.shape)
drop2 = Dropout(0.5)(conv2_2)
pool2 = MaxPool2D(pool_size=(2,2),data_format='channels_first')(drop2)
print('pool2 shape:',pool2.shape)

conv3 = Conv2D(256, (3,3), activation='relu',padding='same',kernel_initializer='he_normal',
               data_format='channels_first',dilation_rate=2,use_bias=True)(pool2)
print('conv2 shape:', conv3.shape)
conv3_2 = Conv2D(256, (3,3), activation='relu',padding='same',kernel_initializer='he_normal',
               data_format='channels_first',dilation_rate=2,use_bias=True)(conv3)
print('conv3_2 shape:', conv3_2.shape)
drop3 = Dropout(0.5)(conv3_2)
pool3 = MaxPool2D(pool_size=(2,2),data_format='channels_first')(drop3)
print('pool3 shape:',pool3.shape)

up4 = Conv2D(128, (3,3), activation='relu',padding='same',kernel_initializer='he_normal',
             data_format='channels_first',dilation_rate=2,use_bias=True)(UpSampling2D(size=(2,2),data_format='channels_first')(pool3))
print('up4 shape:', up4.shape)
merge4 = concatenate([drop3, up4],axis=1)
print('merge4 shape:',merge4.shape)
conv4 = Conv2D(128, (4,4), activation='relu',padding='same',kernel_initializer='he_normal',
             data_format='channels_first',dilation_rate=2,use_bias=True)(merge4)
print('conv4 shape:', conv4.shape)
conv4_2 = Conv2D(128, (4,4), activation='relu',padding='same',kernel_initializer='he_normal',
             data_format='channels_first',dilation_rate=2,use_bias=True)(conv4)
print('conv6 shape:', conv4_2.shape)

up5 = Conv2D(64, (3,3), activation='relu',padding='same',kernel_initializer='he_normal',
             data_format='channels_first',dilation_rate=2,use_bias=True)(UpSampling2D(size=(2,2),data_format='channels_first')(conv4_2))
print('up5 shape:', up5.shape)
merge5 = concatenate([drop2, up5],axis=1)
print('merge4 shape:',merge5.shape)
conv5 = Conv2D(128, (4,4), activation='relu',padding='same',kernel_initializer='he_normal',
             data_format='channels_first',dilation_rate=2,use_bias=True)(merge5)
print('conv5 shape:', conv5.shape)
conv5_2 = Conv2D(128, (4,4), activation='relu',padding='same',kernel_initializer='he_normal',
             data_format='channels_first',dilation_rate=2,use_bias=True)(conv5)
print('conv6 shape:', conv5_2.shape)

up6 = Conv2D(32, (3,3), activation='relu',padding='same',kernel_initializer='he_normal',
             data_format='channels_first',dilation_rate=2,use_bias=True)(UpSampling2D(size=(4,4),data_format='channels_first')(conv4_2))
print('up5 shape:', up6.shape)
merge6 = concatenate([drop1,up6],axis=1)
print('merge6 shape:',merge6.shape)
conv6 = Conv2D(32,3, activation='relu',padding='same',kernel_initializer='he_normal',
             data_format='channels_first',dilation_rate=2,use_bias=True)(merge6)
print('conv6 shape:',conv6.shape)

model_U_net = Model(inputs = inputs, outputs = conv6)
model_U_net.compile(optimizer=keras.optimizers.Adam(),
                            loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

breaks = 500
batch_size = 50
file_path = saving_dir_weight+'weights.1D.best.hdf5'# 'weights.1D.vae.best.hdf5'
checkPoint = ModelCheckpoint(file_path,monitor='val_loss',save_best_only=True,mode='min',period=1,verbose=1)
callback_list = [checkPoint]

temp_results = []
if os.path.exists('D:\\NING - spindle\\Spindle_by_Graphical_Features\\weights.2D_u_net.best.hdf5'):
    model_U_net.load_weights('D:\\NING - spindle\\Spindle_by_Graphical_Features\\weights.2D_u_net.best.hdf5')
for ii in range(breaks):
    model_U_net.fit(x=X_train,y=X_train,batch_size=batch_size,epochs=50,
                    validation_data=(X_validation,X_validation),shuffle=True,callbacks=callback_list)
    X_predict = model_U_net.predict(X_validation)
    validation_measure = [cos_similarity(a,b) for a,b in zip(X_validation, X_predict)]
    print('mean similarity: %.4f +/- %.4f'%(np.mean(validation_measure),np.std(validation_measure)))
    temp_results.append([(ii+1)*50,np.mean(validation_measure),np.std(validation_measure)])
    results_for_saving = pd.DataFrame(np.array(temp_results).reshape(-1,3),columns=['epochs','mean score','score std'])
    results_for_saving.to_csv(saving_dir_weight + 'scores.csv',index=False)












































