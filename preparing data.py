# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:17:10 2017

@author: ning
"""
import os
import mne
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mne.decoding import Vectorizer
working_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_20_2017\\'
labels = []
for e in np.array([f for f in os.listdir(working_dir) if ('-epo.fif' in f)]):
    
    temp_epochs = mne.read_epochs(working_dir + e,preload=True)
#        temp_data = temp_epochs.get_data()
#        data.append(temp_data)
    labels.append(temp_epochs.events[:,-1])
    
    del temp_epochs

#data = np.concatenate(data,0)
labels = np.concatenate(labels)

data = []

for tf in np.array([f for f in os.listdir(working_dir) if ('-tfr.h5' in f)]):
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
idx = np.random.choice(np.arange(len(data)),size=len(data),replace=False)
data,labels = data[idx],labels[idx]
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10,shuffle =True,random_state=12345)
for train,test in kfold.split(data):
    train,test
X_train,X_test = data[train],data[test]
y_train,y_test = labels[train],labels[test]
del data

kfold = KFold(n_splits=5,shuffle =True,random_state=12345)
for train,test in kfold.split(X_train):
    train,test
X_train,X_validation = X_train[train],X_train[test]
y_train,y_validation = y_train[train],y_train[test]


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,2)
y_validation = np_utils.to_categorical(y_validation,2)
y_test = np_utils.to_categorical(y_test,2)

import pickle
for ii,part in enumerate(np.array_split(np.arange(len(X_train)),10)):
    pickle.dump([X_train[part],y_train[part]],open('D:\\NING - spindle\\Spindle_by_Graphical_Features\\data\\train\\train%d.p'%(ii),'wb'))
    
pickle.dump([X_validation,y_validation],open('D:\\NING - spindle\\Spindle_by_Graphical_Features\\data\\validation\\validation.p','wb'))
pickle.dump([X_test,y_test],open('D:\\NING - spindle\\Spindle_by_Graphical_Features\\data\\test\\test.p','wb'))















