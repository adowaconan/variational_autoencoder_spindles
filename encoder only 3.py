# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:31:54 2017

@author: ning
"""

import os
import mne
import numpy as np

os.chdir('D:/Ning - spindle/training set')

working_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_12_2017\\'
saving_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\CNN vae\\'

if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
    
labels = []
for e in [f for f in os.listdir(working_dir) if ('-epo.fif' in f)][:5]:
    
    temp_epochs = mne.read_epochs(working_dir + e,preload=True)
#        temp_data = temp_epochs.get_data()
#        data.append(temp_data)
    labels.append(temp_epochs.events[:,-1])
    
    del temp_epochs

#data = np.concatenate(data,0)
labels = np.concatenate(labels)

data = []

for tf in [f for f in os.listdir(working_dir) if ('-tfr.h5' in f)][:5]:
    tfcs = mne.time_frequency.read_tfrs(working_dir+tf)[0]
    data_ = tfcs.data
    del tfcs
    data.append(data_)
data = np.concatenate(data,axis=0)



from sklearn.pre