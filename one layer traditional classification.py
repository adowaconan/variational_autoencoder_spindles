# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:36:41 2017

@author: ning
"""

import mne
import os
import numpy as np
from mne.decoding import Vectorizer
from tqdm import tqdm
os.chdir('D:\\NING - spindle\\training set\\') # change working directory
saving_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated\\'
data = mne.time_frequency.read_tfrs(saving_dir+'sub8_d2-eventsRelated-tfr.h5')
data = data[0]
events = data.info['event']
labels = events[:,-1]

from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clf = []
clf.append(('vectorizer',Vectorizer()))
clf.append(('scaler',StandardScaler()))
Cs = np.logspace(-3,3,7)
estimator = LogisticRegressionCV(Cs,cv=4,scoring='roc_auc',max_iter=int(3e3),random_state=12345,class_weight='balanced')
clf.append(('estimator',estimator))
clf = Pipeline(clf)
cv = StratifiedShuffleSplit(n_splits=4,random_state=12345)

scores = []
for ii in tqdm(range(data.data.shape[2])):
    temp = data.data[:,:,ii,:]
    scores.append(cross_val_score(estimator=clf,X=temp,y=labels,scoring='roc_auc',cv=cv))
