# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:55:52 2017

@author: ning
"""

import os
import mne
os.chdir('D:/Ning - spindle/')
#import eegPinelineDesign
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from mne.decoding import Vectorizer


os.chdir('D:/Ning - spindle/training set')

working_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_20_2017\\'
saving_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_20_2017\\average_over_channels\\'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
title = {0:'non spindle',1:'spindle'}
for e in os.listdir(working_dir):
    if ('-tfr.h5' in e):
        tfcs = mne.time_frequency.read_tfrs(working_dir+e)
        tfcs = tfcs[0]
        print(e,tfcs.info['event'].values[:,-1].mean(),len(tfcs.info['event'].values[:,-1]))
        data = tfcs.data
        scaler = MinMaxScaler(feature_range=(0,1))
        vectorizer = Vectorizer()
        data_vec = vectorizer.fit_transform(data)
        data_scaled = scaler.fit_transform(data_vec)
        data_scaled = vectorizer.inverse_transform(data_scaled)
        for ii in tqdm(range(data.shape[0]),desc='within'):
            plt.close('all')
            fig,ax = plt.subplots(figsize=(12,5))
            
            
            im = ax.imshow(data_scaled[ii,:,:,:].mean(0),origin='lower',aspect='auto',extent=[0,3000,6,22],
                           vmin=0,vmax=0.3)
            ax.set(xlabel='time (ms)',ylabel='freqency (Hz)',title=title[tfcs.info['event'].values[:,-1][ii]])
            plt.colorbar(im)
            fig.savefig(saving_dir + '%s_%s_%d.png'%(title[tfcs.info['event'].values[:,-1][ii]],e.split('-')[0],ii))
            plt.close('all')
            
        del tfcs,data,data_vec,data_scaled

def spindle_check(x):
    """A simple function to chack if a string contains keyword 'spindle'"""
    import re
    if re.compile('spindle',re.IGNORECASE).search(x):
        return True
    else:
        return False

#tfcs = mne.time_frequency.read_tfrs(working_dir+'sub8_d2-eventsRelated-tfr.h5')[0]
#fig,ax = plt.subplots(figsize=(12,5))
#im = ax.imshow(tfcs.data[226,:,:,:].mean(0),origin='lower',aspect='auto',extent=[0,3000,8,20],vmin=0,vmax=1e-10,
#               )
#ax.set(xlabel='time (ms)',ylabel='freqency (Hz)',title=title[tfcs.info['event'].values[:,-1][226]])
#ax.axhline(11,xmin=500/3000,xmax=1500/3000,color='black')
#ax.axhline(16,xmin=500/3000,xmax=1500/3000,color='black')
#ax.axvline(500,ymin=4/16,ymax=10/15,color='black')
#ax.axvline(1500,ymin=4/16,ymax=10/15,color='black')
#plt.colorbar(im)
#fig.savefig(working_dir+'typical spindle.png',dpi=400)
#
#fig,ax = plt.subplots(figsize=(12,5))
#im = ax.imshow(tfcs.data[133,:,:,:].mean(0),origin='lower',aspect='auto',extent=[0,3000,8,20],vmin=0,vmax=1e-10,
#               )
#ax.set(xlabel='time (ms)',ylabel='freqency (Hz)',title=title[tfcs.info['event'].values[:,-1][133]])
#plt.colorbar(im)
#fig.savefig(working_dir+'typical non-spindle.png',dpi=400)




#EEG = mne.io.read_raw_fif('D:\\NING - spindle\\training set\\suj12_l5nap_day2_raw_ssp.fif',preload=True)
#annotations = pd.read_csv('D:\\NING - spindle\\training set\\suj12_nap_day2_edited_annotations.txt')
#spindle = annotations[annotations.Annotation.apply(spindle_check)]
#EEG_filer=EEG.copy()
#EEG_filer.filter(11,16)
#channelList = ['F3','F4','C3','C4','O1','O2']
#EEG_filer.pick_channels(channelList)
#onset_time,_,Durations,*_,peak_times = eegPinelineDesign.thresholding_filterbased_spindle_searching(EEG_filer,channelList,annotations,moving_window_size=1000,lower_threshold=.4,
#                                        syn_channels=3,l_bound=0.5,h_bound=2,tol=1,higher_threshold=3.4,
#                                        front=300,back=100,sleep_stage=True,proba=False,validation_windowsize=3,l_freq=11,h_freq=16)
#
#onset_pick = onset_time[12]
#
#fig,axes = plt.subplots(nrows=7)
#
#d = []
#for ii, (ch_,ax) in enumerate(zip(channelList,axes.flatten())):
#    idx_ = min(range(len(peak_times[ch_])),key=lambda i:abs(peak_times[ch_][i]-onset_pick))
##    print(peak_times[ch_][idx_])
#    # get temp data to find the peak
#    start,stop = int((peak_times[ch_][idx_] - 2) * 1000), int((peak_times[ch_][idx_] + 2) * 1000)
#    d_,_ = EEG_filer[ii,start:stop] 
#    d_ = d_ * 1e6
#    idxs=eegPinelineDesign.detect_peaks(d_[0,:])
#    idx = np.argmax(d_[0,idxs])
#    start,stop = idxs[idx]-1500,idxs[idx]+1500
#    d_ = d_[0,start:stop]
#    if start < 0:
#        start = int((peak_times[ch_][idx_] - 2) * 1000) - idx + 500
#        stop = start + 3000
#        d_,_ = EEG_filer[ii,start:stop]
#        d_ = d_[0,:] * 1e6
#    d.append(d_)
#    ax.plot(d_)
#d = np.array(d)
#axes[-1].plot(d.mean(0))











































