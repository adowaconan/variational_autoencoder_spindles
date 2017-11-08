# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:50:24 2017

@author: ning
"""

import mne
import numpy as np
import os
from sklearn.model_selection import train_test_split

class generator():
    
    def __init__(self,count_save=[],current_data=None,fileName=None,working_dir=None):
        
        self.working_dir = working_dir
        os.chdir(working_dir)
        files = [f for f in os.listdir() if ('-tfr.h5' in f)]
        files_train,files_test = train_test_split(files,test_size=0.1)
        self.files = files
        self.files_train = files_train
        self.files_test = files_test
        self.count_save=count_save
        self.current_data=current_data
        self.fileName = fileName
        self.test_load = 0
        
    def load(self):
        
        files_train = self.files_train
        count_save = self.count_save
        
        current_data_ = np.random.choice(files_train,size=1,)[0]
        while current_data_ in count_save:
            current_data_ = np.random.choice(files_train,size=1,)[0]
            
            if len(np.unique(p.count_save)) == len(files_train):
                break
        fileName = current_data_
        self.current_data = current_data_
        
        count_save.append(fileName)
        tfr = mne.time_frequency.read_tfrs(fileName)
        data = tfr[0].data
        ch_names = tfr[0].ch_names
        del tfr
        
        return data,ch_names
        
    def load_test(self):
        files_test = self.files_test
        if self.test_load < len(files_test):
            fileName = files_test[self.test_load]
            tfr = mne.time_frequency.read_tfrs(fileName)
            data = tfr[0].data
            ch_names = tfr[0].ch_names
            del tfr
            self.test_load += 1
            return data,ch_names
        else:
            print('all test data were tested once')
    
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    p=generator(working_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\')
#    for ii in range(45):
#        data,ch_names = p.load()
#        del data
#    for ii in range(5):
#        data,ch_names = p.load_test()
#        del data
    data, ch_names = p.load()
    times, freqs = np.arange(0,2002),np.arange(8,21)
    instance=3
    plt.close('all')
    fig, axes = plt.subplots(nrows=4,ncols=8,figsize=(25,10))
    for ii,(ax,ch_) in enumerate(zip(axes.flatten(),ch_names)):
        if ii > 32:
            break
        else:
            ax.pcolormesh(times,freqs,data[instance,ii])
            ax.set(title=ch_)
    fig.tight_layout()
   