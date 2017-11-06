# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:50:24 2017

@author: ning
"""

import mne
import numpy as np
import os

class generator():
    
    def __init__(self,count_save=[],current_data=None,fileName=None,working_dir=None):
        
        self.working_dir = working_dir
        os.chdir(working_dir)
        files = [f for f in os.listdir() if ('-tfr.h5' in f)]
        self.files = files
        self.count_save=count_save
        self.current_data=current_data
        self.fileName = fileName
        
    def load(self):
        files = self.files
        current_data = np.random.choice(files,size=1,)[0]
        fileName = current_data
        self.current_data = current_data
        count_save = self.count_save
        count_save.append(fileName)
        tfr = mne.time_frequency.read_tfrs(fileName)
        data = tfr.get_data()
        del tfr
        
        return data
        
    
    

if __name__ == "__main__":
    p=generator(working_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\')
    data = p.load()
    