# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:06:36 2017

@author: install
"""

import numpy as np
from random import shuffle
import pickle
import os

class DataGenerator(object):
    def __init__(self,batch_size=100,shuffle=True,file_path='D:\\NING - spindle\\Spindle_by_Graphical_Features\\data\\train\\'):
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.file_path=file_path
        
    def generate(self):
        file_path = self.file_path
        files = [f for f in os.listdir(file_path)]
        shuffle(files)
        num_of_files = len(files)
        file_counter = 0
        while True:
            temp = pickle.load(open(file_path+files[file_counter],'rb'))
            X_train_,y_train_ = temp
            file_counter += 1
            if file_counter == num_of_files:
                file_counter = 0
            return (X_train_,y_train_)
            
        
            