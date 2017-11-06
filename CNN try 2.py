# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:06:17 2017

@author: ning
"""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from scipy import stats
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,Reshape
from keras.layers import Input,Permute
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import losses
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.convolutional import MaxPooling2D,MaxPooling1D
from keras.layers import UpSampling2D,UpSampling1D
from keras.utils import np_utils
from keras import backend as K

from sklearn.model_selection import train_test_split

working_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\'


