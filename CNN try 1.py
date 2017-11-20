# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:25:47 2017

@author: ning
"""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from scipy import stats
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,Model
from keras.layers import Dense,Lambda,Layer
from keras.layers import Dropout
from keras.layers import Flatten,Reshape
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import losses,metrics
from keras.layers.convolutional import Conv2D,Conv1D,Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D,MaxPooling1D
from keras.layers import UpSampling2D,UpSampling1D
from keras.utils import np_utils
from keras import backend as K
import os
from sklearn.model_selection import train_test_split

working_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\'
def generate_(n=3):
    temp_ = {}
    for ii in range(n):
        temp_[ii] = np.load(working_dir+'data%d.npy'%(ii+1),)
    temp_list = [temp_[ii] for ii in range(n)]
    data = np.concatenate(temp_list)
    return data
data = generate_(6)
data = data.astype('float32')
data = data * 1e6 # rescale 

def sampling(args):
    z_mean,z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],2),mean=0.,stddev=1)
    return z_mean + K.exp(z_log_var /2 ) * epsilon
class CustomVariationalLayer(Layer):
    
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer,self).__init__(**kwargs)
        
    def vae_loss(self,x,x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = metrics.binary_crossentropy(x,x_decoded_mean_squash)
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis=-1)
        return K.mean(xent_loss + kl_loss)
    
    def call(self,inputs):
        x=inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x,x_decoded_mean_squash)
        self.add_loss(loss,inputs=inputs)
        return x
# some preprocessing the outlier values (extreme values)
#Trimmed_data = stats.trimboth(data.flatten(),0.05)
#idx_max = data > Trimmed_data.max()
#idx_min = data < Trimmed_data.min()
#data[idx_max] = Trimmed_data.max()
#data[idx_min] = Trimmed_data.min()

normal_data = (data - data.mean()) / data.std()
# leave out test data
X_train,X_test = train_test_split(normal_data,test_size=0.01)
# set up some hyper parameters
batch_size = 250
n_filters = 32
length_filters = 50
pool_size = 10
length_strides = 1
n_output = 25
file_path = working_dir+'weights.1D.best.hdf5'# 'weights.1D.vae.best.hdf5'
checkPoint = ModelCheckpoint(file_path,monitor='val_loss',save_best_only=True,mode='min',period=1,verbose=1)
callback_list = [checkPoint]

# input
X = Input(shape=X_train.shape[1:],name='input_layer')
# 1st encoder layer
Enc_1 = Conv1D(n_filters,length_filters,strides=length_strides,padding='same',activation='tanh',name='encoder_layer_1')(X)
normal_Enc_1 = BatchNormalization(name='batch_normalize_layer_1')(Enc_1)
max_pool_Enc_1 = MaxPooling1D(pool_size,name='MaxPooling1D_layer_1')(normal_Enc_1)
dropOut_1 = Dropout(0.3,name='dropout_layer_1')(max_pool_Enc_1)
# 2nd encoder layer
Enc_2 = Conv1D(int(n_filters/2),int(length_filters/2),strides=length_strides,padding='same',activation='sigmoid',
               name='encoder_layer_2')(dropOut_1)
normal_Enc_2 = BatchNormalization(name='batch_normalize_layer_2')(Enc_2)
max_pool_Enc_2 = MaxPooling1D(int(pool_size/5),name='MaxPooling1D_layer_2')(normal_Enc_2)
dropOut_2 = Dropout(0.5,name='dropout_layer_2')(max_pool_Enc_2)
# 3rd encoder layer
Enc_3 = Conv1D(int(n_filters/4),int(length_filters/4),strides=length_strides,padding='same',activation='sigmoid',
               name='encoder_layer_3')(dropOut_2)
normal_Enc_3 = BatchNormalization(name='batch_normalize_layer_3')(Enc_3)
max_pool_Enc_3 = MaxPooling1D(int(pool_size/2),name='MaxPooling1D_layer_3')(normal_Enc_3)
dropOut_3 = Dropout(0.6,name='dropout_layer_3')(max_pool_Enc_3)
# flatten to latend layer
flatten_layer = Flatten(name='flatten_layer')(dropOut_3)
hidden_enc = Dense(64,name='hidden_enc')(flatten_layer)
z_mean = Dense(2,name='z_mean')(hidden_enc)
z_log_var = Dense(2,name='z_log_var')(hidden_enc)
# sampling from latend layer
z = Lambda(sampling, output_shape=(2,),name='sampling')([z_mean,z_log_var])
decoder_hid = Dense(64,activation='relu',name='hidden_side_of_decoder')(z)
decoder_upsample = Dense(160,activation='relu',name='upsample_decoder_hidden_1')(decoder_hid)
# reshape for further processing
decoder_reshape = Reshape((20,8),name='reshape_row_channels')(decoder_upsample)
# 1st decoder layer
Dnc_1 = Conv1D(int(n_filters/4),int(length_filters/4),strides=length_strides,padding='same',activation='sigmoid',
               name='decoder_layer_1')(decoder_reshape)
up_sample_1 = UpSampling1D(int(pool_size/5),name='upsample_1')(Dnc_1)
normal_Dnc_1 = BatchNormalization(name='batch_normalize_layer_dnc_1')(up_sample_1)
dropOut_Dnc_1 = Dropout(0.5,name='drop_out_dnc_1')(normal_Dnc_1)
# 2nd decoder layer
Dnc_2 = Conv1D(int(n_filters/2),int(length_filters/2),strides=length_strides,padding='same',activation='sigmoid',
               name='decoder_layer_2')(Dnc_1)
up_sample_2 = UpSampling1D(int(pool_size),name='upsample_2')(Dnc_2) # don't why I cannot divide by 2 this way
normal_Dnc_2 = BatchNormalization(name='batch_normalize_layer_dnc_2')(up_sample_2)
dropOut_Dnc_2 = Dropout(0.3,name='drop_out_dnc_2')(normal_Dnc_2)
# 3rd decoder layer
Dnc_3 = Conv1D(int(n_filters),int(length_filters),strides=length_strides,padding='same',activation='sigmoid',
               name='decoder_layer_3')(dropOut_Dnc_2)
up_sample_3 = UpSampling1D(int(pool_size),name='upsample_3')(Dnc_3) # don't why I cannot divide by 2 this way
normal_Dnc_3 = BatchNormalization(name='batch_normalize_layer_dnc_3')(up_sample_3)
# output layer
output_layer = Conv1D(32,length_filters,strides=length_strides,padding='same',activation='sigmoid',
                      name='output_layer')(normal_Dnc_3)
# variational compliable layer
y = CustomVariationalLayer()([X,output_layer])
vae = Model(X,y)
vae.compile(optimizer='rmsprop',loss=None,metrics=[metrics.mean_squared_error,
                                                                       metrics.kullback_leibler_divergence,
                                                                       metrics.logcosh])
vae.summary()
if os.path.exists('D:\\NING - spindle\\Spindle_by_Graphical_Features\\weights.1D.vae.best.hdf5'):
    vae.load_weights('D:\\NING - spindle\\Spindle_by_Graphical_Features\\weights.1D.vae.best.hdf5')
X_train_, X_validation = train_test_split(X_train,test_size=0.2)
history_callback = vae.fit(X_train,shuffle=True,batch_size=batch_size,epochs=5000,validation_data=(X_validation,None),callbacks=callback_list,verbose=1,)
pickle.dump(history_callback,open(working_dir + 'history_callback_vae.p','wb'))

encoder = Model(X,z_mean)
x_test_encoded = encoder.predict(X_test,batch_size=batch_size)
fig,ax = plt.subplots(figsize=(8,8))
ax.scatter(x_test_encoded[:,0],x_test_encoded[:,1])

decoder_input = Input(shape=(2,))
_decoder_hid = decoder_hid(decoder_hid)
_decoder_upsample = decoder_upsample(_decoder_hid)
_Dnc_1 = Dnc_1(_decoder_upsample)
_up_sample_1 = up_sample_1(_Dnc_1)


layer_ = {}
model = Sequential()
model.add(Conv1D(n_filters,length_filters,strides=length_strides,padding='same',activation='sigmoid',input_shape=(2000,32)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size))
model.add(Dropout(0.3))
model.add(Conv1D(int(n_filters/2),int(length_filters/2),strides=length_strides,padding='same',activation='sigmoid',))
model.add(BatchNormalization())
model.add(MaxPooling1D(int(pool_size/5)))
model.add(Dropout(0.5))
model.add(Conv1D(int(n_filters/4),int(length_filters/4),strides=length_strides,padding='same',activation='sigmoid',))
model.add(BatchNormalization())
model.add(MaxPooling1D(int(pool_size/2)))
model.add(Dropout(0.6))
model.add(Conv1D(int(n_filters/4),int(length_filters/4),strides=length_strides,padding='same',activation='sigmoid',))
model.add(UpSampling1D(int(pool_size/5)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(int(n_filters/2),int(length_filters/2),strides=length_strides,padding='same',activation='sigmoid',))
model.add(UpSampling1D(int(pool_size/2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(n_filters,length_filters,strides=length_strides,padding='same',activation='sigmoid',))
model.add(UpSampling1D(pool_size))
model.add(BatchNormalization())
model.add(Conv1D(32,length_filters,strides=length_strides,padding='same',activation='sigmoid'))
model.compile(optimizer='sgd',loss=losses.mean_squared_error,metrics=[metrics.mean_squared_error,
                                                                       metrics.kullback_leibler_divergence,
                                                                       metrics.logcosh])
model.summary()
if os.path.exists('D:\\NING - spindle\\Spindle_by_Graphical_Features\\weights.1D.best.hdf5'):
    model.load_weights('D:\\NING - spindle\\Spindle_by_Graphical_Features\\weights.1D.best.hdf5')
history_callback = model.fit(X_train,X_train,batch_size=batch_size,epochs=5000,validation_split=0.2,callbacks=callback_list,verbose=1,)
pickle.dump(history_callback,open(working_dir + 'history_callback.p','wb'))

X_pred = model.predict(X_test)
idx = np.random.choice(np.arange(len(X_test)),size=1,)
fig,ax=plt.subplots(nrows=2)
ax[0].plot(X_test[idx][0])
ax[1].plot(X_pred[idx][0]*data.std() + data.mean())
