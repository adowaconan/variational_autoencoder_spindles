# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:31:54 2017

@author: ning
"""

import os
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
from mne.decoding import Vectorizer
from sklearn import metrics
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from keras.utils import np_utils
#os.chdir('D:/Ning - spindle/variational_autoencoder_spindles')
#from DataGenerator import DataGenerator

os.chdir('D:/Ning - spindle/training set')

working_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_20_2017\\'
saving_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\CNN vae\\'
saving_dir_weight = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\shallow\\'
def cos_similarity(x,y):
    x = Vectorizer().fit_transform(x)
    y = Vectorizer().fit_transform(y)
    metrics_ = np.mean(metrics.pairwise.cosine_similarity(x,y))
    return metrics_
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
if not os.path.exists(saving_dir_weight):
    os.mkdir(saving_dir_weight)
#X_train, y_train = [],[]
#for ii in range(10):
#    temp = pickle.load(open('D:\\NING - spindle\\Spindle_by_Graphical_Features\\data\\train\\train%d.p'%(ii),'rb'))
#    X_train_,y_train_ = temp
#    X_train.append(X_train_)
#    y_train.append(y_train_)
#    
#X_train = np.concatenate(X_train,axis=0)
#y_train = np.concatenate(y_train,axis=0)

X_validation,y_validation = pickle.load(open('D:\\NING - spindle\\Spindle_by_Graphical_Features\\data\\validation\\validation.p','rb'))


#X_train = np.concatenate([X_train,X_validation],axis=0)
#y_train = np.concatenate([y_train,y_validation],axis=0)
#del X_validation,y_validation

#########################################################
############## covn autoencoder model ###################
#########################################################
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D,Dropout,BatchNormalization
from keras.layers import Flatten,Dense
from keras.models import Model
import keras
from keras.callbacks import ModelCheckpoint

inputs = Input(shape=(32,16,192),batch_shape=(None,32,16,192),name='input',dtype='float64',)
conv1 = Conv2D(256,(4,48),strides=(1,1),activation='relu',padding='valid',data_format='channels_first',
              kernel_initializer='he_normal')(inputs)
print('conv1 shape:',conv1.shape)
drop1 = Dropout(0.5)(conv1)
norm1 = BatchNormalization()(drop1)
print('norm1 shape:',norm1.shape)
#down1 = MaxPooling2D((2,4),(1,2),padding='valid',data_format='channels_first',)(norm1)
#print('down1 shape:', down1.shape)

#conv2 = Conv2D(128,(4,48),strides=(1,1),activation='relu',padding='valid',data_format='channels_first',
#              kernel_initializer='he_normal')(norm1)
#print('conv2 shape:',conv2.shape)
#drop2 = Dropout(0.5)(conv2)
#norm2 = BatchNormalization()(drop2)
#print('norm2 shape:',norm2.shape)
#down2 = MaxPooling2D((2,4),(1,2),padding='valid',data_format='channels_first',)(norm2)
#print('down2 shape:',down2.shape)

#conv3 = Conv2D(128,(4,48),strides=(1,1),activation='relu',padding='valid',data_format='channels_first',
#              kernel_initializer='he_normal')(norm2)
#print('conv3 shape:',conv3.shape)
#drop3 = Dropout(0.5)(conv3)
#norm3 = BatchNormalization()(drop3)
#print('norm3 shape:',norm3.shape)
#down3 = MaxPooling2D((2,4),(1,2),padding='valid',data_format='channels_first',)(norm3)
#print('down3 shape:',down3.shape)

#conv4 = Conv2D(64,(4,48),strides=(1,1),activation='relu',padding='valid',data_format='channels_first',
#              kernel_initializer='he_normal')(norm3)
#print('conv4 shape:',conv4.shape)
#drop4 = Dropout(0.5)(conv4)
#norm4 = BatchNormalization()(drop4)
#print('norm4 shape:',norm4.shape)
#down4 = MaxPooling2D((2,4),(1,2),padding='valid',data_format='channels_first',)(norm4)
#print('down4 shape:',down4.shape)

#conv5 = Conv2D(32,(2,2),strides=(1,1),activation='relu',padding='same',data_format='channels_first',
#              kernel_initializer='he_normal')(norm4)
#print('conv5 shape:',conv5.shape)
#drop5 = Dropout(0.5)(conv5)
#norm5 = BatchNormalization()(drop5)
#print('norm5 shape:',norm5.shape)
#down5 = MaxPooling2D((2,2),(1,1),padding='valid',data_format='channels_first',)(norm5)
#print('down5 shape:',down5.shape)

flat6 = Flatten()(norm1)
drop6 = Dropout(0.5)(flat6)
print('flatten 6 shape:',drop6.shape)

dens7 = Dense(kernel_initializer='he_normal',units=2,activation='softmax')(flat6)
drop7 = Dropout(0.5)(dens7)
print('dense 7 shape:',drop7.shape)
#decov4 = Conv2DTranspose(128,(4,8),strides=(2,4),activation='relu',padding='same',data_format='channels_first',
#                        kernel_initializer='he_normal')(down3)
#print('decov4 shape:',decov4.shape)
#drop4 = Dropout(0.5)(decov4)
#norm4 = BatchNormalization()(drop4)
#
#decov5 = Conv2DTranspose(64,(4,8),strides=(2,4),activation='relu',padding='same',data_format='channels_first',
#                        kernel_initializer='he_normal')(norm4)
#print('decov5 shape:',decov5.shape)
#drop5 = Dropout(0.5)(decov5)
#norm5 = BatchNormalization()(drop5)
#
#decov6 = Conv2DTranspose(32,(4,8),strides=(4,4),activation='relu',padding='same',data_format='channels_first',
#                        kernel_initializer='he_normal')(norm5)
#print('decov6 shape:',decov6.shape)
def AUC_(y_true, y_pred):
    return metrics.roc_auc_score(y_true,y_pred)
model_auto = Model(inputs = inputs,outputs=drop7)
model_auto.compile(optimizer=keras.optimizers.SGD(),loss=keras.losses.mse,metrics=['accuracy',
                   'categorical_accuracy'])

#data = np.random.rand(10,32,16,192).astype(np.float64)
#predict = model_auto.predict(data)
#predict.shape

""" auto encoder"""
#breaks = 500
#batch_size = 50
#file_path = saving_dir_weight+'weights.2D_auto_encoder.best.hdf5'
#checkPoint = ModelCheckpoint(file_path,monitor='val_loss',save_best_only=True,mode='min',period=1,verbose=1)
#callback_list = [checkPoint]
#
#temp_results = []
#if os.path.exists('D:\\NING - spindle\\Spindle_by_Graphical_Features\\weights.2D_auto_encoder.best.hdf5'):
#    model_auto.load_weights('D:\\NING - spindle\\Spindle_by_Graphical_Features\\weights.2D_auto_encoder.best.hdf5')
#for ii in range(breaks):
#    model_auto.fit(x=X_train,y=X_train,batch_size=batch_size,epochs=50,
#                    validation_data=(X_validation,X_validation),shuffle=True,callbacks=callback_list)
#    X_predict = model_auto.predict(X_validation)
#    validation_measure = [cos_similarity(a,b) for a,b in zip(X_validation, X_predict)]
#    print('mean similarity: %.4f +/- %.4f'%(np.mean(validation_measure),np.std(validation_measure)))
#    temp_results.append([(ii+1)*50,np.mean(validation_measure),np.std(validation_measure)])
#    results_for_saving = pd.DataFrame(np.array(temp_results).reshape(-1,3),columns=['epochs','mean score','score std'])
#    if os.path.exists(saving_dir_weight + 'scores_autoencoder.csv'):
#        temp_result_for_saving = pd.read_csv(saving_dir_weight + 'scores_autoencoder.csv')
#        results_for_saving = pd.concat([temp_result_for_saving,results_for_saving])
#    results_for_saving.to_csv(saving_dir_weight + 'scores_autoencoder.csv',index=False)

"""classification"""
breaks = 500
batch_size = 100
through = 5
file_path = saving_dir_weight+'weights.2D_classification_shallow.best.hdf5'
checkPoint = ModelCheckpoint(file_path,monitor='val_loss',save_best_only=True,mode='min',period=1,verbose=1)
callback_list = [checkPoint]
temp_results = []
if os.path.exists(saving_dir_weight+'weights.2D_classification_small_to_large.best.hdf5'):
    model_auto.load_weights(saving_dir_weight+'weights.2D_classification_shallow.best.hdf5')


for ii in range(breaks):
    labels = []
    for jj in range(through):# going through the training data 5 times
        step_idx = np.random.choice(np.arange(10),size=10,replace=False)
        for kk in step_idx: # going through 10 splitted training data
            temp = pickle.load(open('D:\\NING - spindle\\Spindle_by_Graphical_Features\\data\\train\\train%d.p'%(kk),'rb'))
            X_train_,y_train_ = temp
            random_inputs = np.random.rand(X_train_.shape[0],32,16,192)
            random_labels = [0]*X_train_.shape[0]
            random_labels = np_utils.to_categorical(random_labels,2)
            X_train_ = np.concatenate([X_train_,random_inputs],axis=0)
            y_train_ = np.concatenate([y_train_,random_labels],axis=0)
            labels.append(y_train_)
            model_auto.fit(x=X_train_,y=y_train_,batch_size=batch_size,epochs=2,
                        validation_data=(X_validation,y_validation),shuffle=True,callbacks=callback_list)
    labels = np.concatenate(labels,axis=0)
    model_auto.load_weights(saving_dir_weight+'weights.2D_classification_shallow.best.hdf5')
    X_predict = model_auto.predict(X_validation)[:,-1] > np.mean(labels[:,-1])
    X_predict_prob = model_auto.predict(X_validation)[:,-1]
    print(metrics.classification_report(y_validation[:,-1],X_predict))
    AUC = metrics.roc_auc_score(y_validation[:,-1], X_predict_prob)
    fpr,tpr,th = metrics.roc_curve(y_validation[:,-1], X_predict_prob,pos_label=1)
    sensitivity = metrics.precision_score(y_validation[:,-1],X_predict,average='weighted')
    selectivity = metrics.recall_score(y_validation[:,-1],X_predict,average='weighted')
    plt.close('all')
    fig,ax = plt.subplots(figsize=(8,8))
    ax.plot(fpr,tpr,label='AUC = %.3f'%(AUC))
    ax.set(xlabel='false postive rate',ylabel='true positive rate',title='%dth 5 epochs'%(ii+1),
           xlim=(0,1),ylim=(0,1))
    ax.legend(loc='best')
    fig.savefig(saving_dir_weight + 'AUC plot_%d.png'%(ii+1),dpi=400)
    plt.close('all')
#    validation_measure = [cos_similarity(a,b) for a,b in zip(X_validation, X_predict)]
#    print('mean similarity: %.4f +/- %.4f'%(np.mean(validation_measure),np.std(validation_measure)))
    temp_results.append([(ii+1)*50,AUC,sensitivity,selectivity])
    results_for_saving = pd.DataFrame(np.array(temp_results).reshape(-1,4),columns=['epochs','AUC','sensitivity','selectivity'])
    if os.path.exists(saving_dir_weight + 'scores_classification.csv'):
        temp_result_for_saving = pd.read_csv(saving_dir_weight + 'scores_classification.csv')
        results_for_saving = pd.concat([temp_result_for_saving,results_for_saving])
    results_for_saving.to_csv(saving_dir_weight + 'scores_classification.csv',index=False)

X_test, y_test = pickle.load(open('D:\\NING - spindle\\Spindle_by_Graphical_Features\\data\\test\\test.p','rb'))

X_predict_ = model_auto.predict(X_test)[:,-1] > 0.5
X_predict_prob_ = model_auto.predict(X_test)[:,-1]
print(metrics.classification_report(y_test[:,-1],X_predict_))
AUC = metrics.roc_auc_score(y_test[:,-1], X_predict_prob_)
fpr,tpr,th = metrics.roc_curve(y_test[:,-1], X_predict_prob_,pos_label=1)
sensitivity = metrics.precision_score(y_test[:,-1],X_predict_,average='weighted')
selectivity = metrics.recall_score(y_test[:,-1],X_predict_,average='weighted')
plt.close('all')
fig,ax = plt.subplots(figsize=(8,8))
ax.plot(fpr,tpr,label='AUC = %.3f\nSensitivity = %.3f\nSelectivity = %.3f'%(AUC,sensitivity,selectivity))
ax.set(xlabel='false postive rate',ylabel='true positive rate',title='test data\nshallow net',
       xlim=(0,1),ylim=(0,1))
ax.legend(loc='best')
fig.savefig(saving_dir_weight + 'test data AUC plot.png',dpi=400)
plt.close('all')

cf =metrics.confusion_matrix(y_test[:,-1],X_predict_)
cf = cf / cf.sum(1)[:, np.newaxis]
import seaborn as sns
plt.close('all')
fig,ax = plt.subplots(figsize=(8,8))
ax = sns.heatmap(cf,vmin=0.,vmax=1.,cmap=plt.cm.Blues,annot=False,ax=ax)
coors = np.array([[0,0],[1,0],[0,1],[1,1],])+ 0.5
for ii,(m,coor) in enumerate(zip(cf.flatten(),coors)):
    ax.annotate('%.2f'%(m),xy = coor,size=25,weight='bold',ha='center')
ax.set(xticks=(0.5,1.5),yticks=(0.25,1.25),
        xticklabels=['non spindle','spindle'],
        yticklabels=['non spindle','spindle'])
ax.set_title('Confusion matrix\nshallow net',fontweight='bold',fontsize=20)
ax.set_ylabel('True label',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig(saving_dir_weight+'confusion matrix.png',dpi=400)


























