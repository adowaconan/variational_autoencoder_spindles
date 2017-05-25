# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:59:04 2017
https://github.com/y0ast/VAE-TensorFlow/blob/master/main.py
https://github.com/pkmital/tensorflow_tutorials/blob/master/python/11_variational_autoencoder.py
@author:  ning
"""

import tensorflow as tf
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('D:\\NING - spindle\\training set\\')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# preparing data: train, validation, test
raw_files = [f for f in os.listdir(os.getcwd()) if ('.fif' in f)]
raw_files = np.random.choice(raw_files,size=1,replace=False)
epochs = []
for ii,raw_name in enumerate(raw_files):
    raw = mne.io.read_raw_fif(raw_name,)
    duration = 5
    events = mne.make_fixed_length_events(raw,id=1,duration=duration,start=300,stop=raw.times[-1]-300)
    epoch=mne.Epochs(raw,events,tmin=0,tmax=duration,preload=True,
                            proj=False).resample(128)
    epoch=epoch.pick_channels(epoch.ch_names[:-2])
    if ii == 0:
        epochs = epoch.get_data()
    else:
        epochs = np.concatenate([epochs,epoch.get_data()])
    raw.close()
print(epochs.shape)
plot_x,plot_y = epochs.shape[1],epochs.shape[2]
epochs = epochs.astype('float32').reshape(epochs.shape[0],-1)
scaler = MinMaxScaler()
scaler.fit(epochs)
x_train,x_test = train_test_split(epochs)
x_validation,x_test = train_test_split(x_test)
x_train = scaler.transform(x_train)
x_validation= scaler.transform(x_validation)
x_test = scaler.transform(x_test)

input_dim = x_train.shape[1]
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
lam = 0

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

x = tf.placeholder("float", shape=[None, input_dim])
l2_loss = tf.constant(0.0)

W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

# Hidden layer encoder
hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

# Mu encoder
mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_logvar = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable ~ posterior
std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.multiply(std_encoder, epsilon)

W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
b_decoder_hidden_reconstruction = bias_variable([input_dim])
l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)
# d_kl(q(z|x)||p(z))
# Appendix B: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
# p(x|z)
x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), reduction_indices=1)

loss = tf.reduce_mean(BCE + KLD)

regularized_loss = loss + lam * l2_loss

loss_summ = tf.summary.scalar("lowerbound", loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver()

n_steps = int(1e6)
batch_size = 256
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('experiment',graph=sess.graph)
    if os.path.isfile('D:/NING - spindle/variational_autoencoder_spindles/save/model.ckpt'):
        print('restoring saved parameters')
        saver.restore(sess,'D:/NING - spindle/variational_autoencoder_spindles/save/model.ckpt')
    else:
        print('Initializing parameters')
        sess.run(tf.global_variables_initializer())
    for step in range(1,n_steps):
        for i in range(int(x_train.shape[0]/batch_size)):
            print(step,i)
            batch = x_train[i*batch_size:(i+1)*batch_size]
            feed_dict = {x:batch}
            _,cur_loss,summary_str = sess.run([train_step,loss,summary_op],feed_dict=feed_dict)
        summary_writer.add_summary(summary_str,step)
        if step % 50 == 0:
            save_path = saver.save(sess,'D:/NING - spindle/variational_autoencoder_spindles/save/model.ckpt')
            print('step {0} | Loss: {1}'.format(step,cur_loss))
