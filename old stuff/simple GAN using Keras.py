# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:11:51 2017

@author: ning
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense

def sample(N,mu=2.0,sigma=0.5):
    samples = np.random.normal(mu,sigma, N)
    samples.sort()
    return np.reshape(samples,(len(samples),1))
    
def g_sample(N,rng=5):
    a = np.linspace(-rng, rng, N) + np.random.random(N) * 0.01
    return np.reshape(a,(len(a),1))
    
def gen(hidden_size):
    g_input = Input(shape=[1],name='input 1')
    h0 = Dense(hidden_size,activation='softplus')(g_input)
    
    h1 = Dense(1,name='g2')(h0)
    g=Model(g_input,h1)
    g.compile(loss='binary_crossentropy',optimizer='adadelta')
    return g

def disc(hidden_size):
    input = Input(shape=[1],name='input 2')
    h0 = Dense(hidden_size*2, activation='tanh',name='d1')(input)
    h1 = Dense(hidden_size*2, activation='tanh',name='d2')(h0)
    h2 = Dense(hidden_size*2, activation='tanh',name='d3')(h1)
    h3 = Dense(2, activation='sigmod',name='d4')(h2)
    d=Model(input,h3)
    d.compile(loss='binary_crossentropy',optimizer='adadelta')
    return d

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        
noise = Input(shape=[1])
generator = gen(100)
g=generator(noise)
discriminator = disc(100)
d=discriminator(g)
make_trainable(discriminator,False)
GAN = Model(noise,d)
GAN.compile(loss='binary_crossentropy',optimizer='adadelta')
GAM.summary()

def train_for_n(nb_epoch=50, plt_freq = 25, batch_size = 100):
    for e in range(nb_epoch):
        sample_batch = sample(batch_size)
        noise_gen = g_sample(batch_size)
        generated_sample = generator.predict(noise_gen)
        
        X = np.concatenate((sample_batch,generated_sample))
        y = np.zeros([2*batch_size,2])
        y[0:batch_size,0] = 1
        y[batch_size:,1] = 1
        make_trainable(discriminator,True)
        d_loss = discriminator.train_on_batch(X,y)
        y2 = np.zeros([batch_size,2])
        y2[0:batch_size,0] = 1
        noise_gen = g_sample(batch_size)
        make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_gen,y2)
        print(d_loss,g_loss)
        