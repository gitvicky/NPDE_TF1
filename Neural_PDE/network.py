#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:22:58 2020

@author: Vicky

Neural PDE - Tensorflow 1.14
Module : Network    

"""
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.compat.v1.set_random_seed(42)

from . import options

class Network(object):
    def __init__(self, layers, lb, ub, activation, initialiser):
        
        self.layers = layers
        self.lb = lb 
        self.ub = ub 
        
        self.initialiser = options.get_initialiser(initialiser)
        self.activation = options.get_activation(activation)

        
    def initialize(self):        
        weights = []
        biases = []
        num_layers = len(self.layers) 
        for l in range(0,num_layers-1):
            W = self.initialiser(size=[self.layers[l], self.layers[l+1]]) #Sending for creating Weight tf.Variables and initialising them with the chosen initialiser
            b = tf.Variable(tf.zeros([1, self.layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size): #Xavier Initialisation of the Weight Values
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.compat.v1.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def normalise(self, X):
        return 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
    
    def forward(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = self.normalise(X)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = self.activation(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    
