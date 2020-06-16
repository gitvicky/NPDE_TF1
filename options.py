#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:37:55 2020

@author: Vicky

Neural PDE - Tensorflow 1.14
Module : options
"""

import numpy as np
import tensorflow as tf

#Activation Function

def get_activation(name):
    return {
            "tanh": tf.tanh,
            "sigmoid": tf.sigmoid,
            "relu": tf.nn.relu
            }[name]
    raise ValueError("Unknown Activation Function")
    
    
#Optimiser 
    
def get_optimiser(type, name, loss=None, lr=0.01):
    if type == 'GD':
        return optimiser_GD(name, lr)
    elif type == 'QN':
        return optimiser_QN(name, loss)
        
    
    
def optimiser_GD(name, lr=0.001):
    return {
            'Adam': tf.compat.v1.train.AdamOptimizer(),
            'Adadelta': tf.compat.v1.train.AdadeltaOptimizer(),
            'Adagrad': tf.compat.v1.train.AdagradOptimizer(lr),
            'RMS': tf.compat.v1.train.RMSPropOptimizer(lr),
            'SGD': tf.compat.v1.train.GradientDescentOptimizer(lr)
            }[name]
    
def optimiser_QN(name, loss):
     return tf.contrib.opt.ScipyOptimizerInterface(loss, 
                                                   method = 'L-BFGS-B', 
                                                   options = {'maxiter': 50000,
                                                              'maxfun': 50000,
                                                              'maxcor': 50,
                                                              'maxls': 50,
                                                              'ftol' : 1.0 * np.finfo(float).eps})
        
        
#Initialiser 
        
def get_initialiser(name):
    return{'Xavier': xavier_init}[name]
    raise ValueError("Unknown Initialiser")

    
        
def xavier_init(size): #Xavier Initialisation of the Weight Values
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.compat.v1.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

'''
Notes: 
    Might still have to implement other initialisation strategies as functions by hand.
    and connect is it to the options as done in the NPDE_TF2 package. 
    
'''