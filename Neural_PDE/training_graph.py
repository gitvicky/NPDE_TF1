#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:35:46 2020

@author: Vicky

Neural PDE - Tensorflow 1.14
Module : Training Graph

"""
import time 
import numpy as np
import tensorflow as tf  

np.random.seed(42)
tf.compat.v1.set_random_seed(42)


from .network import Network
from . import boundary_conditions
from . import options
from .sampler import Sampler

class TrainingGraph(Network, Sampler):
    def __init__(self, layers, lb, ub, activation, initialiser, N_f, GD_opt, QN_opt, pde):
        
        Network.__init__(self, layers, lb, ub, activation, initialiser)
        Sampler.__init__(self, N_f, subspace_N = int(N_f/10))
        
        self.layers = layers 
        self.input_size = self.layers[0]
        self.output_size = self.layers[-1]
        
        self.bc = boundary_conditions.select('Dirichlet')
        self.pde = pde
        
        # Initialize NNs
        self.weights, self.biases = self.initialize()
        
        # Creating the placeholders for the tensorflow variables
        self.sess = tf.compat.v1.Session()
        
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, self.input_size])
        self.u = self.forward(self.X, self.weights, self.biases)
                
        self.X_i = tf.compat.v1.placeholder(tf.float32, shape=[None, self.input_size])
        self.u_i = tf.compat.v1.placeholder(tf.float32, shape=[None, self.output_size])
        
        self.X_b = tf.compat.v1.placeholder(tf.float32, shape=[None, self.input_size])
        self.u_b = tf.compat.v1.placeholder(tf.float32, shape=[None, self.output_size])

        self.X_f = tf.compat.v1.placeholder(tf.float32, shape=[None, self.input_size])

        
        self.residual_val = tf.reduce_mean(tf.square(self.pde_func(self.X_f)))
        
        self.initial_loss = self.ic_func(self.X_i, self.u_i)  #Initial Loss 
        self.boundary_loss =  self.bc_func(self.X_b, self.u_b)  #Boundary Loss
        self.domain_loss = self.pde_func(self.X_f) #Domain Loss
        
        
        self.loss = tf.reduce_mean(tf.square(self.initial_loss)) + \
                    tf.reduce_mean(tf.square(self.boundary_loss)) + \
                    tf.reduce_mean(tf.square(self.domain_loss))
                    
        self.iteration = 1
        
        self.optimiser_GD = options.get_optimiser('GD', GD_opt)
        self.optimiser_QN = options.get_optimiser('QN', QN_opt, self.loss)
                    
        self.train_GD = self.optimiser_GD.minimize(self.loss)
        
        self.loss_list = []
        

        init = tf.compat.v1.global_variables_initializer()
#        self.saver = tf.train.Saver()
        self.sess.run(init)
        
       
        
    def ic_func(self, X, u):
        u_pred = self.forward(X, self.weights, self.biases)
        ic_loss = u_pred - u
        return ic_loss

    
    def bc_func(self, X, u):
        bc_loss = self.bc(self.forward, X, u, self.weights, self.biases)
        return bc_loss
        
    def pde_func(self, X):
        pde_loss = self.pde(self.forward, X, self.weights, self.biases)

        return pde_loss
    
    def callback_QN(self, loss):
        self.iteration += 1
        self.loss_list.append(loss)
        if self.iteration%10 ==0:
            print('Loss at iteration: ' + str(self.iteration) + " = " , loss)
            
    def callback_GD(self, it, train_input):
        elapsed = time.time() - self.init_time
        loss_value = self.sess.run(self.loss, train_input)
        self.loss_list.append(loss_value)
        print('It: %d, Loss: %.3e, Time: %.2f' % 
                  (it, loss_value, elapsed))
        self.init_time = time.time()
        
            
    def train(self, nIter, input_dict):
        
        nIter_2 = int(nIter/2)
        
        train_input = {self.X_i: input_dict['X_i'], self.u_i: input_dict['u_i'],
                       self.X_b: input_dict['X_b'], self.u_b: input_dict['u_b'],
                       self.X_f: input_dict['X_f']
                       }
        
        start_time = time.time()
        self.init_time = time.time()
        
        
        for it in range(nIter_2):
            self.sess.run(self.train_GD, train_input)
            
            # Print
            if it % 10 == 0:
                self.callback_GD(it, train_input)

        for it in range(nIter_2, nIter):
            if it %500 ==0:
                X_f = Sampler.str_sampler(self)
                train_input = {self.X_i: input_dict['X_i'], self.u_i: input_dict['u_i'],
                               self.X_b: input_dict['X_b'], self.u_b: input_dict['u_b'],
                               self.X_f: X_f
                               }       
                
            self.sess.run(self.train_GD, train_input)
            
            # Print
            if it % 10 == 0:
                self.callback_GD(it, train_input)

                                    


        X_f = Sampler.uniform_sampler(self)
        train_input = {self.X_i: input_dict['X_i'], self.u_i: input_dict['u_i'],
                       self.X_b: input_dict['X_b'], self.u_b: input_dict['u_b'],
                       self.X_f: X_f
                       }       
                                                                                              
        self.optimiser_QN.minimize(self.sess,  
                                feed_dict = train_input,         
                                fetches = [self.loss], 
                                loss_callback = self.callback_QN)
        
#        self.saver.save(self.sess, "model.ckpt")
                                    
        print("Total Training Time : {}".format(time.time() - start_time))
        
        return np.asarray(self.loss_list)
            
    def predict(self, X):
        u = self.sess.run(self.u, {self.X: X})
        return u 
                        
        
        
        