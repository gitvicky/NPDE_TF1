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


from network import Network
#import boundary_conditions
import options

class TrainingGraph(Network):
    def __init__(self, layers, lb, ub, activation, initialiser,GD_opt, QN_opt):
        
        super().__init__(layers, lb, ub, activation, initialiser)
        
        # Initialize NNs
        self.weights, self.biases = self.initialize()
        
        # Creating the placeholders for the tensorflow variables
        self.sess = tf.compat.v1.Session()
        
        self.t = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u = self.forward(tf.concat([self.t, self.x], 1), self.weights, self.biases)
                
        self.t_i = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_i = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u_i = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        
        self.t_b = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_b = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u_b = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        self.t_f = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) 
        self.x_f = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        
        
        
        self.initial_loss = self.ic_func(self.t_i, self.x_i, self.u_i)  #Initial Loss 
        self.boundary_loss =  self.bc_func(self.t_b, self.x_b, self.u_b)  #Boundary Loss
        self.domain_loss = self.pde_func(self.t_f, self.x_f)
        
        
        self.loss = tf.reduce_mean(tf.square(self.initial_loss)) + \
                    tf.reduce_mean(tf.square(self.boundary_loss)) + \
                    tf.reduce_mean(tf.square(self.domain_loss))
                    
        self.iteration = 1
        
#        self.optimiser_GD = options.get_optimiser('GD', GD_opt)
#        self.optimiser_QN = options.get_optimiser('QN', QN_opt, self.loss)
                    
        self.optimiser_GD = tf.compat.v1.train.AdamOptimizer()
        self.train_GD = self.optimiser_GD.minimize(self.loss)
        
        self.optimiser_QN = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                   method = 'L-BFGS-B', 
                                                                   options = {'maxiter': 50000,
                                                                              'maxfun': 50000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol' : 1.0 * np.finfo(float).eps})
                
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        
       
        
    def ic_func(self, t, x, u):
        u_pred = self.forward(tf.concat([t, x],1), self.weights, self.biases)
        ic_loss = u_pred - u
        return ic_loss

    
    def bc_func(self, t, x, u):
        u_pred = self.forward(tf.concat([t, x],1), self.weights, self.biases)
        bc_loss = u_pred - u 
        return bc_loss
    
    def pde_func(self, t, x):
        u = self.forward(tf.concat([t,x],1), self.weights, self.biases)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        pde_loss = u_t + u*u_x - 0.1*u_xx
        
        return pde_loss
    
    def callback_QN(self, loss):
        self.iteration += 1
        if self.iteration%10 ==0:
            print('Loss at iteration: ' + str(self.iteration) + " = " , loss)
            
    def train(self, nIter, input_dict):
        
        train_input = {self.t_i: input_dict['t_i'], self.x_i: input_dict['x_i'], self.u_i: input_dict['u_i'],
                       self.t_b: input_dict['t_b'], self.x_b: input_dict['x_b'], self.u_b: input_dict['u_b'],
                       self.t_f: input_dict['t_f'], self.x_f: input_dict['x_f']
                       }
        
        start_time = time.time()
        init_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_GD, train_input)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - init_time
                loss_value = self.sess.run(self.loss, train_input)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                init_time = time.time()
                                                                                                                          
        self.optimiser_QN.minimize(self.sess,  
                                feed_dict = train_input,         
                                fetches = [self.loss], 
                                loss_callback = self.callback_QN)
        
                                    
        print("Total Training Time : {}".format(time.time() - start_time))
        
            
    def predict(self, X):
        u = self.sess.run(self.u, {self.t: X[:,0:1] , self.x: X[:,1:2]})  
        return u 
                        
        
        
        