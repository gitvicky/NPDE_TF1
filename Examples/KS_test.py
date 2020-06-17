#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 22:56:34 2020

@author: Vicky



Neural PDE - Tensorflow 1.14
Testing with the Kuramoto-Sivashinsky

IC: u(0, x) = −sin(πx/10)
BC: Periodic 


"""

import numpy as np 
import tensorflow as tf 
from matplotlib import pyplot as plt 
import scipy.io
from pyDOE import lhs

import os 
npde_path = os.path.abspath('..')
#npde_path = npde_path + '/Neural_PDE'

import sys 
sys.path.insert(1, npde_path) 


import Neural_PDE as npde
# %%
#Neural Network Hyperparameters
NN_parameters = {
                'input_neurons' : 2,
                'output_neurons' : 1,
                'num_layers' : 4,
                'num_neurons' : 64
                }


#Neural PDE Hyperparameters
NPDE_parameters = {'Sampling_Method': 'Random',
                   'N_initial' : 200, #Number of Randomly sampled Data points from the IC vector
                   'N_boundary' : 300, #Number of Boundary Points
                   'N_domain' : 20000 #Number of Domain points generated
                  }


#PDE 
PDE_parameters = {'Equation': 'u_t + u*u_x + u_xx + u_xxxx', 
                  'order': 2,
                  'lower_range': [0.0, -10.0], #Float [t, x]
                  'upper_range': [50.0, 10.0], #Float [t, x]
                  'Boundary_Condition': "Dirichlet",
                  'Boundary_Vals' : None,
                  'Initial_Condition': None,
                  'Initial_Vals': None
                 }


def pde_func(forward, X, w, b):
    t = X[:, 0:1]
    x = X[:, 1:2]
        
    u = forward(tf.concat([t,x],1), w, b)
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_xxx = tf.gradients(u_xx, x)[0]
    u_xxxx = tf.gradients(u_xxx, x)[0]  

    pde_loss = u_t + u*u_x + u_xx + u_xxxx
    
    return pde_loss

# %%

N_f = NPDE_parameters['N_domain']
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']

data = scipy.io.loadmat('/Users/Vicky/Documents/Code/NPDE_TF1/Data/KS.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None])) #Flattened array with the inputs  X and T 
u_star = Exact.flatten()[:,None]              

# Domain bounds
lb = X_star.min(0) #Lower bounds of x and t 
ub = X_star.max(0) #Upper bounds of x and t
    
X_i = np.hstack((T[0:1,:].T, X[0:1,:].T)) #Initial condition value of X (x=-1....1) and T (t = 0) 
u_i = Exact[0:1,:].T #Initial Condition value of the field u

X_lb = np.hstack((T[:,0:1], X[:,0:1])) #Lower Boundary condition value of X (x = -1) and T (t = 0...0.99)
u_lb = Exact[:,0:1] #Bound Condition value of the field u at (x = 11) and T (t = 0...0.99)
X_ub = np.hstack((T[:,-1:], X[:,-1:])) #Uppe r Boundary condition value of X (x = 1) and T (t = 0...0.99)
u_ub = Exact[:,-1:] #Bound Condition value of the field u at (x = 11) and T (t = 0...0.99)

X_b = np.vstack((X_lb, X_ub))
u_b = np.vstack((u_lb, u_ub))

X_f = lb + (ub-lb)*lhs(2, N_f) #Factors generated using LHS 

idx = np.random.choice(X_i.shape[0], N_i, replace=False)
X_i = X_i[idx, :] #Randomly Extract the N_u number of x and t values. 
u_i = u_i[idx,:] #Extract the N_u number of field values 

idx = np.random.choice(X_b.shape[0], N_b, replace=False)
X_b = X_b[idx, :] #Randomly Extract the N_u number of x and t values. 
u_b = u_b[idx,:] #Extract the N_u number of field values 



training_data = {'X_i': X_i, 'u_i': u_i,
                'X_b': X_b, 'u_b': u_b,
                'X_f': X_f}

# %%

model, input_dict = npde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters, training_data, pde_func)

nIter  = 5000
# %%

model.train(nIter, input_dict)


# %%

u_pred = model.predict(X_star)
u_pred = np.reshape(u_pred, np.shape(Exact))

def moving_plot(u_actual, u_sol):
    actual_col = '#302387'
    nn_col = '#DF0054'
    
    plt.figure()
    plt.plot(0, 0, c = actual_col, label='Actual')
    plt.plot(0, 0, c = nn_col, label='NN', alpha = 0.5)
    plt.legend()
    
    for ii in range(len(t)):
        plt.plot(x, u_actual[ii], c = actual_col)
        plt.plot(x, u_sol[ii], c = nn_col)
        plt.pause(0.01)
        plt.clf()
        
moving_plot(Exact, u_pred)
