#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:34:36 2020

@author: Vicky

Neural PDE - Tensorflow 1.14
Module : main

"""
import numpy as np

from . import training_graph

def setup(NN, NPDE, PDE, training_data, pde_func):
    
    input_size = NN['input_neurons']
    num_layers = NN['num_layers']
    num_neurons = NN['num_neurons']
    output_size = NN['output_neurons']
    
    lb = np.asarray(PDE['lower_range'])
    ub = np.asarray(PDE['upper_range'])
    BC = PDE['Boundary_Condition']

    layers = np.concatenate([[input_size], num_neurons*np.ones(num_layers), [output_size]]).astype(int).tolist() 
    
    
    activation = 'tanh'
    initialiser = 'Xavier'
    GD_opt = 'Adam'
    QN_opt = 'L-BFGS-B'
    
    model = training_graph.TrainingGraph(layers,
                                         lb, ub,
                                         activation, 
                                         initialiser,
                                         GD_opt,
                                         QN_opt, pde_func)
            
    
    return model, training_data 