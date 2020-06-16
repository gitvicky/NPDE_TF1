#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:08:59 2020

@author: Vicky

Neural PDE - Tensorflow 1.14
Module : Boundary Conditions

"""

import tensorflow as tf 

def select(name):
    return {
        "Dirichlet": dirichlet,
        "Neumann": neumann,
        "Periodic": periodic
    }[name]
    

def dirichlet(forward, X, u, w, b):
    u_pred = forward(X, w, b)
    return u - u_pred


def neumann(forward, X, f, w, b): #Currently only for 1D
    u =  forward(X, w, b)
    u_X = tf.gradients(u, X)[0]
    
    return u_X[:, 1:2] - f


def periodic(forward, X, f, w, b): # Currently for only 1D
    n = int(len(X)/2)
    u =  forward(X, w, b)
    u_X = tf.gradients(u, X)[0]
    u_XX = tf.gradients(u_X, X)[0]
    
    return (u[:n] - u[n:]) + (u_X[:,1:2][:n] - u_X[:,1:2][:n]) + (u_XX[:,1:2][:n] - u_XX[:,1:2][:n])


        