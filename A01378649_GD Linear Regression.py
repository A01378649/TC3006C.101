#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Octavio Andrick Sánchez Perusquia                                            A01378649
import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


#Dummy

import pandas as pd

def normalizeX(arr):
    '''
    Normalizes a 2D array by column using min-max method
    
    Args:
        
        arr -> Array to br normalized
        
    Returns:
    
        A normalized 2D array
    '''
    data = pd.DataFrame(arr, index=None)
    print(data.shape)
    for i in range(data.shape[1]-1):
        data.iloc[:,i] = (data.iloc[:,i] - data.iloc[:,i].min())/(data.iloc[:,i].max() - data.iloc[:,i].min())
        
    return data.to_numpy()

def normalize(arr):
    '''
    Normalizes a 1D array using min-max method
    
    Args:
        
        arr -> Array to br normalized
        
    Returns:
    
        A normalized 1D array
    '''
    return (arr - arr.min())/(arr.max() - arr.min())

def setTest(D, N, max_X ,max_theta, max_noise):
    '''
    Initializes all parameters via random values that generates a close representation of a multivariate linear function
    
    Args:
    
        D -> Number of dimensions 
        N -> Number of samples
        max_X -> Maximum absolute value for an element in the domain space
        max_theta -> Maximum absolute value for a X(i) coefficient
        max_noise -> Maximum absolute value for a scalar that deviates the value of the dependent variable
        
    Returns:
    
        None
    
    '''
    global theta
    global X
    global Y
    global alpha
    global total_error
    
    theta = []
    X = []
    Y = []
    total_error = 0
    alpha = 0.1
    
    if D < 2:
        return
    
    for i in range(D):
        theta.append(random.uniform(-max_theta,max_theta))
        
    for i in range(N):
        entry = []
        for j in range(D-1):
            entry.append(random.uniform(-max_X,max_X))
            
        entry.append(1)
        X.append(entry)
        Y.append(np.dot(X[i],theta) + random.uniform(-max_noise,max_noise))
    
    theta = np.array(theta)
    X = normalizeX(np.array(X))
    Y = normalize(np.array(Y))
    
    print('Theta (not for normalized data): ' + str(theta))
    print('X: ' + str(X))
    print('Y: ' + str(Y))
    
setTest(3, 50, 10, 10, 10)


# In[3]:


def h(theta, x):
    '''
    Computes an evaluation of a linear function given a uni-dimensional or multi-dimensional point
    
    Args:
    
        theta -> Array of coefficients to be used per dimension
        x -> Vector representation of the point of evaluation
        
    Returns:
    
        An evaluation of h(x)
    '''
    return sum(theta * x)


# In[4]:


def update(theta, j, X, Y):
    '''
    Computes the partial derivative of a point in the domain and adjusts the corresponding coefficient based on its value
    
    Args:
    
        theta -> Array of coefficients to be used per dimension
        j -> the current dimension corresponding to the coefficient to be updated
        X -> Array of points in the domain
        Y -> True values of evaluation for each point
        
    Returns:
    
        An updated value of theta
    '''
    global total_error
    acum = 0
    for i in range(len(X)):
        error = (h(theta, X[i]) - Y[i]) * X[i][j]
        acum += error
        total_error += error
        
    return theta[j] - (alpha/len(X)*acum)


# In[5]:


def descent(theta, X, Y):
    '''
    Computes an iteration to update all the current values of theta simultaneously
    
    Args:
    
        theta -> Array of coefficients to be used per dimension
        X -> Array of points in the domain
        Y -> True values of evaluation for each point
        
    Returns:
    
        An array that contains an updated value for each previous theta
    '''
    T = []
    for j in range(len(theta)):
        T.append(update(theta, j, X, Y))
        
    return T


# In[6]:


def GD(theta, X, Y, N):
    '''
    Updates an array of coefficients for a number of iterations using regular Gradient Descent.
    Prints accumulated error per iteration
    
    Args:
    
        theta -> Array of coefficients to be used per dimension
        X -> Array of points in the domain
        Y -> True values of evaluation for each point
        N -> Number of iterations
        
    Returns:
    
        An array of coefficients given a number of iterations of GD
    '''
    global total_error
    
    for i in range(N):
        total_error = 0
        theta = descent(theta, X, Y)
        print('Error: ' + str(total_error))
        
    return theta


# In[7]:


print(GD(theta,X,Y,10000))

