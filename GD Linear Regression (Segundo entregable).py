#!/usr/bin/env python
# coding: utf-8

# # Gradient descent
# 
# ## Implementation

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


def GD(theta, X, Y):
    '''
    Updates an array of coefficients for a single iteration using regular Gradient Descent.
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
    
    total_error = 0
    theta = descent(theta, X, Y)
    print('Train Error: ' + str(total_error))
        
    return theta


# ## Dataset training

# In[7]:


#DB to be used
import pandas as pd
auto = pd.read_csv('Automobile.csv', na_values = '?')
auto.head()


# To decide a dependent variable to predict, we are going to pick one that holds a significant correlation (>0.7) with other atributes and choose them as independent variables. In this case, we will pick *wheel-base* as the dependent variable and *width*, *length* and *curb-weight* as independent variables.

# In[8]:


import seaborn as sns
plt.figure(figsize=(30,30))
sns.heatmap(auto.corr(), annot=True)


# In[9]:


# Setting domain and range
X_auto = auto[['length','width','curb-weight']]
Y_auto = auto['wheel-base']
X_auto


# In[10]:


from sklearn.model_selection import train_test_split

def setTrain(D, T, Xt, Yt, max_theta, state):
    '''
    Initializes all parameters via random values that generates a close representation of a multivariate linear function.
    Instantiates a normalized train and test set with a train size ratio of T
    
    Args:
    
        D -> Number of dimensions 
        T -> Proportion (percentage) of elements used for training
        Xt -> A bidimensional list with values that corresponds to the function domain
        Yt -> An uni-dimensional list that corresponds to the function range
        max_theta -> Maximum absolute value for a X(i) coefficient
        state -> Seed used for randomized entry selection for train and test sets
        
    Returns:
    
        None
    
    '''
    global theta
    global X
    global Y
    global Xtest
    global ytest
    global alpha
    global total_error
    
    theta = []
    X = []
    Y = []
    N = np.ceil(len(Xt) * T)
    total_error = 0
    alpha = 0.1
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xt, Yt, train_size=T, random_state=state)
    
    if D < 2:
        return
    
    for i in range(D):
        theta.append(random.uniform(-max_theta,max_theta))
        
    for entry in Xtrain:
        entry.append(1)    
        X.append(entry)
        
    for i in range(len(Xtest)):
        Xtest[i].append(1)
        
    Y = ytrain
    
    theta = np.array(theta)
    X = normalizeX(np.array(X))
    Xtest = normalizeX(np.array(Xtest))
    Y = normalize(np.array(Y))
    ytest = normalize(np.array(ytest))
    
    print('Theta (not for normalized data): ' + str(theta))
    print('X: ' + str(X))
    print('Y: ' + str(Y))


# We set the train and test sets using the values of X and Y picked before. In this case, *0.8* indicates that 80% of data will be included in the test and just 20% in the test set.

# In[11]:


setTrain(4, 0.8, X_auto.values.tolist(), Y_auto.values.tolist(), 10, 2)


# In[12]:


def run_test(epoch):
    '''
    Runs GD for a number of epochs, updating the set of coefficients and computing per-epoch testing error
    
    Args:
    
        epoch -> Number of iterations of Gradient Descent
        
    Returns:
    
        Final test error using the prediction given by the final values of the coefficients
    '''
    global Xtest
    global ytest
    global theta
    
    for i in range(epoch):
        print('Epoch ', i)
        theta = GD(theta,X,Y)
        
        accum = 0
        for j in range(len(Xtest)):
            accum += abs(h(theta, Xtest[j]) - ytest[j])
            
        print('Test total error: ', accum)
        
    
        
    print(theta)
    return accum


# It is shown in the cell below the test and training error. We can see that both decrease as epoch size increases at a steady rate. Nonetheless, the train error remains significantly lower than the test counterpart. This can be mitigated with cross-validation methods by varying the sections of data selected as train and test.

# In[ ]:


run_test(10000)

