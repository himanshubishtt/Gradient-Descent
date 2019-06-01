#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
data=np.genfromtxt(r"C:\Users\Home\Downloads\0000000000002419_training_ccpp_x_y_train.csv",delimiter=",")
X=pd.DataFrame(data)
l=1
X.columns=["A","B","C","D","E"]
X["F"]=l
X["G"]=X["E"]
del X["E"]
X["A2"]=X["A"]**2
X["B2"]=X["B"]**2
X["D2"]=X["D"]**2
X["A3"]=X["A"]**3
X["F1"]=X["F"]
X["Output"]=X["G"]
del X["G"]
del X["F"]
X=X.values
X.shape


# In[14]:


def step(points, learning_rate, m):
    m_slope = np.zeros(9)
    M = len(points)
    for i in range(M):
        ans=0
        x = points[i, 0:9]
        y = points[i, 9]
        for k in range(0,9):
            ans+=m[k]*x[k]
        for j in range(0,9):
            m_slope[j] += (-2/M)* (y - ans)*x[j]
    new_m = m - learning_rate*m_slope
    return new_m


# In[15]:


def grad(points, learning_rate, num_iterations):
    m = np.zeros(9)
    for i in range(num_iterations):
        m= step(points, learning_rate, m )
        print(i, " Cost: ", cost(points, m))
    return m


# In[16]:


def cost(points, m):
    total_cost=0
    M = len(points)
    for i in range(M):
        ans=0
        x = points[i, 0:9]
        y = points[i, 9]
        for k in range(0,9):
            ans+=m[k]*x[k]
        total_cost += (1/M)*((y - ans)**2)  
    return total_cost


# In[54]:


learning_rate = 0.000000003
num_iterations = 5000
m = grad(X, learning_rate, num_iterations)
print(m)


# In[55]:


xtest=np.genfromtxt(r"C:\Users\Home\Downloads\0000000000002419_test_ccpp_x_test.csv",delimiter=",")
xtest=pd.DataFrame(xtest)
xtest.columns=["A","B","C","D"]
xtest["A2"]=xtest["A"]**2
xtest["B2"]=xtest["B"]**2
xtest["D2"]=xtest["D"]**2
xtest["A3"]=xtest["A"]**3
xtest["F1"]=1
xtest=xtest.values
ypred=np.array([])

h=0
for i in range(2392):
    for j in range(9):
        h+=m[j]*xtest[i][j]
    ypred=np.append(ypred,h)
    h=0
print(ypred)
np.savetxt(r"C:\Users\Home\Desktop\Assignment 3.csv",ypred,delimiter=",")


# In[38]:




