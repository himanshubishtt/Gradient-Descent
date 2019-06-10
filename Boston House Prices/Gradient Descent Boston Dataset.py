#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[23]:


import numpy as np
data=np.genfromtxt(r"C:\Users\Home\Downloads\Testing with mn+1.csv",delimiter=",")
data.shape


# In[32]:


scaler.fit_transform(data[:,0:15])
#improves speed


# In[31]:


def step(points, learning_rate, m):
    m_slope = np.zeros(14)
    M = len(points)
    for i in range(M):
        ans=0
        x = points[i, 0:14]
        y = points[i, 14]
        for k in range(0,14):
            ans+=m[k]*x[k]
        for j in range(0,14):
            m_slope[j] += (-2/M)* (y - ans)*x[j]  
    new_m = m - learning_rate*m_slope
    return new_m


# In[30]:


def grad(points, learning_rate, num_iterations):
    m = np.zeros(14)
    for i in range(num_iterations):
        m= step(points, learning_rate, m )
        print(i, " Cost: ", cost(points, m))
    return m


# In[29]:


def cost(points, m):
    total_cost=0
    M = len(points)
    for i in range(M):
        ans=0
        x = points[i, 0:14]
        y = points[i, 14]
        for k in range(0,14):
            ans+=m[k]*x[k]
        total_cost += (1/M)*((y - ans)**2)  
    return total_cost


# In[35]:


data = np.genfromtxt(r"C:\Users\Home\Downloads\Testing with mn+1.csv", delimiter=",")
scaler.transform(data)
learning_rate = 0.15
num_iterations = 150
m = grad(data, learning_rate, num_iterations)
print(m)


# In[36]:


ypred=np.array([])
h=0
xtest=np.genfromtxt(r"C:\Users\Home\Downloads\0000000000002417_test_boston_x_test.csv",delimiter=",")
for i in range(127):
    for j in range(14):
        h+=m[j]*xtest[i][j]
    ypred=np.append(ypred,h)
    h=0
        
np.savetxt(r"C:\Users\Home\Desktop\predictions.csv",ypred,delimiter=",")


# In[ ]:




