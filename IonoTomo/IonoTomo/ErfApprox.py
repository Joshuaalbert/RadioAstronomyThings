
# coding: utf-8

# In[15]:

import numpy as np
from scipy.special import erf as serf

def erf(x):
    return serf(x)

def erfApptanh(x):
    return np.tanh(np.log(2)*np.sqrt(np.pi)*x)

import pylab as plt
x = np.linspace(-4,4,100)
y1 = erf(x)
y2 =erfApptanh(x)
plt.plot(x,y1,label='erf')
plt.plot(x,y2,label='erfAppTanh')
plt.show()


# In[12]:

print(erf(3/np.sqrt(2)))


# In[14]:

np.sqrt(2)


# In[ ]:



