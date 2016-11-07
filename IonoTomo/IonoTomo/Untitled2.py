
# coding: utf-8

# In[11]:

import numpy as np
import pylab as plt

N = 1000
F = np.zeros(N)
for i in range(10000):
    r = np.random.uniform(size=N)
    f = np.fft.fft(np.fft.ifftshift(r))
    f[0] = 0
    f = np.fft.fftshift(f)
    F += np.abs(f)
plt.plot(F)
plt.show()


# In[ ]:



