
# coding: utf-8

# In[ ]:


from rathings import tec_solver
import h5py
import pylab as plt
import numpy as np

data = "../../../goods-n.hdf5"

h = h5py.File(data,'r')
freqs = h['freq'][...]

def print_attrs(name, obj):
    if "facet_patch" not in name:
        return
    print("Solving {}".format(name))
    phase = obj['phase'][:,:,0]
    phase[phase==0] = np.nan
    ni = phase.shape[0]
    f = plt.figure(figsize=(7*4,7*4))
    for i in range(ni):
        m = tec_solver.l1data_l2model_solve(np.reshape(phase[i,:],(len(freqs),1)),freqs,5)
        print(m)


h.visititems(print_attrs)


# In[ ]:




