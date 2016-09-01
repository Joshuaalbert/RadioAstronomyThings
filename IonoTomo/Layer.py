
# coding: utf-8

# In[1]:

import numpy as np

class Layer(object):
    def __init__(self,idx,cells,dx,dy,width):
        self.idx = idx
        self.cells = cells
        self.dx = dx
        self.dy = dy
        self.width = width
    def getRefractiveIndex(self,x,y):
        '''Get refractive index of a particular cell'''
        nx = np.int64(x/dx)
        ny = np.int64(y/dy)
        return self.cells[ny,nx]
    
    

