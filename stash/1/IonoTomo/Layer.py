
# coding: utf-8

# In[1]:

import numpy as np

class Layer(object):
    def __init__(self,center,,cells,dx,dy,width,center):
        '''width=np.inf is a semi-infinite layer'''
        self.idx = idx
        self.cells = cells
        self.dx = dx
        self.dy = dy
        self.width = width
        self.center = center
        self.c0 = self.cells.shape[]
    
    def index2coords(self,xidx,yidx):
        self.c0[0] + xidx*self.dx
            
    def getRefractiveIndex(self,x,y):
        '''Get refractive index of a particular cell'''
        nx = np.int64(x/dx)
        ny = np.int64(y/dy)
        return self.cells[ny,nx]
    
    

