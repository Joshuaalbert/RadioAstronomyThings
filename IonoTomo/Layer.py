
# coding: utf-8

# In[1]:

import numpy as np

class Layer(object):
    def __init__(self,center,cells,dx,dy,width):
        '''width=np.inf is a semi-infinite layer'''
        self.cells = cells
        self.dx = dx
        self.dy = dy
        self.x = center[1]+np.linspace(-self.cells.shape[1]*dx/2.,self.cells.shape[1]*dx/2.,self.cells.shape[1])
        self.y = center[0]+np.linspace(-self.cells.shape[0]*dy/2.,self.cells.shape[0]*dy/2.,self.cells.shape[0])
        self.z = z
        self.center = center
    
    def index2coords(self,xidx,yidx):
        return self.x[xidx],self.y[yidx]
            
    def getRefractiveIndex(self,x,y):
        '''Get refractive index of a particular cell'''
        nx = np.int64(x/dx)
        ny = np.int64(y/dy)
        return self.cells[ny,nx]
    
    

