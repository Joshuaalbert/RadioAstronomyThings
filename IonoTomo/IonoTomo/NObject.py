
# coding: utf-8

# In[3]:

'''Describe the refractive index as a sum of basis functions with corresponding jacobians given.
The refractive index is described per layer.
'''
import numpy as np

class gaussianDecomposition(object):
    def __init__(self,params):
        self.x0 = params[:,0]
        self.y0 = params[:,1]
        self.z0 = params[:,2]
        self.a = params[:,3]
        self.bx = params[:,4]
        self.by = params[:,5]
        self.bz = params[:,6]    

    def compute_zeroth(self,x,y,z):
        '''Return the nearest.'''
        zero = 1. 
        i = 0
        while i < np.size(self.x0):
            zero += self.a[i]*np.exp(-(x-self.x0[i])**2/self.bx[i]**2-(y-self.y0[i])**2/self.by[i]**2-(z-self.z0[i])**2/self.bz[i]**2)
            i += 1
        return zero

class NObject(gaussianDecomposition):
    def __init__(self,params):
        super(NObject,self).__init__(params)
#    def __init__(self,data,x,y,z):
#        '''data is cartesian, but compute will be in spherical'''
#        super(NObject,self).__init__(data,x,y,z)
    def compute_n(self,x,y,z):
        #convert r,theta,phi to x,y,z
        #x,y,z = coordsSph2Cart(r,theta,phi)
        return self.compute_zeroth(x,y,z)
    
def noAtmosphere():
    x0 = [0]#*np.cos(c0.spherical.lon.rad+0.1)*np.sin(np.pi/2-c0.spherical.lat.rad)]
    y0 = [0.]#*np.sin(c0.spherical.lon.rad)*np.sin(np.pi/2-c0.spherical.lat.rad)]
    z0 = [0]#*np.cos(np.pi/2-c0.spherical.lat.rad)]
    a = [0.]
    bx=[1]
    by=[1]
    bz=[1]
    params = np.array([x0,y0,z0,a,bx,by,bz]).transpose()
    NObj = NObject(params)
    return NObj
    
if __name__ == '__main__':
    NObj = noAtmosphere()
    print ("N no atmostphere: x,y,z=0:",NObj.compute_n(0,0,0))
    


# In[ ]:



