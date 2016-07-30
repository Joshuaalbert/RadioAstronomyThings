
# coding: utf-8

# In[ ]:

import numpy as np
import astropy.coordinates as ac
import astropy.units as au

class Layer(object):
    def __init__(self,radioArray,pointing,height,dr,D):
        '''radio array object, pointing is SkyCoord in ICRS, height is radial in km
        dr is the resolution of square cells
        D is size of one side of the square aperture'''
        self.processedTimes = []
        self.radioArray = radioArray
        self.pointing = pointing
        self.height = height
        self.dr = dr
        self.D = D
        
        
        theta = D/z.to(au.m).value
        N = np.ceil(D/dr)
        if (N % 2) == 0:
            N += 1 #make 2n+1
        longitudes = np.linespace(lon0.to(au.rad).value - theta/2.,
                                 lon0.to(au.rad).value +theta/2.,
                                 N)
        latitudes = np.linespace(lat0.to(au.rad).value - theta/2.,
                                 lat0.to(au.rad).value +theta/2.,
                                 N)
        
        
        
    def processed(self,time):
        if time in self.processedTimes:
            return True
        else:
            return False    

class Atmosphere(object):
    def __init__(self,layerHeights,radioArray,self.pointing):
        '''Defines an atmosphere box that can be turned into layer'''
        sortedLayers = []
        for layerHeight in layerHeights:
            if layerHeight not in sortedLayers:
                sortedLayers.append(layerHeight)
        self.layerHeights = np.sort(np.array(sortedLayers))
        #make layers which are spherically symmetric
        self.layers = {}
        #array centroid
        c0 = radioArray.getCenter()#ITRS frames
        #pointing ra, and dec plus distance
        i = 0
        while i < len(self.layerHeights):
            self.layer[i] = Layer(self.radioArray,
                                  self.pointing,
                                  self.getLayerHeight(i),
                                  self.getWavelength()*10,#resolution?
                                  1000.#size in m of layer? or Automatic with fresnel zone size
                                 ) 
            i += 1
        
    def setTime(self,time):
        '''Set the time of the atmosphere, time is astropy.time.Time'''
        self.time = time 
        
        

