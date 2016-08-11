
# coding: utf-8

# In[ ]:

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import numpy as np

class RadioArray(object):
    def __init__(self,arrayFile = None,log = None,name = None,msFile=None,numAntennas=0,earthLocs=None):
        self.log = log
        self.locs = []
        if arrayFile is not None:
            self.arrayFile = arrayFile
            self.loadArrayFile(arrayFile)
    def loadArrayFile(self,arrayFile):
        '''Loads a csv where each row is x,y,z in geocentric coords of the antennas'''
        d = np.genfromtxt(arrayFile)
        i = 0
        locs = []
        while i < d.shape[0]:
            earthLoc = ac.SkyCoord(x=d[i,0]*au.m,y=d[i,1]*au.m,z=d[i,2]*au.m,frame='itrs')
            locs.append(earthLoc)
            i += 1
        self.addLocs(locs)
    def saveArrayFile(self,arrayFile):
        pass
    def loadMsFile(self,msFIle):
        '''Get antenna positions from ms, array name, frequency'''
        pass

    def addLocs(self,locs):
        '''Add antenna locations. Each location is an EarthLocation.'''
        i = 0
        while i < len(locs):
            self.locs.append(locs[i])
            i += 1
        self.calcCenter()
    def calcCenter(self):
        '''calculates the centroid of the array based on self.locs returns the EarthLocation of center'''
        r0 = np.array([0,0,0])*au.m
        i = 0
        while i < len(self.locs):
            locgc = self.locs[i].earth_location.geocentric
            r0 += np.array([locgc[0].to(au.m).value,
                              locgc[1].to(au.m).value,
                              locgc[2].to(au.m).value])*au.m
            i += 1
        r0 /= float(len(self.locs))
        self.center = ac.SkyCoord(x=r0[0],y=r0[1],z=r0[2],frame='itrs')
        self.log("Center of array: {0}".format(self.center))
        return self.center
    
    def getCenter(self):
        try:
            return self.center
        except:
            self.calcCenter()
            return self.center

