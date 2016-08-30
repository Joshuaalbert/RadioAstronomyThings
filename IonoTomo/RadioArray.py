
# coding: utf-8

# In[4]:

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import numpy as np

class RadioArray(object):
    def __init__(self,arrayFile = None,log = None,name = None,msFile=None,numAntennas=0,earthLocs=None):
        self.log = log
        self.locs = []
        self.Nantenna = 0
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
    def getFov(self,wavelength):
        '''get the field of view in radians'''
        return 0.5*np.pi/180.
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
        self.Nantenna = len(self.locs)
    def calcBaselines(self,times,pointing):
        self.restBaselines = []
        self.baselineMap = {}
        count = 0
        i = 0
        while i < self.Nantenna:
            j = i + 1
            while j < self.Nantenna:
                u = self.locs[j].itrs.earth_location.geocentric[0].to(au.m).value - self.locs[i].itrs.earth_location.geocentric[0].to(au.m).value
                v = self.locs[j].itrs.earth_location.geocentric[1].to(au.m).value - self.locs[i].itrs.earth_location.geocentric[1].to(au.m).value
                w = self.locs[j].itrs.earth_location.geocentric[2].to(au.m).value - self.locs[i].itrs.earth_location.geocentric[2].to(au.m).value
                bl = [u,v,w]
                self.baselineMap[i] = {j:count}
                count += 1
                self.restBaselines.append(bl)
                j += 1
            i += 1
        self.restBaselines = np.array(self.restBaselines)
        #make baselines over the course of the observation, needs times and pointings
        self.baselines = {}
        i = 0
        while i < len(times):
            #trackingCenter
            #self.baselines[times[i].isot] = self.restBaselines*(1-np.dot(self.restBaselines,trackingCenter))
            i += 1

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
        #n = self.center.itrs.earth_location.geocentric.to(au.m).value
        #self.n = n/np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        return self.center
    
    def getCenter(self):
        try:
            return self.center
        except:
            self.calcCenter()
            return self.center

if __name__=='__main__':
    from Logger import Logger
    logger = Logger()
    radioArray = RadioArray(arrayFile='arrays/gmrtPos.csv',log=logger.log)
    print radioArray.center.earth_location.height

