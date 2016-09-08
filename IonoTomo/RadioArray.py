
# coding: utf-8

# In[2]:

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import numpy as np

class RadioArray(object):
    def __init__(self,arrayFile = None,log = None,name = None,msFile=None,numAntennas=0,earthLocs=None,wavelength=0.21):
        self.log = log
        self.locs = []
        self.frames = []
        self.Nantenna = 0
        self.wavelength = wavelength
        if arrayFile is not None:
            self.arrayFile = arrayFile
            self.loadArrayFile(arrayFile)
    def loadArrayFile(self,arrayFile):
        '''Loads a csv where each row is x,y,z in geocentric coords of the antennas'''
        d = np.genfromtxt(arrayFile)
        self.locs = ac.SkyCoord(x=d[:,0]*au.m,y=d[:,1]*au.m,z=d[:,2]*au.m,frame='itrs')
        self.calcCenter()
        self.Nantenna = len(self.locs)

    def getFov(self,wavelength):
        '''get the field of view in radians. todo '''
        return 0.5*np.pi/180.
    
    def saveArrayFile(self,arrayFile):
        pass
    def loadMsFile(self,msFIle):
        '''Get antenna positions from ms, array name, frequency'''
        pass

    def calcFrames(self,times):
        '''Create the alt/az frames of the timestamps'''
        frames = {}
        for t in times:
            frames[t.isot] = ac.AltAz(obstime=t,location=self.getCenter(),obswl=self.wavelength*au.m)
        return frames
    
    def baselineIdx2Pair(self,index):
        '''Unique map from integer to pair'''
        j = int(index) % self.Nantenna
        i = (int(index) - j)/self.Nantenna
        return (i,j)

    def baselinePair2Idx(self,i,j):
        '''Unique pair to integer. i,j zero indexed.'''
        i_ = min(i,j)
        j_ = max(i,j)
        return int(i*(self.Nantenna+1) - i*(i+1)/2 + j)
    
    def lhs2rhs(self,xyzLhs):
        '''Takes a numpy array and swaps 1,0 indices'''
        xyzRhs = np.copy(xyzLhs)
        xyzRhs[:,1] = xyzLhs[:,0]
        xyzRhs[:,0] = xyzLhs[:,1]
        return xyzRhs
        
    def calcBaselines(self,times,pointing):
        '''Compute baselines in u,v,w axes. times is astropy.time array, and point is [ra,dec] in ICRS frame (not astropy object)'''
        self.pointing = ac.SkyCoord(ra=pointing[0]*au.deg,dec=pointing[1]*au.deg,frame='icrs')
        self.frames = self.calcFrames(times)
        self.baselineMap = {}#We map a unique integer to baseline pair
        for t in times:
            self.baselineMap[t.isot] = np.zeros([self.Nantenna*(self.Nantenna-1)/2 + self.Nantenna,3])
            s = self.pointing.transform_to(self.frames[t.isot]).cartesian.xyz#LHS
            i = 0
            count = 0
            while i < self.Nantenna:
                b = self.locs[i].cartesian.xyz.to(au.m).value.transpose() - self.locs[i:].cartesian.xyz.to(au.m).value.transpose()
                bmag = np.sum(b*b,axis=1)
                b[np.isnan(b)] = 0.
                uvw = b - np.outer(np.sum(b*s,axis=1),s)/np.outer(bmag,np.ones(3))
                uvw[np.isnan(uvw)] = 0.
                #self.baselineMap[t.isot][self.baselinePair2Idx(i,0):self.baselinePair2Idx(i,i+1),:] = self.lhs2rhs(uvw)#RHS so y is north, x is east, z is out
                self.baselineMap[t.isot][count:count+uvw.shape[0],:] = self.lhs2rhs(uvw)#RHS so y is north, x is east, z is out
                count += uvw.shape[0]
                i += 1
    def plotUV(self):
        '''Compare against Casa plotuv todo'''
        import pylab as plt
        for t in self.baselineMap.keys():
            print t
            plt.scatter(self.baselineMap[t][:,0],self.baselineMap[t][:,1])
        plt.show()
            

    def calcCenter(self):
        '''calculates the centroid of the array based on itrs array self.locs returns the SkyCoord of center in itrs frame'''
        self.center = ac.SkyCoord(*np.mean(self.locs.cartesian.xyz,axis=1),frame='itrs')
        self.log('Center is {0}'.format(self.center))
        self.arrayHeight = self.center.geocentrictrueecliptic.distance#from Earth's core
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
    times = at.Time([0,1,2,3,4],format='gps')
    radioArray.calcBaselines(times,[45,45])
    radioArray.plotUV()
    print radioArray.arrayHeight

