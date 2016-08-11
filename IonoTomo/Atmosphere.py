
# coding: utf-8

# In[6]:

import numpy as np
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at


def fft2(A):
    '''Performs 2D fft then shifts'''
    return np.fft.fftshift(np.fft.fft2(A))

def fftfreq(n,d=1):
    return np.fft.fftshift(np.fft.fftfreq(n,d=d))

def ifft2(A):
    '''Assumes A has shifted frequency to center.
    Unshifts and applies ifft'''
    return np.fft.ifft2(np.fft.ifftshift(A))

class Atmosphere(object):
    def __init__(self,radioArray,boxSize,times,log=None,**args):
        '''Radio Array object, boxSize (EastWest(m),NorthSouth(m),z(m))
        timeslices utc at.Time object, logger object
        '''
        #units
        self.tecu = 1e16
        #physical constants
        self.eCharge = 1.60217662e-19#C = F.V = W.s/V^2.V = kg.m^2/s^2/V
        self.epsilonPerm = 8.854187817e-12#F/m = kg.m/s^2/V^2
        self.eMass = 9.10938215e-31#kg
        
        if log is not None:
            self.log = log
        else:
            self.log = radioArray.log
        self.radioArray = radioArray#object that has center info
        self.xdim = boxSize[0]#units in m
        self.ydim = boxSize[1]#units in m
        self.zdim = boxSize[2]#units in m
        self.times = times#at.Time objects
        self.initialize(**args)
    
    def initialize(self,**args):
        '''Any special things for atmosphere come in args'''
        self.xangleres = 1./3600.*np.pi/180.# 60 arcsec = 1 arcmin in rad
        self.yangleres = 1./3600.*np.pi/180.# 60 arcsec = 1 arcmin in rad
        self.zres = 1000.#1000m = 1km
        try:
            self.wavelength = args['wavelength']
        except:
            self.wavelength = 0.21
        self.defineBox()
        self.defineCells()
        #should be ready to simulate after this finishes with a call to run()
        
    def defineBox(self):
        '''defines the box given a radio array center'''
        self.center = self.radioArray.getCenter()#itrs frame
        self.arrayHeight = self.center.geocentrictrueecliptic.distance.to(au.m).value
        el = self.center.earth_location.geodetic
        #wrap a box around the earth (this is bottom of box)
        self.xangle = self.xdim/self.arrayHeight
        self.yangle = self.ydim/self.arrayHeight
        self.Nxangle = int(np.ceil(self.xangle/self.xangleres))
        self.Nyangle = int(np.ceil(self.yangle/self.yangleres))
        self.Nz = int(np.ceil(self.zdim/self.zres))
        #defines the cell centers then there is 1 more edge 
        self.longitudes = np.linspace(el[0].deg - self.xangle/2.*180./np.pi,el[0].deg + self.xangle/2.*180./np.pi,self.Nxangle)
        self.latitudes = np.linspace(el[1].deg - self.yangle/2.*180./np.pi,el[1].deg + self.yangle/2.*180./np.pi,self.Nyangle)
        self.heights = np.linspace(self.arrayHeight, self.arrayHeight + self.zdim,self.Nz)
        #make the mesh each array is xres x yres x zres
        Lon, Lat, Hei = np.meshgrid(self.longitudes,self.latitudes,self.heights)
        #earth locs and then to itrs frame
        self.earthLocs = ac.EarthLocation(lon = Lon*au.deg,lat = Lat*au.deg,height=Hei*au.m)
        self.itrsLocs = ac.SkyCoord(*self.earthLocs.geocentric,frame='itrs')
        #now we have a spherically symmetric set of coords, and a simulation can populate the cells as a function of time
        
    def defineCells(self):
        '''define cells of electron density (which is all we need)'''
        self.cells = {}
        i = 0
        while i < len(self.times):
            self.cells[i] = {}
            #enables getting alt and az of sources and cells, they won't move for itrs frame things but will for icrs
            self.cells[i]['frame'] = ac.AltAz(obstime=self.times[i],location=self.radioArray.getCenter(),obswl=self.wavelength*au.m)
            #for now store all in memory, but could replace with an hdf5 location
            #also enables simply loading a precomputed siulation
            self.cells[i]['electronDensity'] = np.zeros([self.Nxangle,self.Nyangle,self.Nz])
            i += 1
    
    def getLayerWidth(self):
        '''Get the layer width in m (this is not layer from radioArray to pointing but the atmospheric layer)'''
        return self.zdim/self.Nz

    def Kolmogorov(self, Q, r0,alpha ):
        '''Kolmogorov turbulence says '''
        return 0.023 * (Q*r0+(1e-15))**(-alpha)

    def vonKarman(self, Q, r0, L0, l0 ,alpha):
        '''min scale r0, outer scale l0'''
        return 0.0299 * (r0)**(-alpha)/( Q**2 + L0**-2 )**((alpha + 1.)/2.) * numpy.exp(-Q**2 * l0**2 )
    
    def run(self):
        '''fill in cells'''
        #Let's put a simple simulation with a bulk layer at 350km and a turbluence layer 
        # at 600km
        tecuZenithNight = 1e16/1000e3/self.tecu#electrons m^-3 when the sun is down
        tecuZenithDay = tecuZenithNight*10.
        i = 0
        while i < len(self.times):
            #put cells in altaz frame
            #AltAzCells = self.itrsLocs.transform_to(self.cells[i]['frame'])
            sunLoc = ac.get_body('Sun',self.times[i],location = self.radioArray.getCenter().earth_location)
            AltAzSun = sunLoc.transform_to(self.cells[i]['frame'])
            #due to sun the ne is:
            bulkTec = 0.5*(tecuZenithNight + tecuZenithDay)*np.cos(AltAzSun.alt.deg*np.pi/180.) + 0.5*(tecuZenithDay - tecuZenithNight)*np.sin(AltAzSun.alt.deg*np.pi/180.)
            #spatially located around 350km +- 200km
            l = 0
            while l < self.Nz:
                self.cells[i]['electronDensity'][:,:,l] = bulkTec*np.exp(-(self.heights[l] - self.arrayHeight - 350.*1000.)**2/(200.*1000.)**2)
                l += 1              
            #turbluent layer around 600km for fun use 
            gammaDay = 1.
            gammaNight = np.random.uniform(low=2*0.69,high=1.5)#intema 2009
            gamma = 0.5*(gammaDay + gammaNight)*np.cos(AltAzSun.alt.deg*np.pi/180.) + 0.5*(gammaDay - gammaNight)*np.sin(AltAzSun.alt.deg*np.pi/180.)
            alpha = gamma + 2.
            #which layer is turbulent
            turbLayerIdx = np.argmin((self.heights - self.arrayHeight - 600.*1000.)**2)
            #height of layer from earth center to get size of cell in meters
            turbHeight = self.heights[turbLayerIdx]
            xCellSize = turbHeight * self.xangleres
            yCellSize = turbHeight * self.yangleres
            qx = fftfreq(self.Nxangle,d=xCellSize)
            qy = fftfreq(self.Nyangle,d=yCellSize)
            Qx, Qy = np.meshgrid(qx,qy)
            Q = np.sqrt(Qx**2 + Qy**2)
            r0 = 500.#minimal scale size in m
            #Pne = |FFT(ne)|^2
            Pne = np.random.uniform(size=[self.Nxangle,self.Nyangle])*self.Kolmogorov(Q,r0,alpha)
            import pylab as plt
            #plt.imshow(Pne)
            #plt.show()
            print fft2(np.sqrt(Pne))
            turbLayer = np.abs(fft2(np.sqrt(Pne)))
            turbLayerNe = tecuZenithNight*turbLayer/np.mean(turbLayer)
            self.cells[i]['electronDensity'][:,:,turbLayerIdx] += turbLayerNe
            i += 1

if __name__=='__main__':
    #test cases
    from Logger import Logger
    from RadioArray import RadioArray
    log = Logger().log
    radioArray = RadioArray("arrays/gmrtPos.csv",log=log)
    times = at.Time(np.linspace(0,10000,10),format='gps',scale='utc')
    A = Atmosphere(radioArray=radioArray,boxSize=(10000,10000,10000),times=times,log=log,wavelength=1.)
    A.run()


# In[ ]:



