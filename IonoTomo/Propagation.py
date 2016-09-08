
# coding: utf-8

# In[1]:

import numpy as np
from Logger import Logger
from Layer import Layer

class Propagation(object):
    '''Defines propagation along l,m forwards and backwards'''
    def __init__(self,l,m,wavelength,log=None):
        if log is None:
            self.log = Logger().log
        else:
            self.log = log
        self.l = l
        self.m = m
        self.n = np.sqrt(1j - l**2 - m**2)
        self.wavelength = wavelength#m
        #snells law
        if l != 0:
            self.theta0 = np.arccot(l/m)
        else:
            self.theta0 = 0.#arbitrary
        if l**2 + m**2 > 1.:
            self.log("Complex phi0 -> evanescent waves")
            self.phi0 = np.sqrt(1j - l**2 - m**2)
        self.phi0 = np.sqrt(1 - l**2 - m**2)
        self.nsin0 = 1.*np.sin(self.phi0)#constant real number
        self.setDiffractiveScale()
    
    def setLayers(self,layers):
        self.layers = layers
    
    def setDiffractiveScale(self,scale=2000.):
        '''set diffractive scale in m'''
        self.diffScale = scale#m
        
    def calcFresnelZone(self,layerWidth,wavelength,n=1):
        '''return in radians max of sqrt(n lambda d1 d2/(d1+d2))'''
        return np.sqrt(n*layerWidth*wavelength)/2.#m
    
    def spherical2cartesian(self,theta,phi,r=1):
        x = r*np.cos(theta)*np.sin(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(phi)
    
    def calculateWaveVector(self,layerIdx,x,y):
        kz = 2*np.pi*self.layers[layerIdx].
    def propagateForward(self, x,y,z):
        '''
        Propagate sky component forward from source to final layer. Initial layer is semi-infinite.
        '''
        k = self.getForwardWaveVector()
        layerIdx = len(self.layers) - 1
        apLoc = 
        Mtotinv = np.array([[1,0],[0,1]],dtype=type(1j))
        while layerIdx >= 0:
            #get cells = 
            #at top of layer
            if layerIdx == len(self.layers):
                deltan = 0
            else:
                refractiveIdx = self.layers
                deltan = 2*np.pi/self.wavelength*refractiveIdx
            #get fresnel zone
            fresnelZone = self.calcFresnelZone(self.layers[layerIdx].width,self.wavelength)
            #get resolution scale -> half diffScale
            Ncell = fresnelZone/self.diffScale*2.
            cells = np.linspace(-fresnelZone/2.,fresnelZone/2.,Nx)
            Cells = np.meshgird(cells,cells)
            #get refractive 
            
        
        
        
    
    
""" fourier plane method (unstable)

'''Propagation from one layer to the next. '''
import numpy as np

def fft(A):
    '''exp{-i2pi (lu + mv)}'''
    return np.fft.fftshift(np.fft.fftn(np.ftt.ifftshift(A)))

def ifft(A):
    '''exp{i2pi (lu + mv)}'''
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(A)))

class Propagation(object):
    '''
    Each layer is formed of an array of cells with refractive index.
    The interfaces are optically thin.
    '''
    def __init__(self,w):
        self.w = w
    def setAperture(self,U,u,v):
        self.U = U
        
        self.A = fft(U)

        self.u = u
        self.v = v
    def R2propKernel(self,UVmax2):
        D = UVmax2/self.w**2
        pa = 1
        pb = D+1
        return 2.*np.pi*self.w*3./16.*(-2./3.*(D+1)**2* (pb**(-3./2.) - pa**(-3./2.)) + 4.*(D+1)*(pb**(-1./2.) - pa**(-1./2.)) + 2.*(pb**(1./2.) - pa**(1./2.)))
    def solveAperturePartition(self):
        '''Find UVmax such that lagrange remainder in phase is less than 1 radian.'''
        a0 = 2*np.pi*self.w
        p = [a0/8., 0., -a0*3./4., a0, -a0*3./8. - 1]
        roots = np.roots(p)
        i = 0
        while i < 4:
            if np.imag(roots[i]) == 0 and np.real(roots[i]) > 0:
                return np.real(np.sqrt(self.w**2*(roots[i]**2 - 1)))
            i += 1
        return np.real(np.sqrt(self.w**2*(roots[1]**2 - 1))) #typically second root when solved by numpy
         
    def plotR2propKernel(self):
        import pylab as plt
        sol = self.solveAperturePartition()

        UVmax = np.linspace(10,1000,100000)
        r2 = self.R2propKernel(UVmax**2)
        plt.plot(UVmax,r2)
        plt.plot([sol,sol],[np.min(r2),1.],ls='--',c='black')
        plt.plot([np.min(UVmax),sol],[1.,1.],ls='--',c='black')
        plt.xlabel('Aperture size (lambda)')
        plt.ylabel('Phase error w={0}'.format(self.w))
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
       
    def propagate(self):
        '''Piecewise propagate'''
        UVmax = self.solveAperturePartition()
        Nu = np.ceil((self.u[-1] - self.u[0])/UVmax*2 - 1)#minimum pieces with overlap
        Nv = np.ceil((self.v[-1] - self.v[0])/UVmax*2 - 1)
        u0 = np.linspace(self.u[0],self.u[-1],int(Nu))
        v0 = np.linspace(self.v[0],self.v[-1],int(Nv))
        #coords of recieving aperture
        U,V = np.meshgrid(self.u,self.v)
        Uprop = np.zeros_like(U)
        #Fourier parameters
        l = np.fft.fftshift(np.fft.fftfreq(len(self.u),d = np.abs(self.u[1] - self.u[0])))
        m = np.fft.fftshift(np.fft.fftfreq(len(self.v),d = np.abs(self.v[1] - self.v[0])))
        L,M = np.meshgrid(l,m)
    
        propKernel1 = np.exp(1j*2.*np.pi*self.w*(1 + (U**2 + V**2)/self.w**2))
        propKernel2fourier = 1j/2.*self.w*np.exp(-1j*2.*np.pi/self.w*(U**2 + V**2 + self.w*(L*U + M*V) + (L**2 + M**2)*self.w**2/4.))
        plt.imshow(np.angle(propKernel2fourier))
        plt.colorbar()
        plt.show()
        #weight factor to do with R2, angular powerspectrum, prop
        Uprop = -1j*propKernel1*ifft(self.A*propKernel2fourier)
        return Uprop
                
                

if __name__=='__main__':
    #P = Propagation(100)
    #P.plotR2propKernel()
    import pylab as plt
    sols = []
    for w in np.linspace(10,100000,250):
        P = Propagation(w)
        sols.append(P.solveAperturePartition())
        print("w {0} -> uvmax {1}".format(w,sols[-1]))
    plt.plot(sols,np.linspace(10,100000,250))
    plt.xlabel('Aperture size (lambda)')
    plt.ylabel(r'Minimum $w$')
    plt.show()
    P = Propagation(100)
    M = 1
    x0 = np.random.uniform(low=-50,high=50,size=M)
    y0 = np.random.uniform(low=-50,high=50,size=M)
    p0 = np.random.uniform(size=M)

    wavelength = 1.
    N=100
    print('number:{0}'.format(N))
    x = np.linspace(-100,100,N)
    dx = np.abs(x[1]-x[0])
    X,Y = np.meshgrid(x,x)
    Usky = np.zeros([N,N])
    for xi,yi,pi in zip(x0,y0,p0):
        Usky += pi*np.exp(-((X-xi)**2 + (Y-yi)**2)/(2*dx)**2)
    P.setAperture(Usky,x,x)
    Uprop = P.propagate()
    """
        

