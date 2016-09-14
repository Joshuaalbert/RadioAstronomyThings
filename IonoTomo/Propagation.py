
# coding: utf-8

# In[ ]:

import numpy as np
from Logger import Logger
from Layer import Layer
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at
from scipy.integrate import ode
from scipy.interpolate import UnivariateSpline

def fft(A):
    return np.fft.fftshift(np.ftt.fftn(np.fft.ifftshift(A)))

def ifft(A):
    return np.fft.fftshift(np.ftt.ifftn(np.fft.ifftshift(A)))

def ITRS2Frame(theta,phi):
    s1,s2 = np.sin(theta),np.sin(phi)
    c1,c2 = np.cos(theta),np.cos(phi)
    return np.array([[s1,c1,0],
                  [c1,-s1,0],
                  [0,0,1]]).dot(np.array([[c2,s2,0],
                                         [0,0,1],
                                         [-s2,c2,0]]))
def Frame2ITRS(theta,phi):
    s1,s2 = np.sin(theta),np.sin(phi)
    c1,c2 = np.cos(theta),np.cos(phi)
    return np.array([[c2,s2,0],
                  [s2,-c2,0],
                  [0,0,1]]).dot(np.array([[s1,c1,0],
                                         [0,0,-1],
                                         [c1,-s1,0]]))

def Frame2Frame(theta0,phi0,theta,phi):
    '''Rotate frames from those theta, phi to those at theta0, phi0'''
    s1,c1 = np.sin(theta0),np.cos(theta0)
    s2,c2 = np.sin(phi - phi0),np.cos(phi-phi0)
    s3,c3 = np.sin(theta),np.cos(theta)
    return np.array([[s1,c1,0],
                  [c1,-s1,0],
                  [0,0,1]]).dot(np.array([[c2,s2,0],
                                         [0,0,1],
                                         [s2,-c2,0]])).dot(np.array([[s3,c3,0],[0,0,-1],[c3,-s3,0]]))

def polarSphericalVars(x):
    '''transforms itrs whose lat is from equator'''
    theta = np.pi/2. - x.spherical.lat.rad
    phi = x.spherical.lon.rad
    r = x.spherical.distance.m
    return r,theta,phi


    
class splineFit(object):
    def __init__(self,data,x,y,z):
        '''creates a class where data is nxmxp and x is 1xn, y is 1xm, and z is 1xp.
        Does derivatives using analytic interpolation.
        Tried not to do things twice.'''
        self.data = data
        self.x = x
        self.dx = np.abs(x[1] - x[0])
        self.y = y
        self.dy = np.abs(y[1] - y[0])
        self.z = z
        self.dz = np.abs(z[1] - z[0])
        self.current_x = None
        self.current_y = None
        self.current_z = None

    def compute_zeroth(self,x,y,z):
        '''Return the nearest.'''
        if (self.current_x != x or self.current_y != y or self.current_z != z):
            self.current_x = x
            self.current_y = y
            self.current_z = z
            self.zerothDone = False
            self.onethDone = False
            self.twothDone = False
        if self.zerothDone:
            return self.zero
        else:
            nx = np.argmin((x - self.x)**2)
            ny = np.argmin((y - self.y)**2)
            nz = np.argmin((z - self.z)**2)
            self.xsp = UnivariateSpline(self.x,self.data[:,ny,nz] , k=2 , s = 2)
            self.ysp = UnivariateSpline(self.y,self.data[nx,:,nz] , k=2 , s = 2)
            self.zsp = UnivariateSpline(self.z,self.data[nx,ny,:] , k=2 , s = 2)
            self.zerothDone = True
            gx = self.xsp(x)
            gy = self.ysp(y)
            gz = self.zsp(z)
            self.zero = (gx+gy+gz)/3.
            return self.zero
        
    def compute_oneth(self,x,y,z):
        '''Calculate fourier of dsinc/dx and use that to compute du/dx then nearest.'''
        if (self.current_x != x or self.current_y != y or self.current_z != z):
            self.current_x = x
            self.current_y = y
            self.current_z = z
            self.zerothDone = False
            self.onethDone = False
            self.twothDone = False
            self.compute_zeroth(x,y,z)
        if self.onethDone:
            return self.one
        else:
            self.dxsp = self.xsp.derivative(n=1)
            self.dysp = self.ysp.derivative(n=1)
            self.dzsp = self.zsp.derivative(n=1)
            self.onethDone = True
            gx = self.dxsp(x)
            gy = self.dysp(y)
            gz = self.dzsp(z)
            self.one = (gx,gy,gz)
            return self.one
    def compute_twoth(self,x,y,z):
        '''Calculate fourier of dsinc/dx and use that to compute du/dx then nearest.'''
        if (self.current_x != x or self.current_y != y or self.current_z != z):
            self.current_x = x
            self.current_y = y
            self.current_z = z
            self.zerothDone = False
            self.onethDone = False
            self.twothDone = False
            self.compute_oneth(x,y,z)
        if self.twothDone:
            
            return self.two
        else:
            #should build xy,xz,yz components but those are second order
            self.dxxsp = self.xsp.derivative(n=2)
            self.dyysp = self.ysp.derivative(n=2)
            self.dzzsp = self.zsp.derivative(n=2)
            self.twothDone = True
            gxx = self.dxxsp(x)
            gxy = 0.
            gxz = 0.
            gyy = self.dyysp(x)
            gyz = 0.
            gzz = self.dzzsp(z)
            self.two = (gxx,gxy,gxz,gyy,gyz,gzz)
            return self.two
        
def coordsCart2Sph(x,y,z):
    r = np.sqrt(x*x+y*y+z*z)
    theta = np.arccos(z/r)
    phi = np.arctan2(y/x)
    return r,theta,phi

def coordsSph2Cart(r,theta,phi):
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return x,y,z
        
def compCart2SphMatrix(r,theta,phi):
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    return np.array([[cosphi*sintheta,sinphi*sintheta,costheta],
             [-sinphi,cosphi,0],
             [cosphi*costheta,sinphi*costheta,-sintheta]])

def compSph2CartMatric(r,theta,phi):
    return compCart2SphMatrix(r,theta,phi).transpose()

def compCart2Sph(compCart,r,theta,phi):
    '''(ar,atheta,aphi) = M.(ax,ay,az)'''
    M = compCart2SphMatrix(r,theta,phi)
    return M.dot(compCart)

def compSph2Cart(compSph,r,theta,phi):
    '''(ar,atheta,aphi) = M.(ax,ay,az)'''
    M = compSph2CartMatrix(r,theta,phi)
    return M.dot(compSph)

def gradSph2CartMatrix(r,theta,phi):
    '''problems at theta = 0
    {{Cos[phi]*Sin[theta], Cos[phi]*Cos[theta]/r,
    -Sin[phi]/r/Sin[theta]}, {Sin[phi]*Sin[theta],
    Sin[phi]*Cos[theta]/r,Cos[phi]/r/Sin[theta]}, {Cos[theta],-Sin[theta]/r,0}}
    '''
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    return np.array([[cosphi*sintheta, cosphi*costheta/r,-sinphi/r/sintheta],
                    [sinphi*sintheta,sinphi*costheta/r,cosphi/r/sintheta],
                     [costheta,-sintheta/r,0.]])

def gradCart2SphMatrix(r,theta,phi):
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    return np.array([[cosphi*sintheta,-r*sintheta*sinphi,r*costheta*cosphi],
                    [sintheta*sinphi,r*cosphi*sintheta,r*costheta*sinphi],
                    [costheta,0,-r*sintheta]])
    
def gradSph2Cart(gradSph, r,theta,phi):
    M = gradSph2CartMatrix(r,theta,phi)
    return M.dot(gradSph)

def gradCart2Sph(gradCart, r,theta,phi):
    M = gradCart2SphMatrix(r,theta,phi)
    return M.transpose().dot(gradCart)

def hessianSph2Cart(hessSph,r,theta,phi):
    M = gradSph2CartMatrix(r,theta,phi)
    return M.dot(hessSph).dot(M.transpose())

def hessianCart2Sph(hessCart,r,theta,phi):
    M = gradCart2SphMatrix(r,theta,phi)
    m00 = np.outer(M[:,0],M[:,0])
    m01 = np.outer(M[:,0],M[:,1])
    m02 = np.outer(M[:,0],M[:,2])
    m11 = np.outer(M[:,1],M[:,1])
    m12 = np.outer(M[:,1],M[:,2])
    m22 = np.outer(M[:,2],M[:,2])
    hessSph = np.zeros([3,3])
    hessSph[0,0] = np.trace(m00.dot(hessCart))
    hessSph[0,1] = np.trace(m01.dot(hessCart))
    hessSph[1,0] = hessSph[0,1]
    hessSph[0,2] = np.trace(m02.dot(hessCart))
    hessSph[2,0] = hessSph[0,2]
    hessSph[1,1] = np.trace(m11.dot(hessCart))
    hessSph[1,2] = np.trace(m12.dot(hessCart))
    hessSph[2,1] = hessSph[1,2]
    hessSph[2,2] = np.trace(m22.dot(hessCart))
    return hessSph

def gradAndHessCart2Sph(gradCart,hessCart,r,theta,phi):
    M = gradCart2SphMatrix(r,theta,phi)
    m00 = np.outer(M[:,0],M[:,0])
    m01 = np.outer(M[:,0],M[:,1])
    m02 = np.outer(M[:,0],M[:,2])
    m11 = np.outer(M[:,1],M[:,1])
    m12 = np.outer(M[:,1],M[:,2])
    m22 = np.outer(M[:,2],M[:,2])
    hessSph = np.zeros([3,3])
    gradSph = np.zeros(3)
    hessSph[0,0] = np.trace(m00.dot(hessCart))
    hessSph[0,1] = np.trace(m01.dot(hessCart))
    hessSph[1,0] = hessSph[0,1]
    hessSph[0,2] = np.trace(m02.dot(hessCart))
    hessSph[2,0] = hessSph[0,2]
    hessSph[1,1] = np.trace(m11.dot(hessCart))
    hessSph[1,2] = np.trace(m12.dot(hessCart))
    hessSph[2,1] = hessSph[1,2]
    hessSph[2,2] = np.trace(m22.dot(hessCart))
    gradSph[0] = M[:,0].dot(gradCart)
    gradSph[1] = M[:,1].dot(gradCart)
    gradSph[2] = M[:,2].dot(gradCart)
    return gradSph,hessSph

class numericDiff(object):
    def __init__(self,data,x,y,z):
        '''creates a class where data is nxmxp and x is 1xn, y is 1xm, and z is 1xp.
        Tried not to do things twice.'''
        self.data = np.ones([data.shape[0]+2,data.shape[1]+2,data.shape[2]+2])
        self.data[:data.shape[0],:data.shape[1],:data.shape[2]] = data
        self.x = x
        self.dx = np.abs(x[1] - x[0])
        self.y = y
        self.dy = np.abs(y[1] - y[0])
        self.z = z
        self.dz = np.abs(z[1] - z[0])
        self.current_x = None
        self.current_y = None
        self.current_z = None
        
    def compute_zeroth(self,x,y,z):
        '''Return the nearest.'''
        if (self.current_x != x or self.current_y != y or self.current_z != z):
            self.current_x = x
            self.current_y = y
            self.current_z = z
            self.current_nx = np.argmin((self.x - x)**2)
            self.current_ny = np.argmin((self.y - y)**2)
            self.current_nz = np.argmin((self.z - z)**2)
            #check if on edge
            self.zerothDone = False
            self.onethDone = False
            self.twothDone = False
        if self.zerothDone:
            return self.zero
        else:            
            g = self.data[self.current_nx,self.current_ny,self.current_nz]
            self.zerothDone = True
            self.zero = g
            return self.zero
        
    def compute_oneth(self,x,y,z):
        '''Calculate fourier of dsinc/dx and use that to compute du/dx then nearest.'''
        if (self.current_x != x or self.current_y != y or self.current_z != z):
            self.current_x = x
            self.current_y = y
            self.current_z = z
            self.zerothDone = False
            self.onethDone = False
            self.twothDone = False
            self.compute_zeroth(x,y,z)
        if self.onethDone:
            return self.one
        else:
            gx = self.data[self.current_nx+1,self.current_ny,self.current_nz] - self.zero
            gy = self.data[self.current_nx,self.current_ny+1,self.current_nz] - self.zero
            gz = self.data[self.current_nx,self.current_ny,self.current_nz+1] - self.zero
            self.one = (gx/self.dx,gy/self.dy,gz/self.dz)
            self.onethDone = True
            return self.one
    def compute_twoth(self,x,y,z):
        '''Calculate fourier of dsinc/dx and use that to compute du/dx then nearest.'''
        if (self.current_x != x or self.current_y != y or self.current_z != z):
            self.current_x = x
            self.current_y = y
            self.current_z = z
            self.zerothDone = False
            self.onethDone = False
            self.twothDone = False
            self.compute_oneth(x,y,z)
        if self.twothDone:
            return self.two
        else:
            nx,ny,nz = self.current_nx,self.current_ny,self.current_nz
            gxx = (self.data[nx+2,ny,nz] - 2*self.data[nx+1,ny,nz] + self.data[nx,ny,nz])/self.dx**2
            gxy = ((self.data[nx+1,ny+1,nz] - self.data[nx,ny+1,nz])/self.dx - self.one[0])/self.dy
            gxz = ((self.data[nx+1,ny,nz+1] - self.data[nx,ny,nz+1])/self.dx - self.one[0])/self.dz
            gyy = (self.data[nx,ny+2,nz] - 2*self.data[nx,ny+1,nz] + self.data[nx,ny,nz])/self.dy**2
            gyz = ((self.data[nx,ny+1,nz+1] - self.data[nx,ny,nz+1])/self.dy - self.one[1])/self.dz
            gzz = (self.data[nx,ny,nz+2] - 2*self.dta[nx,ny,nz+1] + self.data[nx,ny,nz])/self.dz**2
            self.two = (gxx,gxy,gxz,gyy,gyz,gzz)
            return self.two

#class NObject(splineFit):
class NObject(numericDiff):
    def __init__(self,data,x,y,z):
        '''data is cartesian, but compute will be in spherical'''
        super(NObject,self).__init__(data,x,y,z)
    def compute_n(self,r,theta,phi):
        #convert r,theta,phi to x,y,z
        x,y,z = coordsSph2Cart(r,theta,phi)
        return self.compute_zeroth(x,y,z)
    def compute_dn(self,r,theta,phi):
        x,y,z = coordsSph2Cart(r,theta,phi)
        nx,ny,nz = self.compute_oneth(x,y,z)
        nr,ntheta,nphi = gradCart2Sph(np.array([nx,ny,nz]),r,theta,phi)
        return nr,ntheta,nphi
    def compute_ddn(self,r,theta,phi):
        x,y,z = coordsSph2Cart(r,theta,phi)
        nxx,nxy,nxz,nyy,nyz,nzz = self.compute_twoth(x,y,z)
        H = np.array([[nxx,nxy,nxz],
                    [nxy,nyy,nyz],
                    [nxz,nyz,nzz]])
        Hsph = hessianCart2Sph(H,r,theta,phi)
        return Hsph[0,0],Hsph[0,1],Hsph[0,2],Hsph[1,1],Hsph[1,2],Hsph[2,2]
           
def eulerEqns(t,p, nObj):
    pr = p[0]
    ptheta = p[1]
    pphi = p[2]
    r = p[3]
    theta = p[4]
    phi = p[5]

    n = nObj.compute_n(r,theta,phi)
    nr,ntheta,nphi = nObj.compute_dn(r,theta,phi)
    
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    n_r = n*r
    r2 = r*r
    
    prdot = (ptheta**2 + pphi**2/sintheta**2)/(n_r*r2) + nr
    pthetadot = costheta * pphi**2/(n_r*r*sintheta**3) + ntheta
    pphidot = nphi
    rdot = pr/n
    thetadot = ptheta/(n_r*r)
    phidot = pphi/(n_r*r*sintheta**2)
    return [prdot,pthetadot,pphidot,rdot,thetadot,phidot]
    
def eulerJac(t,p,nObj):
    pr = p[0]
    ptheta = p[1]
    pphi = p[2]
    r = p[3]
    theta = p[4]
    phi = p[5]
    
    n = nObj.compute_n(r,phi,theta)
    nr,ntheta,nphi = nObj.compute_dn(r,theta,phi)
    nrr,nrtheta,nrphi,nthetatheta,nthetaphi,nphiphi = nObj.compute_ddn(r,theta,phi)
    
    sintheta = np.sin(theta)
    sintheta2 = sintheta*sintheta
    costheta = np.cos(theta)
    n_r = n*r
    r2 = r*r
    pphi2 = pphi*pphi
    
    jac = np.zeros([6,6])
    #dprdot
    A1 = (ptheta*ptheta + pphi2/sintheta2)/n_r
    jac[0,1] = 2.*ptheta/(n_r*r2)
    jac[0,2] = 2*pphi/(n_r*r2*sintheta2)
    jac[0,3] = (-nr/(n_r*r) - 3./(r2*r))*A1 + nrr
    jac[0,4] = -ntheta/(n_r*r)*A1 - 2*pphi2*costheta/(n_r*r2*sintheta2*sintheta) + nrtheta
    jac[0,5] = -nphi/(n_r*r)*A1 + nrphi
    #dpthetadot
    A2 = costheta/(sintheta2*sintheta)/n_r
    jac[1,2] = 2*pphi*A2/r
    jac[1,3] = (-nr/n_r - 2/r2)*A2*pphi2 + nrtheta
    jac[1,4] = (-1./(n_r*r*sintheta2) - (3*costheta/(r*sintheta) + ntheta/n_r)*A2)*pphi2 + nthetatheta
    jac[1,5] = -nphi*A2/r*pphi2 + nthetaphi
    #dpphidot
    jac[2,3] = nrphi
    jac[2,4] = nthetaphi
    jac[2,5] = nphiphi
    #drdot
    A3 = pr/(n*n)
    jac[3,0] = 1./n
    jac[3,3] = A3*nr
    jac[3,4] = A3*ntheta
    jac[3,5] = A3*nphi
    #dthetadot
    A4 = -ptheta/(n_r*n_r)
    jac[4,1] = 1./(n_r*r)
    jac[4,3] = (nr + 2*n/r)*A4
    jac[4,4] = ntheta*A4
    jac[4,5] = nphi*A4
    #dphidot
    A5 = 1./(n_r*r*sintheta2)
    jac[5,2] = A5
    jac[5,3] = (-nr/n - 2/r)*A5*pphi
    jac[5,4] = (-ntheta/n*A5 - A2/r) * pphi
    return jac
    
def propagateBackwards(l,m,s,x0,xi,obstime,NObj,rmaxRatio,plot=False):
    '''Propagate a ray from observer to source plane using numerical integration.
    l - direction cosine pointing East
    m - direction cosine pointing West
    s - ponting center of field, ICRS object
    x0 - Location of observer coordinate system origin, ITRS object
    obstime - ISOT or time object
    rmaxRatio - multiple of earth radius to propagate to in plane perp to s pointing
    '''
    #initial location
    r,theta,phi = polarSphericalVars(xi)
    #center where frame defined
    r0,theta0,phi0 = polarSphericalVars(x0) 
    #transform matrix cosines from ray location to center
    M = Frame2Frame(theta0,phi0,theta,phi)
    #direction cosines on pointing at center
    frame = ac.AltAz(location = x0, obstime = obstime, pressure=None, copy = True) 
    s_ = s.transform_to(frame)
    theta_s = np.pi/2. - s_.spherical.lat.rad
    phi_s = s_.spherical.lon.rad
    
    alt = s_.alt.rad#of pointing
    Az = s_.az.rad
    #l,m relative to s pointing
    alt_ = np.arcsin(np.sqrt(1-l**2-m**2))
    Az_ = np.arctan2(l,m)
    #direction cosines of l,m
    ar0 = np.sin(alt+alt_)
    atheta0 = np.cos(alt+alt_)*np.sin(Az+Az_)
    aphi0 = np.cos(alt+alt_)*np.cos(Az+Az_)
    #direction cosines of s pointing    
    ar_s0 = np.sin(alt)#(1 - l0**2 - m0**2)
    aphi_s0 = np.cos(alt)*np.cos(Az)#l0
    atheta_s0 = np.cos(alt)*np.sin(Az)#m0
    #transform direction cosines to xi
    ar,atheta,aphi = M.transpose().dot(np.array([ar0,atheta0,aphi0]))
    
    #define parameters
    n0 = NObj.compute_n(r0,theta0,phi0)
    #for analytic radial profile
    C = n0*r0*np.cos(alt+alt_)
    pr0 = n0*ar
    ptheta0 = n0*r0*atheta
    pphi0 = n0*r0*np.sin(theta0)*aphi
    rmax = r0*rmaxRatio
    cosx0s = s_.cartesian.xyz.value.dot(x0.cartesian.xyz.value)
    rNum = np.sqrt(cosx0s**2 - r0**2 + rmax**2)
    #ODE = ode(eulerEqns, eulerJac).set_integrator('vode', method='bdf').set_jac_params(NObj)
    ODE = ode(eulerEqns).set_integrator('vode', method='bdf')
    ODE.set_initial_value([pr0,ptheta0,pphi0,r0,theta0,phi0], 0)#set initit and time=0
    ODE.set_f_params(NObj)
    dt = (rmax-r0)/100.#sections of arc. Maybe should be larger
    if plot:
        sols = []
    while r < rNum/np.abs(np.sin(theta_s)*np.sin(theta)*np.cos(phi_s - phi) + np.cos(theta_s)*np.cos(theta)) and ODE.successful():
        pr,ptheta,pphi,r,theta,phi = ODE.integrate(ODE.t + dt)
        if plot:
            pathlength = ODE.t
            psi = -np.arccos(C/r/NObj.compute_n(r,theta,phi))#+(alt+alt_)
            sols.append([ODE.t,pr,ptheta,pphi,r,theta,phi,psi])
            print(pathlength,pr,ptheta,pphi,r,theta,phi)
    if plot:
        import pylab as plt
        sols= np.array(sols)
        plt.subplot(121)
        plt.plot(sols[:,4],sols[:,0])
        #plt.scatter(sols[:,3],sols[:,6])
        plt.subplot(122)
        plt.plot(sols[:,4],sols[:,5])
        plt.show()
    return ODE.t

def plotPathLength(lvec,mvec,s,x0,xi,obstime,NObj,ramxRatio):
    pl = np.zeros([np.size(lvec),np.size(mvec)])
    i = 0
    while i < len(l):
        j = 0
        while j < len(m):
            print("doing:",i,j)
            pl[i,j] = propagateBackwards(l[i],m[j],s,x0,xi,obstime,NObj,2)
            j += 1
        i += 1
    
    
if __name__=='__main__':
    l=0.1
    m=0.6
    obstime = at.Time('2000-01-01T00:00:00.000',format='isot',scale='utc')
    x0 = ac.ITRS(*ac.EarthLocation(lon=0*au.deg,lat=0*au.deg,height=0*au.m).geocentric)
    xvec = np.linspace(x0.cartesian.x.value,x0.cartesian.x.value*2,100)
    yvec = np.linspace(-x0.cartesian.x.value/2.,x0.cartesian.x.value/2.,100)
    zvec = np.linspace(-x0.cartesian.x.value/2.,x0.cartesian.x.value/2.,100)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    ndata = 1 + 0.1*np.cos(R/50000.)
    #ndata = 0.95+0.1*np.random.uniform(size=[100,100,100])
    NObj = NObject(ndata,xvec,yvec,zvec)
    xi = ac.ITRS(*ac.EarthLocation(lon=0*au.deg,lat=0.001*au.deg,height=0*au.m).geocentric)
    s = ac.SkyCoord(ra=0*au.deg,dec=0*au.deg,frame='icrs')
    propagateBackwards(l,m,s,x0,xi,obstime,NObj,2)
    l = np.linspace(-0.5,0.5,10)
    m = np.linspace(-0.5,0.5,10)
    pl = np.zeros([10,10])
    i = 0
    while i < len(l):
        j = 0
        while j < len(m):
            print("doing:",i,j)
            pl[i,j] = propagateBackwards(l[i],m[j],s,x0,xi,obstime,NObj,2)
            j += 1
        i += 1
    import pylab as plt
    plt.pl
    plt.imshow(pl)
    import time
    t1 = time.time()
    i = 20
    while i < 20:
        propagateBackwards(l,m,s,x0,xi,obstime,NObj)
        i += 1
    print(20./(time.time()-t1))
    
    
    
    
    
            
        
        
        
 
        


# In[ ]:



