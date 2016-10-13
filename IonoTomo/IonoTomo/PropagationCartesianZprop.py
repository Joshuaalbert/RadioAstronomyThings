
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
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    x1 = p[3]
    x2 = p[4]
    S = p[5]
    
    n = nObj.compute_n(x1,x2,t)
    n1,n2,n3 = nObj.compute_dn(x1,x2,t)
    
    p1dot = n*n1/p3
    p2dot = n*n2/p3
    p3dot = n*n3/p3
    x1dot = p1/p3
    x2dot = p2/p3
    Sdot = n**2/p3

    return [p1dot,p2dot,p3dot,x1dot,x2dot,Sdot]
    
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
    

def propagateBackwards(l,m,xi,NObj,zmax, plot=False):
    '''Propagate a ray from observer to source plane using numerical integration.
    l - direction cosine pointing East
    m - direction cosine pointing West
    s - ponting center of field, ICRS object
    x0 - Location of observer coordinate system origin, ITRS object
    obstime - ISOT or time object
    rmaxRatio - multiple of earth radius to propagate to in plane perp to s pointing
    '''
    #Relative to center with xi[2] pointing in +z
    x10,x20,x30 = xi[0],xi[1],xi[2]
    n0 = NObj.compute_n(x10,x20,x30)
    p10,p20,p30 = l*n0,m*n0,np.sqrt(1-l**2-m**2)*n0
    S = 0
    
    #ODE = ode(eulerEqns, eulerJac).set_integrator('vode', method='bdf').set_jac_params(NObj)
    ODE = ode(eulerEqns).set_integrator('vode', method='bdf')
    ODE.set_initial_value([p10,p20,p30,x10,x20,S], x30)#set initit and z=x30
    ODE.set_f_params(NObj)
    #p1,p2,p3,x1,x2,x3 = ODE.integrate(ODE.t + zmax)
    p1,p2,p3,x1,x2,S = p10,p20,p30,x10,x20,S
    dz = zmax/100.#sections of arc. Maybe should be larger
    if plot:
        sols = []
    while ODE.t < x30+zmax and ODE.successful():
        p1,p2,p3,x1,x2,S = ODE.integrate(ODE.t + dz)
        if plot:
            n = NObj.compute_n(x1,x2,ODE.t)
            l,m = p1/n,p2/n
            sols.append([ODE.t,S,l,m,x1,x2])
            sols[-1]
    if plot:
        import pylab as plt
        sols= np.array(sols)
        plt.subplot(131)
        plt.plot(sols[:,0],sols[:,1])
        plt.xlabel('r (m)')
        plt.ylabel('pathlength (m)')
        #plt.scatter(sols[:,3],sols[:,6])
        plt.subplot(132)
        plt.plot(sols[:,2],sols[:,3])
        plt.xlabel('l')
        plt.ylabel('m')
        plt.subplot(133)
        plt.plot(sols[:,4],sols[:,5])
        plt.xlabel('x1 (m)')
        plt.ylabel('x2 (m))')
        plt.show()
    return S

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
    import pylab as plt
    plt.imshow(pl.transpose(),origin='lower',extent=(l[0],l[-1],m[0],m[-1]))
    plt.xlabel('l')
    plt.ylabel('m')
    plt.show()
    
    
if __name__=='__main__':
    xi = [0,0,0]
    l=0.1
    m=0.6
    propagateBackwards(l,m,xi,NObj,zmax, plot=False)

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
    propagateBackwards(l,m,s,x0,xi,obstime,NObj,2,plot=True)
    lvec = np.linspace(-0.5,0.5,10)
    mvec = np.linspace(-0.5,0.5,10)
    plotPathLength(lvec,mvec,s,x0,xi,obstime,NObj,rmaxRatio)
    
    
    
            
        
        
        
 
        

