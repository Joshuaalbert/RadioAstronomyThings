
# coding: utf-8

# In[202]:

import numpy as np
from Logger import Logger
from Layer import Layer
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at
from scipy.integrate import ode
from scipy.interpolate import UnivariateSpline

def fft(A):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(A)))

def ifft(A):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(A)))

def transformCosines(theta1,phi1,theta2,phi2):
    #switch theta and phi for this implementation
    cosphi1 = np.cos(theta1)
    sinphi1 = np.sin(theta1)
    costheta1 = np.cos(phi1)
    sintheta1 = np.sin(phi1)
    cosphi2 = np.cos(theta2)
    sinphi2 = np.sin(theta2)
    costheta2 = np.cos(phi2)
    sintheta2 = np.sin(phi2)
    costheta12 = np.cos(phi1-phi2)
    sintheta12 = np.sin(phi1-phi2)
    
    return np.array([[cosphi1*cosphi2 + costheta12*sinphi1*sinphi2,sinphi1*sintheta12,cosphi2*costheta12*sinphi1 - cosphi1*sinphi2],
       [cosphi2*sinphi1 - cosphi1*costheta12*sinphi2,-cosphi1*sintheta12,-cosphi1*cosphi2*costheta12 - sinphi1*sinphi2],
       [sinphi2*sintheta12,-costheta12,cosphi2*sintheta12]])

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

class gaussianDecomposition(object):
    def __init__(self,params):
        self.x0 = params[:,0]
        self.y0 = params[:,1]
        self.z0 = params[:,2]
        self.a = params[:,3]
        self.bx = params[:,4]
        self.by = params[:,5]
        self.bz = params[:,6]    
        self.zeroarray = np.zeros(np.size(self.x0))
        self.onearray = np.zeros([3,np.size(self.x0)])
        self.current_x = None
        self.current_y = None
        self.current_z = None

    def fitParameters(self,N):
        '''Fit N component Gaussian model to data'''
        
        data = np.copy(self.data) - 1#zero centered 1-vp^2/v^2
        xdata = np.sum(np.sum(data,axis=2),axis=1)
        ydata = np.sum(np.sum(data,axis=2),axis=0)
        zdata = np.sum(np.sum(data,axis=1),axis=0)
        xsp = UnivariateSpline(self.x,xdata , k=5 , s = 2)
        ysp = UnivariateSpline(self.y,ydata , k=5 , s = 2)
        zsp = UnivariateSpline(self.z,zdata , k=5 , s = 2)
        dxsp = xsp.derivative(n=1)
        dddxsp = UnivariateSpline(self.x,dxsp(self.x) , k=5 , s = 2).derivative(n=2)
        ddxsp = xsp.derivative(n=2)
        ddddxsp = xsp.derivative(n=4)
        dysp = ysp.derivative(n=1)
        dddysp = UnivariateSpline(self.y,dysp(self.y) , k=5 , s = 2).derivative(n=2)
        ddysp = ysp.derivative(n=2)
        ddddysp = ysp.derivative(n=4)
        dzsp = zsp.derivative(n=1)
        dddzsp = UnivariateSpline(self.z,dxsp(self.z) , k=5 , s = 2).derivative(n=2)
        ddzsp = zsp.derivative(n=2)
        ddddzsp = zsp.derivative(n=4)
        #find parameters that fit f>ep, ddf<0, dddf=0, ddddf > 0
        xroots = dddxsp.roots()
        yroots = dddysp.roots()
        zroots = dddzsp.roots()
        print xroots,yroots,zroots
       
    def compute_zeroth(self,x,y,z):
        '''Return the nearest.'''
        if (self.current_x != x or self.current_y != y or self.current_z != z):
            self.current_x = x
            self.current_y = y
            self.current_z = z
        i = 0
        while i < np.size(self.x0):
            self.zeroarray[i] = self.a[i]*np.exp(-(x-self.x0[i])**2/self.bx[i]**2-(y-self.y0[i])**2/self.by[i]**2-(z-self.z0[i])**2/self.bz[i]**2)
            i += 1
        self.zero = 1+np.sum(self.zeroarray)
        return self.zero
                
    def compute_oneth(self,x,y,z):
        '''Calculate grad of n'''
        if (self.current_x != x or self.current_y != y or self.current_z != z):
            self.current_x = x
            self.current_y = y
            self.current_z = z
            self.compute_zeroth(x,y,z)
        i = 0
        while i < np.size(self.x0):
            self.onearray[0,i] = -2*(x-self.x0[i])/self.bx[i]**2 * self.zeroarray[i]
            self.onearray[1,i] = -2*(y-self.y0[i])/self.by[i]**2 * self.zeroarray[i]
            self.onearray[2,i] = -2*(z-self.z0[i])/self.bz[i]**2 * self.zeroarray[i]
            i += 1
        self.one = np.sum(self.onearray,axis=1)
        #print self.one,(x-self.x0[0])/self.bx[0],(y-self.y0[0])/self.by[0],(z-self.z0[0])/self.bz[0]
        return self.one
    
    def compute_twoth(self,x,y,z):
        '''Calculate Hessian of n'''
        if (self.current_x != x or self.current_y != y or self.current_z != z):
            self.current_x = x
            self.current_y = y
            self.current_z = z
            self.compute_oneth(x,y,z)
        nxx,nxy,nxz,nyy,nyz,nzz = 0.,0.,0.,0.,0.,0.
        i = 0
        while i < np.size(self.x0):
            nxx += -2*self.zeroarray[i]*(self.bx[i]**2 - 2*(x-self.x0[i])**2)/self.bx[i]**4
            nyy += -2*self.zeroarray[i]*(self.by[i]**2 - 2*(y-self.y0[i])**2)/self.by[i]**4
            nzz += -2*self.zeroarray[i]*(self.bz[i]**2 - 2*(z-self.z0[i])**2)/self.bz[i]**4
            nxy += self.onearray[0,i]*self.onearray[1,i]/self.zeroarray[i]
            nxz += self.onearray[0,i]*self.onearray[2,i]/self.zeroarray[i]
            nyz += self.onearray[1,i]*self.onearray[2,i]/self.zeroarray[i]
            i += 1
            self.two = nxx,nxy,nxz,nyy,nyz,nzz
            return self.two

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
            gzz = (self.data[nx,ny,nz+2] - 2*self.data[nx,ny,nz+1] + self.data[nx,ny,nz])/self.dz**2
            self.two = (gxx,gxy,gxz,gyy,gyz,gzz)
            return self.two

#class NObject(splineFit):
#class NObject(numericDiff):
class NObject(gaussianDecomposition):
    def __init__(self,params):
        super(NObject,self).__init__(params)
#    def __init__(self,data,x,y,z):
#        '''data is cartesian, but compute will be in spherical'''
#        super(NObject,self).__init__(data,x,y,z)
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
    phase = p[6]

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
    phasedot = n*np.sqrt(rdot**2 + thetadot**2*r**2 + r**2*sintheta**2*phidot**2)
    #ar,atheta,aphi = transformCosines(theta0,phi0,theta,phi).dot(np.array([pr/n,ptheta/n/r,pphi/n/r/np.sin(theta)]))
    #phasedot = np.cos(np.arcsin(ar) - alt_s)*n*2*np.pi
    return [prdot,pthetadot,pphidot,rdot,thetadot,phidot,phasedot]
    
def eulerJac(t,p,nObj):
    pr = p[0]
    ptheta = p[1]
    pphi = p[2]
    r = p[3]
    theta = p[4]
    phi = p[5]
    phase = p[6]
    
    n = nObj.compute_n(r,phi,theta)
    nr,ntheta,nphi = nObj.compute_dn(r,theta,phi)
    nrr,nrtheta,nrphi,nthetatheta,nthetaphi,nphiphi = nObj.compute_ddn(r,theta,phi)
    
    sintheta = np.sin(theta)
    sintheta2 = sintheta*sintheta
    #costheta = np.cos(theta)
    
    pphi2 = pphi*pphi
    
    csctheta2 = 1./np.sin(theta)**2
    cottheta = 1./np.tan(theta)
    jac = np.zeros([6,6])
    r2 = 1/r**2
    n2 = 1/n**2
    nr2 = 1/n*r2
    nr3 = nr2/r
    n2r2 = n2*r2
    n2r3 = n2r2/r
    A0 = pphi*csctheta2
    A1 = pphi*A0
    A2 = ptheta**2 + A1
    #col pr
    
    jac[:,0]=np.array([0,0,0,1/n,0,0])
    #col ptheta
    jac[:,1]=np.array([(2*ptheta)*nr3,
                       0,
                       0,
                       0,
                       0,
                       nr2])
    #col pphi
    jac[:,2] = np.array([(2*A0)*nr3 ,
                         (2*A0*cottheta)*nr2,
                         0,
                         0,
                         csctheta2*nr2,
                         0])
    #col r
    jac[:,3] = np.array([-((A2*(3*n + r*nr))*n2r2*r2) + nrr,
                         -((A1*cottheta*(2*n + r*nr))*n2r3) + nrtheta,
                         nrphi,
                         ((pr*nr)*n2),
                         -((A0*(2*n + r*nr))*n2r3),
                         -((ptheta*(2*n + r*nr))*n2r3)])

    #col theta
    jac[:,4] = np.array([-((2*n*A1*cottheta + A2*ntheta)*n2r3) + nrtheta,
                          - (nthetatheta/n2r2 - (A1*csctheta2*(2*n*(2 + np.cos(2*theta)) + ntheta*np.sin(2*theta)))/2.)*n2r2,
                         nthetaphi,
                         -((pr*ntheta)*n2),
                         ((A0*(2*n*cottheta + ntheta))*n2r2),
                         -((ptheta*ntheta)*n2r2)])
    #col phi
    jac[:,5] = np.array([-((A2*nphi)*n2r3) + nrphi,
                         -((A1*cottheta*nphi)*n2r2) + nthetaphi,
                         nphiphi,
                         ((pr*nphi)*n2),
                         -((A0*nphi)*n2r2),
                         -((ptheta*nphi)*n2r2)])
    return jac

def LM2DiffAltAz(l,m):
    dAlt = np.arccos(np.sqrt(1-l**2-m**2))
    dAz = np.arctan2(l,m)
    return dAlt,dAz

def getKzComp(r,theta,phi,n,alt_s,theta0,phi0):
    #ar = sin(as+da)
    
    #cos(da) = sqrt(1-l**2-m**2)
    #kz = 2pin/lambda*sqrt(1-l**2-m**2)
    cosphi1 = np.cos(theta0)
    sinphi1 = np.sin(theta0)
    #costheta1 = np.cos(phi0)
    sintheta1 = np.sin(phi0)
    cosphi2 = np.cos(theta)
    sinphi2 = np.sin(theta)
    costheta2 = np.cos(phi)
    sintheta2 = np.sin(phi)
    costheta12 = np.cos(phi-phi)
    sintheta12 = np.sin(phi-phi)
    
    ar = (cosphi1*cosphi2 + costheta12*sinphi1*sinphi2)*pr/n+    (sinphi1*sintheta12)*ptheta/n/r+    (cosphi2*costheta12*sinphi1 - cosphi1*sinphi2)*pphi/n/r/np.sin(theta)
    da = np.arcsin(ar)-alt_s
    kz = np.cos(np.arcsin(ar) - alt_s)*2*np.pi
    return kz
    
def zDist(r,theta,phi,s,x0):
    phis = s.spherical.lon.rad
    thetas = np.pi/2. - s.spherical.lat.rad
    r0,theta0,phi0 = polarSphericalVars(x0)
    
    costhetas = np.cos(thetas)
    sinthetas = np.sin(thetas)
    
    zDist = -r*(costhetas*np.cos(theta)+np.cos(phis-phi)*sinthetas*np.sin(theta)) + r0*(costhetas*np.cos(theta0)+np.cos(phis-phi0)*sinthetas*np.sin(theta0))
    return zDist
    
def zDot(r,theta,phi,s,pr,ptheta,pphi,n):
    phis = s.spherical.lon.rad
    thetas = np.pi/2. - s.spherical.lat.rad
    
    costhetas = np.cos(thetas)
    costheta = np.cos(theta)
    cosphis1 = np.cos(phis - phi)
    sintheta = np.sin(theta)
    sinthetas = np.sin(thetas)
    zdot = (costhetas*costheta+cosphis1*sinthetas*sintheta)*pr/n +(-costhetas*sintheta+cosphis1*sinthetas*costheta)*ptheta/n/r +(-np.sin(phi-phis)*sinthetas*sintheta)*pphi/n/r/sintheta**2
    return zdot
    
def propagateBackwards(l,m,s,x0,xi,obstime,NObj,rmaxRatio,plot=False,ax=None):
    '''Propagate a ray from observer to source plane using numerical integration.
    l - direction cosine pointing East
    m - direction cosine pointing West
    s - ponting center of field, ICRS object
    x0 - Location of observer coordinate system origin, ITRS object
    obstime - ISOT or time object
    rmaxRatio - multiple of earth radius to propagate to in plane perp to s pointing
    '''
    r2d = 180./np.pi
    #initial location
    r,theta,phi = polarSphericalVars(xi)
    #center where frame defined
    r0,theta0,phi0 = polarSphericalVars(x0) 
    #transform matrix cosines from ray location to center
    
    #direction cosines on pointing at center
    frame = ac.AltAz(location = x0, obstime = obstime, pressure=None, copy = True) 
    s_ = s.transform_to(frame)
    #for stopping criterion
    theta_s = np.pi/2. - s_.spherical.lat.rad
    phi_s = s_.spherical.lon.rad
    
    alt_s = s_.alt.rad#of pointing
    az_s = s_.az.rad
    #l,m alt/az relative to s pointing
    dAlt,dAz = LM2DiffAltAz(l,m)
    #alt,az of s+l,m
    alt = alt_s + dAlt
    az = az_s + dAz
    #direction cosines of s+l,m at center
    ar0 = np.sin(alt)
    atheta0 = np.cos(alt)*np.cos(az)
    aphi0 = np.cos(alt)*np.sin(az)
    #transform to xi
    M = transformCosines(theta0,phi0,theta,phi)
    ar,atheta,aphi = M.transpose().dot(np.array([ar0,atheta0,aphi0]))
    if plot:
        print("----")
        print("Obs. location (aperture center): lon: {0}, lat: {1}, radial: {2}".format(x0.earth_location.geodetic[0].deg,
                                                                      x0.earth_location.geodetic[1].deg,
                                                                      x0.earth_location.geodetic[2]))
        print("Obs. offset (ray emitter): lon: {0}, lat: {1}, radial: {2}".format(xi.earth_location.geodetic[0].deg-x0.earth_location.geodetic[0].deg,
                                                                      xi.earth_location.geodetic[1].deg-x0.earth_location.geodetic[1].deg,
                                                                      xi.earth_location.geodetic[2]-x0.earth_location.geodetic[2]))
        print("Obs. time: {0}".format(obstime.isot))
        print("Pointing center: ra = {0}, dec = {1}".format(s.ra.deg,s.dec.deg))
        print("\talt = {0}, az = {1}".format(alt_s*r2d,az_s*r2d))
        print("Image plane cosines: l = {0}, m = {1}".format(l,m))
        print("Ray initial direction: alt = {0}, az = {1}".format(alt*r2d,az*r2d))
        print("Ray initial cosines: ar = {0}, atheta = {1}, aphi = {2}".format(ar,atheta,aphi))
        print("----")
    #define parameters
    n = NObj.compute_n(r,theta,phi)
    #print(n)
    #for analytic radial profile
    #C = n0*r0*np.cos(alt)
    pr = n*ar
    ptheta = n*r*atheta
    pphi = n*r*np.sin(theta)*aphi
    rmax = r0*rmaxRatio
    cosx0s = s_.cartesian.xyz.value.dot(x0.cartesian.xyz.value)
    rNum = np.sqrt(cosx0s**2 - r0**2 + rmax**2)
    #ODE = ode(eulerEqns, eulerJac).set_integrator('vode',method='adams').set_jac_params(NObj)
    ODE = ode(eulerEqns).set_integrator('vode', method='adams')
    phase = 0
    ODE.set_initial_value([pr,ptheta,pphi,r,theta,phi,phase], 0)#set initit and time=0
    ODE.set_f_params(NObj)
    zMax = rmax - r0
    #one go
    if not plot:
        pr,ptheta,pphi,r,theta,phi,phase = ODE.integrate(rmax)
        M = transformCosines(theta0,phi0,theta,phi)
        n = NObj.compute_n(r,theta,phi)
        ar,atheta,aphi = M.dot(np.array([pr/n,ptheta/n/r,pphi/n/r/np.sin(theta)]))
        xf=r*np.cos(phi)*np.sin(theta)
        yf=r*np.sin(phi)*np.sin(theta)
        zf=r*np.cos(theta)
        xs = np.cos(phi_s)*np.sin(theta_s)
        ys = np.sin(phi_s)*np.sin(theta_s)
        zs = np.cos(theta_s)
        x0=r0*np.cos(phi0)*np.sin(theta0)
        y0=r0*np.sin(phi0)*np.sin(theta0)
        z0=r0*np.cos(theta0)
        #isoplanDiff = xf*xs+yf*ys+zf*zs - (x0*xs+y0*ys+z0*zs)
        #phase += np.cos(np.arcsin(ar) - alt_s)*n*2*np.pi*isoplanDiff
        
        return phase
    zMax = rmax-r0
    
    if plot:
        sols = []
        X,Y,Z,N  = [],[],[],[]
        X.append(r*np.cos(phi)*np.sin(theta))
        Y.append(r*np.sin(phi)*np.sin(theta))
        Z.append(r*np.cos(theta))
    #while r < rNum/np.abs(np.sin(theta_s)*np.sin(theta)*np.cos(phi_s - phi) + np.cos(theta_s)*np.cos(theta)) and ODE.successful():
    z = zDist(r,theta,phi,s,x0)
    print zDot(r,theta,phi,s,pr,ptheta,pphi,n)
    while r < rmax:#zMax:
        dt = zMax/100.
        #dt = max(zMax/10000,(zMax - z)/zDot(r,theta,phi,s,pr,ptheta,pphi,n)/10.)#sections of arc.
        pr,ptheta,pphi,r,theta,phi,phase = ODE.integrate(ODE.t + dt)
        #print zDot(r,theta,phi,s,pr,ptheta,pphi,n),dt
        M = transformCosines(theta0,phi0,theta,phi)
        n = NObj.compute_n(r,theta,phi)
        ar,atheta,aphi = M.dot(np.array([pr/n,ptheta/n/r,pphi/n/r/np.sin(theta)]))
        z = zDist(r,theta,phi,s,x0)
        #print z,zMax,dt
        #ar,atheta,aphi = r/n,ptheta/n/r,pphi/n/r/np.sin(theta)
        #print ar, atheta, aphi
        if plot: 
            pathlength = ODE.t
            X.append(r*np.cos(phi)*np.sin(theta))
            Y.append(r*np.sin(phi)*np.sin(theta))
            Z.append(r*np.cos(theta))
            N.append(n)
            #print (ar,ar_)
            #psi = -np.arccos(C/r/NObj.compute_n(r,theta,phi))#+(alt+alt_)
            sols.append([pathlength,ar,atheta,aphi,r,theta,phi,dt,phase])
            #print(pathlength,pr,ptheta,pphi,r,theta,phi)
    if plot:
        import pylab as plt
        ax.plot(X,Y,Z)
        #plt.gcf().savefig('figs/axes_{0:04d}'.format(num))
        sols= np.array(sols)
        f = plt.figure()
        plt.subplot(131)
        plt.plot(sols[:,4]-xi.spherical.distance.m,sols[:,8])
        plt.xlabel('r (m)')
        plt.ylabel('pathlength (m)')
        plt.subplot(132)
        plt.plot(sols[:,4]-xi.spherical.distance.m,N)
        #plt.scatter(sols[:,4],sols[:,2])
        plt.xlabel('r (m)')
        plt.ylabel('n')
        plt.subplot(133)
        plt.plot(sols[:,4]-xi.spherical.distance.m,sols[:,1])
        plt.xlabel('r (m)')
        plt.ylabel('ar Sqrt(1-l^2-m^2)')
        plt.show()
    
    #isoplanDiff = xf*xs+yf*ys+zf*zs - (x0*xs+y0*ys+z0*zs)
    #phase += np.cos(np.arcsin(ar) - alt_s)*n*2*np.pi*isoplanDiff
    return phase

def plotPathLength(lvec,mvec,s,x0,xi,obstime,NObj,rmaxRatio,num=0):
    pl = np.zeros([np.size(lvec),np.size(mvec)])
    i = 0
    while i < len(lvec):
        j = 0
        while j < len(mvec):
            pl[i,j] = propagateBackwards(lvec[i],mvec[j],s,x0,xi,obstime,NObj,rmaxRatio)*np.pi*2
            j += 1
        i += 1
    pl = np.angle(ifft(np.abs(fft(pl/3e8))**2))
    import pylab as plt
    f=plt.figure()
    plt.imshow((pl.transpose()-pl[0,0]),origin='lower',extent=(lvec[0],lvec[-1],mvec[0],mvec[-1]),interpolation='nearest')
    plt.colorbar(label='rad')
    plt.xlabel('l')
    plt.ylabel('m')
    frame = ac.AltAz(location = x0, obstime = obstime, pressure=None, copy = True) 
    s_ = s.transform_to(frame)    
    alt_s = s_.alt.deg#of pointing
    az_s = s_.az.deg
    plt.title("Time: {0}, Alt: {1:.0f}, Az: {2:.0f}".format(obstime.isot,alt_s,az_s))
    f.savefig("figs/fig_{0:04d}".format(num))
    plt.close()
    
    
if __name__=='__main__':
    l=0.0
    m=0.0
    obstime = at.Time('2000-01-01T00:00:00.000',format='isot',scale='utc')
    c0 = ac.ITRS(*ac.EarthLocation(lon=0*au.deg,lat=0*au.deg,height=0*au.m).geocentric)
    xi = ac.ITRS(*ac.EarthLocation(lon=0*au.deg,lat=0.001*au.deg,height=0*au.m).geocentric)
    s = ac.SkyCoord(ra=90*au.deg,dec=0*au.deg,frame='icrs')
    xvec = np.linspace(c0.cartesian.x.value,c0.cartesian.x.value*2,100)
    yvec = np.linspace(-c0.cartesian.x.value/2.,c0.cartesian.x.value/2.,100)
    zvec = np.linspace(-c0.cartesian.x.value/2.,c0.cartesian.x.value/2.,100)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    #ndata = 1 + 0.1*np.cos(R/60000.)
    frame = ac.AltAz(location = c0, obstime = obstime, pressure=None, copy = True) 
    s_ = s.transform_to(frame)    
    x0 = [(c0.cartesian.x.value+s_.cartesian.x.value*350000)]#*np.cos(c0.spherical.lon.rad+0.1)*np.sin(np.pi/2-c0.spherical.lat.rad)]
    y0 = [(c0.cartesian.y.value+s_.cartesian.y.value*350000)]#*np.sin(c0.spherical.lon.rad)*np.sin(np.pi/2-c0.spherical.lat.rad)]
    z0 = [(c0.cartesian.z.value+s_.cartesian.z.value*350000)]#*np.cos(np.pi/2-c0.spherical.lat.rad)]
    a = [1.]
    bx=[3500000]
    by=[3500000]
    bz=[3500000]
    params = np.array([x0,y0,x0,a,bx,by,bz]).transpose()
    NObj = NObject(params)
    rvec = np.linspace(xi.spherical.distance.m,4*xi.spherical.distance.m,10)
    thetavec = np.linspace(np.pi/2.-xi.spherical.lat.rad-0.5,np.pi/2.-xi.spherical.lat.rad+0.5,10)
    phivec = np.linspace(xi.spherical.lon.rad-.5,xi.spherical.lon.rad+.5,10)
    R,Theta,Phi = np.meshgrid(rvec,thetavec,phivec)
    X = R*np.cos(Phi)*np.sin(Theta)
    Y = R*np.sin(Phi)*np.sin(Theta)
    Z = R*np.cos(Theta)
    dnu = np.zeros_like(X)
    dnv = np.zeros_like(X)
    dnw = np.zeros_like(X)
    n = np.ones_like(X)
    i = 0
    while i < X.shape[0]:
        j = 0
        while j < X.shape[1]:
            k = 0
            while k < X.shape[2]:
                n[i,j,k] = NObj.compute_zeroth(X[i,j,k],Y[i,j,k],Z[i,j,k])
                dnu[i,j,k],dnv[i,j,k],dnw[i,j,k] = NObj.compute_oneth(X[i,j,k],Y[i,j,k],Z[i,j,k])
                #print dnu[i,j,k]
                k += 1
            j += 1
        i += 1
    from mpl_toolkits.mplot3d import axes3d
    import pylab as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.quiver(X,Y,Z,dnu/np.max(dnu),dnv/np.max(dnu),dnw/np.max(dnu),length=1e7/4.)
    ax.scatter(X,Y,Z,c=n)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ndata = 0.95+0.1*np.random.uniform(size=[100,100,100])
    #NObj = NObject(ndata,xvec,yvec,zvec)
    
    propagateBackwards(l,m,s,c0,xi,obstime,NObj,3,plot=True,ax=ax)
    lvec = np.linspace(-0.5,0.5,10)
    mvec = np.linspace(-0.5,0.5,10)
    import os
    try:
        os.mkdir('./figs')
    except:
        pass
    obstimes = at.Time(np.linspace(obstime.gps,obstime.gps+1*60*60,10),format='gps',scale='utc')
    c = 0
    for obstime in obstimes:
        plotPathLength(lvec,mvec,s,c0,xi,obstime,NObj,100,num=c)
        c += 1
    
    
    
            
        
        
        
 
        


# In[147]:

"{0:04d}".format(4)


# In[ ]:



