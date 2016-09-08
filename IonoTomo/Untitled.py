
# coding: utf-8

# In[20]:

import numpy as np
import pylab as plt
from scipy.signal import resample,correlate2d
from scipy.ndimage.filters import gaussian_filter

def fft(A,n=None):
    '''exp{-i2pi (lu + mv)}'''
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(A),s=n))

def ifft(A,n=None):
    '''exp{i2pi (lu + mv)}'''
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(A),s=n))

def conv2d(A,B):
    if np.size(A) != np.size(B):
        print("wrong sizes")
        return
    return np.fft.ifft2(np.fft.fft2(A)*np.fft.fft2(B))

def autocorr(A):
    F = fft(A)
    return ifft(F*np.conjugate(F))

def regrid(A,shape,*presampledAx):
    '''Uses fft to regrid ...'''
    n = len(shape)
    if len(presampledAx) != n:
        print("wrongsize sample axis")
        return
    B = np.copy(A)
    resampledAx = []
    i = 0
    while i < n:
        B,t = resample(B,shape[i],t=presampledAx[i],axis=i)
        resampledAx.append(t)
        i += 1
    return B,resampledAx

def dft2(A,L,M,x,y):
    res = np.zeros_like(L,dtype=type(1j))*1j

    n = 0
    while n < np.size(x):
        print(n/np.size(x))
        p = 0
        while p < np.size(y):
            res += A[n,p]*np.exp(-1j*2.*np.pi*(x[n]*L + y[p]*M))
            #print(res[i,j])
            p += 1
        n += 1
       
           
    return res

def complexGaussianFilter(A,sigma=3,order=0):
    return gaussian_filter(np.real(A),sigma=sigma,order=order) + 1j*gaussian_filter(np.imag(A),sigma=sigma,order=order)

'''propagate distortions'''
l = np.linspace(-0.5,0.5,100)

A = np.exp(-1j*2*np.pi*np.sqrt(1-l**2))
U = ifft(A)
#plt.plot(U)
#plt.show()
#plt.plot(fft(U))
#plt.show()
A = np.random.uniform(size=[5,5,5])
x = np.linspace(0,1,5)
B,y = regrid(A,[10,11,120],*(x,x,x))
print (B)
print (y)
#plt.imshow(B[0,:,:],interpolation='nearest')
#plt.show()
w = 100000
up = np.linspace(-10,10,1000)
dx=np.abs(up[1]-up[0])
U,V = np.meshgrid(up,up)
l = np.fft.fftshift(np.fft.fftfreq(1000,d=dx))
dl = l[1]-l[0]
m = l
L,M = np.meshgrid(l,m)
u = 0
v = 0
#k = np.exp(1j*2*np.pi*w)*np.exp(-1j*np.pi*w*(U**2+V**2))
k = complexGaussianFilter(np.exp(1j*2*np.pi*w*(-2/w**2*(u*U + v*V) + V**2/w**2 + U**2/w**2)))
#k = w/(1j*distance)*exp(1j*2*np.pi*0.5/distance*Z)
#Af = dft2(k,L,M,up,up)*dx**2

Af = fft(k)*dx**2
A = complexGaussianFilter(1j*w/2.*np.exp(-1j*np.pi/2./w*(4*L*u*w + 4*M*v*w + 4*(u**2+v**2) + (L**2 + M**2)*w**2)))
kf = ifft(A/dx**2)
import pylab as plt
#print(np.mean(np.abs(kf)))
plt.imshow(np.angle (Af),interpolation='nearest',origin='lower')
plt.colorbar()
plt.show()
    


# In[6]:

help(plt.imshow
)


# In[116]:

x = np.linspace(0,1,5)
X,Y = np.meshgrid(x,x)
l = np.fft.fftfreq(5,x[1]-x[0])
print (l)


# In[ ]:



