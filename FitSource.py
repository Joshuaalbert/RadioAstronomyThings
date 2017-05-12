
# coding: utf-8

# In[2]:

from astropy.wcs import WCS
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
from astropy.io import fits
import pylab as plt
import numpy as np
from matplotlib.patches import Ellipse
from astropy.modeling.models import Ellipse2D

def ellipse(x,y,x0,y0,bmaj,bmin,bpa,peak):
    res = np.zeros_like(x,dtype=np.double)
    dx = x-x0
    dy = y-y0
    res += ((dy*np.cos(-bpa) + dx*np.sin(-bpa))/(bmaj/2.))**2 + ((-dy*np.sin(-bpa) + dx*np.cos(-bpa))/(bmin/2.))**2
    res *= np.log(0.5)
    np.exp(res,out=res)
    res *= peak
    return res
    
fitsfile = 'plckg004-19_150_multiscale.pybdsm_gaus_resid.fits'

hdu = fits.open(fitsfile)[0]
deg2pix = 1./np.sqrt(hdu.header['CDELT1']**2 + hdu.header['CDELT2']**2)
wcs = WCS(hdu.header)

#plt.imshow(hdu.data[0,0,:,:],origin='lower', cmap=plt.cm.viridis,vmin=-3*1.4e-3,vmax=5*1.4e-3)
#plt.show()

bmaj = hdu.header["BMAJ"]#deg
bmin = hdu.header["BMIN"]#deg
bpa = hdu.header["BPA"]#deg


x = np.arange(hdu.data.shape[3])
y = np.arange(hdu.data.shape[2])
Y,X = np.meshgrid(y,x,indexing='ij')
x0,y0 = 289.28456, -33.52576
#289.28472222222223 -33.52583333333333
c = ac.SkyCoord('289d17m05s','-33d31m33s',frame='icrs')
x0 = c.ra.deg
y0 = c.dec.deg
print("Center fixed to: {}".format(c))
crd = wcs.wcs_world2pix(((x0,y0,0,0),),0)
x0_,y0_ = crd[0,0:2]
tb = (int(x0_-bmaj*deg2pix/2.),int(x0_+bmaj*deg2pix/2.),int(y0_-bmaj*deg2pix/2.),int(y0_+bmaj*deg2pix/2.))
print(tb)
tb = (int(x0_-4),int(x0_+4),int(y0_-4),int(y0_+4))
tb = (853, 860 ,855, 868)
print(tb)
data = hdu.data[0,0,tb[2]:tb[3]+1,tb[0]:tb[1]+1]
X_t = X[tb[2]:tb[3]+1,tb[0]:tb[1]+1]
Y_t = Y[tb[2]:tb[3]+1,tb[0]:tb[1]+1]


beam0 = bmaj*bmin*np.pi/4./np.log(2)
rms = 1.4e-3
CdCt = rms**2 + (np.abs(data)*0.05)**2
std_CdCt = np.sqrt(CdCt)

def L(X,Y,peak,x0_=x0_,y0_=y0_,bmaj=bmaj,bmin=bmin,bpa=bpa):
    beam = bmaj*bmin*np.pi/4./np.log(2)
    e = ellipse(X,Y,x0_,y0_,bmaj*deg2pix,bmin*deg2pix,bpa*np.pi/180.,peak)
    L1 = np.sum(np.abs(e - data)/(std_CdCt*np.sqrt(beam/beam0)))
    #L2 = np.sum((e - data)**2/CdCt)
    
    return L1
    return np.exp(-L1)

#initial
paramName=['Peak','x0','y0','bmaj','bmin']
peaki = np.max(data)
peaki = 0.031
xi = x0_
yi = y0_
bmaji = bmaj
bmini = bmin

N = int(1e6)
params = np.zeros([N,5],dtype=np.double)
params[0,:] = peaki,xi,yi,bmaji,bmini

Li = L(X_t,Y_t,*params[0,:])
maxL = Li
maxParams = np.zeros(5,dtype=np.double)
maxParams = params[0,:]
print("Initial L: {}".format(Li))
i = 1
accepted = 0
bins = 50
while accepted < bins**2 and i < N:
    #peakj = np.exp(np.random.uniform(low=np.log(peaki/2.),high=np.log(peaki*2.)))
    peakj = 0.031 + rms*np.random.normal()
    xj = x0_#np.random.uniform(low = x0_ - 0.25,high = x0_ + 0.25)
    yj = y0_#np.random.uniform(low = y0_ - 0.25, high = y0_ + 0.25)
    bmajj = bmaj#np.random.uniform(low=bmaj,high=bmaj*1.1)
    bminj = bmin#np.random.uniform(low=bmin,high=bmin*1.1)
    Lj = L(X_t,Y_t,peakj,x0_=xj,y0_=yj,bmaj=bmajj,bmin=bminj)
    if Lj > Li or np.random.uniform() < np.exp(Lj-Li):
        peaki = peakj
        xi = xj
        yi = yj
        bmaji = bmajj
        bmini = bmini
        params[i,:] = peaki,xi,yi,bmaji,bmini
        Li = Lj
        #print("acceptance")
        accepted += 1
    else:
        params[i,:] = peaki,xi,yi,bmaji,bmini
    if Lj > maxL:
        maxL = Lj
        maxParams[:] =  peakj,xj,yj,bmajj,bminj
    i += 1
params = params[:i,:]
if accepted == bins**2:
    print("converged in {} steps, {} acceptenace rate".format(i,float(accepted)/i))
else :
    print("No convergence, {} acceptenace rate".format(i,float(accepted)/i))

print("Max Likelihood: {}".format(maxL))
for i in range(5):
    plt.hist(params[:,i],bins=bins)
    plt.title(paramName[i])
    plt.show()
    print("{} = {} +- {}".format(paramName[i],np.mean(params[:,i]),np.std(params[:,i])))
    print("{} MAP = {}".format(paramName[i],maxParams[i]))


params_mean = np.mean(params,axis=0)

e1 = ellipse(X,Y,params_mean[1],params_mean[2],
             params_mean[3]*deg2pix,params_mean[4]*deg2pix,bpa*np.pi/180.,params_mean[0])


ax = plt.subplot(projection=wcs,slices=('x','y',0,0))
ax.imshow(hdu.data[0,0,:,:],origin='lower', cmap=plt.cm.viridis,vmin=-3*1.4e-3,vmax=5*1.4e-3)
ax.imshow(e1,origin='lower', cmap=plt.cm.viridis,vmin=-3*1.4e-3,vmax=5*1.4e-3,alpha=0.5)

e2 = Ellipse(xy=crd[0,0:2], width=bmin*deg2pix, height=bmaj*deg2pix, angle = bpa, edgecolor='red',
                      facecolor='none')
ax.add_artist(e2)  
ax.scatter(x0_,y0_)

plt.show()

ax = plt.subplot(projection=wcs,slices=('x','y',0,0))
ax.imshow(hdu.data[0,0,:,:]-e1,origin='lower', cmap=plt.cm.viridis,vmin=-3*1.4e-3,vmax=5*1.4e-3)
#ax.imshow(e1,origin='lower', cmap=plt.cm.viridis,vmin=-3*1.4e-3,vmax=5*1.4e-3,alpha=0.5)

e2 = Ellipse(xy=crd[0,0:2], width=bmin*deg2pix, height=bmaj*deg2pix, angle = bpa, edgecolor='red',
                      facecolor='none')
ax.add_artist(e2)  
ax.scatter(x0_,y0_)

plt.show()
beam = bmaj*bmin*np.pi/4./np.log(2)

tb = (int(x0_-bmaj*deg2pix*3),int(x0_+bmaj*deg2pix*3),int(y0_-bmaj*deg2pix*3),int(y0_+bmaj*deg2pix*3))
X_t = X[tb[2]:tb[3]+1,tb[0]:tb[1]+1]
Y_t = Y[tb[2]:tb[3]+1,tb[0]:tb[1]+1]

flux = np.zeros(params.shape[0])
i = 0
while i < params.shape[0]:
    e1 = ellipse(X_t,Y_t,params[i,1],params[i,2],
             params[i,3]*deg2pix,params[i,4]*deg2pix,bpa*np.pi/180.,params[i,0])
    flux[i] = (np.sum(e1)/deg2pix**2/beam*1e3)
    i += 1
print(np.mean(flux),np.std(flux))


# In[58]:

hdu.header


# In[62]:

hdu.header['BMAJ'] / hdu.header['CDELT1']


# In[ ]:



