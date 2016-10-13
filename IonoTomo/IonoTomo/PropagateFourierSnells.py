
# coding: utf-8

# In[68]:

from NObject import NObject, noAtmosphere
import numpy as np
import pylab as plt
from scipy.signal import resample
from scipy.interpolate import griddata

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

def fft(A):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(A)))

def ifft(A):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(A)))

def fresnelZone(wavelength,dist):
    return 0.5*np.sqrt(wavelength*dist)

def fresnelSep(wavelength,diffractiveScale):
    return (diffractiveScale*2)**2/wavelength

def propagate(x,y,z,layers,wavelength,L,M,NObj):
    
    numLayers = len(layers.keys())
    #propagate 1,0 from ground to sky
    #first find x,y offset
    
    #Ground layer
    X = x
    Y = y
    
    n0sintheta0 = np.sqrt(L**2+M**2)
    mask = n0sintheta0 == 0
    Cl = np.ones_like(L)
    Cm = np.ones_like(M)
    Cl = L/np.sqrt(L**2+M**2)
    Cm = M/np.sqrt(L**2+M**2)
    Cl[mask] = 1.
    Cm[mask] = 1.
    layer = 0
    Ms = []
    Mp = []
    #setup directions to project
    for l,m in zip(L.flatten(),M.flatten()):
        Ms.append(np.eye(2))
        Mp.append(np.eye(2))
    n1 = np.ones_like(L)#vacuum base
    while layer < numLayers:
        X = X - layers[layer]['width']*n0sintheta0/n1 * Cl
        Y = Y - layers[layer]['width']*n0sintheta0/n1 * Cm
        #print x0
        n2 = n1
        #print n2,n1
        #for each direction build,Mn s/p
        if layer == numLayers-1:
            n1 = np.ones_like(n1sintheta1)
        else:
            n1 = NObj.compute_n(X,Y,layers[layer+1]['height'])
        #print n1/n2
        n1sintheta1 = n0sintheta0
        n2sintheta2 = n0sintheta0
        sintheta1 = n1sintheta1/n1
        sintheta2 = n2sintheta2/n2
        costheta2 = np.sqrt(1-sintheta2**2)
        costheta1 = np.sqrt(1-sintheta1**2)
        n2costheta2 = n2*costheta2
        n1costheta1 = n1*costheta1
        n1costheta2 = n1*costheta2
        n2costheta1 = n2*costheta1
        #print n2costheta2,n1costheta1,costheta1,sintheta2,costheta2,n1
        #fresnel equations
        rnn1s = (n1costheta1 - n2costheta2)/(n1costheta1 + n2costheta2)
        tnn1s = 2*n1costheta1/(n1costheta1 + n2costheta2)
        rnn1p = (n2costheta1 - n1costheta2)/(n2costheta1 + n1costheta2)
        tnn1p = 2*n1costheta1/(n2costheta1 + n1costheta2)
        #print rnn1s,tnn1s,rnn1p,tnn1p
        if layer == numLayers-1:
            deltan = np.zeros_like(L)
        else:
            deltan = layers[layer+1]['width']*2*np.pi/wavelength*n1costheta1
        i = 0
        for rnn1si,rnn1pi,tnn1si,tnn1pi,deltani in zip(rnn1s.flatten(),rnn1p.flatten(),tnn1s.flatten(),tnn1p.flatten(),deltan.flatten()):
            Ms[i] = (np.array([[np.exp(-1j*deltani),0],[0,np.exp(1j*deltani)]]).dot(np.array([[1,rnn1si],[rnn1si,1]]))/tnn1si).dot(Ms[i])
            Mp[i] = (np.array([[np.exp(-1j*deltani),0],[0,np.exp(1j*deltani)]]).dot(np.array([[1,rnn1pi],[rnn1pi,1]]))/tnn1pi).dot(Mp[i])
            i += 1
        layer += 1

    ts = []
    tp = []

    i = 0
    for l,m in zip(L.flatten(),M.flatten()):
        ts.append(1./Ms[i][0,0])
        tp.append(1./Mp[i][0,0])
        i += 1
    ts = np.reshape(ts,L.shape)
    tp = np.reshape(tp,L.shape)
    return ts,tp#*np.exp(1j*layers[layer-1]['height']/wavelength*np.sqrt(1j - L**2 - M**2)),tp*np.exp(1j*layers[layer-1]['height']/wavelength*np.sqrt(1j - L**2 - M**2))
    
def makeLayers(widths,numLayers):
    layer = 1
    layers = {}
    layers[0] = {'height':0,'width':widths}
    while layer < numLayers:
        layers[layer] = {'width' : widths,
                        'height' : layers[layer-1]['height']+layers[layer-1]['width']}
                        #'n':1+0.1*np.exp(-((X)**2+Y**2+(width*layer-1000-imi*F1/15.)**2)/50.)+0.1*np.exp(-((X)**2+(Y-5*F1/15.)**2+(width*layer-1000)**2)/50.)+0.0001*np.random.uniform(size=X.shape)}#m
        layer += 1
    return layers

def computeVisibilities(xvec,yvec,zvec,lvec,mvec,avec,NObj,wavelength):
    N = len(xvec)
    i = 0
    Us,Up = [],[]
    while i < N:
        #antenna based gains in lvec,mvec direction
        ts,tp = propagate(xvec[i],yvec[i],zvec[i],layers,wavelength,lvec,mvec,NObj)
        #could also apply reception pattern
        Us.append(np.sum(ts*avec*np.exp(1j*np.pi*2*(lvec*xvec[i]+mvec*yvec[i])/wavelength)))
        Up.append(np.sum(tp*avec*np.exp(1j*np.pi*2*(lvec*xvec[i]+mvec*yvec[i])/wavelength)))
        i += 1
    
    Vs = np.outer(Us,np.conj(Us))
    Vp = np.outer(Up,np.conj(Up))
    U = np.zeros([N,N])
    V = np.zeros([N,N])
    i = 0
    while i < N:
        j = i+1
        while j < N:
            U[i,j] = xvec[i]-xvec[j]
            U[j,i] = -U[j,i] 
            V[i,j] = yvec[i]-yvec[j]
            V[j,i] = -V[j,i]    
            j += 1
        i += 1
    
    return U,V,Vs,Vp
    
def simVis(params,args):
    shape = args[0]
    xvec,yvec,zvec = args[1],args[2],args[3]#antenna positions
    lvec,mvec,avec = args[4],args[5],args[6]#model
    Vs_true,Vp_true = args[7],args[8]
    NObj = NObject(np.reshape(params,shape))
    Vs,Vp = computeVisibilities(xvec,yvec,zvec,lvec,mvec,avec,NObj)
    chi = np.mean(np.angle(Vs_true - Vs) + np.angle(Vp_true - Vp))
    print chi
    return chi
    
def noAtmosphere():
    x0 = [0]#*np.cos(c0.spherical.lon.rad+0.1)*np.sin(np.pi/2-c0.spherical.lat.rad)]
    y0 = [0.]#*np.sin(c0.spherical.lon.rad)*np.sin(np.pi/2-c0.spherical.lat.rad)]
    z0 = [0]#*np.cos(np.pi/2-c0.spherical.lat.rad)]
    a = [0.]
    bx=[1]
    by=[1]
    bz=[1]
    params = np.array([x0,y0,z0,a,bx,by,bz]).transpose()
    NObj = NObject(params)
    return NObj


   
        
def dft(L,M,U,V,Vis):
    I = np.zeros_like(L)*1j
    i = 0
    while i < np.size(U):
        I += Vis[i]*np.exp(-1j*np.pi*2*(L*U[i] + M*V[i]))
        i += 1
    return I
    

if __name__=='__main__':
    plotIntensity()
    #test that ti*tj.cong = 1 for all i,j and for all l,m
    x = np.array([1,2,3])
    y = np.array([4,5,6])
    z = np.array([0,0,0])
    wavelength = 1.
    l = np.array([0])
    propagate(x,y,z,layers,wavelength,L,M,NObj)
    
    lmin = 100/60.*np.pi/180.#radians
    psf = 1/60.*np.pi/180./2.
    l = np.linspace(-lmin,lmin,1000)
    dl = np.abs(l[1]-l[0])
    uvec = np.linspace(-2/dl,2/dl,1000)
    U,V = np.meshgrid(uvec,uvec)
    R = np.sqrt(U**2+V**2)
    L,M = np.meshgrid(l,l)
    I = np.exp(-((L-0.001)**2 + (M-0.00)**2)/psf**2)
    Vis = ifft(I)
    plt.scatter(R.flatten(),np.abs(Vis).flatten())
    plt.show()
    plt.imshow(np.angle(Vis),extent=[uvec[0],uvec[-1],uvec[0],uvec[-1]])
    plt.show()
    plt.imshow(I,extent=[l[0],l[-1],l[0],l[-1]])
    plt.show()
    wavelength = 1.
    width = 1000#fresnelSep(wavelength,2000)
    numLayers = int(100000./width)
    numLayers = 1
    print("Number of layers:",numLayers)
    F1 = fresnelZone(wavelength,width)
    print ("Fresnel zone:",F1)
    layers = makeLayers(width,numLayers)
    print("Width",width)
    #for each component in I propagate through an atmosphere
    #antennas
    numAntennas = 7
    maxUV = 2./dl
    xvec = np.sort(np.random.uniform(size=numAntennas))*maxUV*wavelength
    yvec = np.sort(np.random.uniform(size=numAntennas))*maxUV*wavelength
    zvec = np.random.uniform(size=numAntennas)*0
    #atmosphere
    NObj = noAtmosphere()    
    lvec = L[I>1e-2].flatten()
    mvec = M[I>1e-2].flatten()
    avec = np.sqrt(I[I>1e-2].flatten())
    #print lvec,mvec,avec
    U2,V2,Vs,Vp = computeVisibilities(xvec,yvec,zvec,lvec,mvec,avec,NObj,wavelength)
    R2 = np.sqrt(U2**2+V2**2)
    plt.scatter(R2.flatten(),np.abs(Vs).flatten())
    plt.show()

    dl = 1/np.max(np.abs(U[U!=0]))
    print dl
    lvec = np.linspace(-dl*1000,dl*1000,2000)
    L,M = np.meshgrid(lvec,lvec)

    plt.imshow(np.abs(Vs))
    plt.show()
    
    img = dft(L,M,U.flatten(),V.flatten(),Vs.flatten())
    print img
    plt.imshow(np.abs(img))
    plt.colorbar()
    plt.show()
    
    
        


# In[81]:

def makeTransferMatrix(X,Y,n1,n2,width1,n0sintheta0,wavelength):
    ''' Make M(n) [x,y,pol,pol]
    n1,n2 are same shape as X,Y
    n1 is layer above, and n2 is layer below.
    n0sintheta0 is np.sqrt(L**2+M**2) (outside in vacuum)
    width1 is width of layer 1, above'''
    
    theta2 = np.arcsin(n0sintheta0/n2)
    theta1 = np.arcsin(n0sintheta0/n1)
    costheta1 = np.cos(theta1)
    costheta2 = np.cos(theta2)
    n1costheta1 = n1*costheta1
    n2costheta1 = n2*costheta1
    n2costheta2 = n2*costheta2
    n1costheta2 = n1*costheta2
    
    rnn1s = (n1costheta1 - n2costheta2)/(n1costheta1 + n2costheta2)
    tnn1s = 2*n1costheta1/(n1costheta1 + n2costheta2)
    rnn1p = (n2costheta1 - n1costheta2)/(n2costheta1 + n1costheta2)
    tnn1p = 2*n1costheta1/(n2costheta1 + n1costheta2)
        
    if np.isinf(width1):
        deltan = np.zeros_like(X)
    else:
        deltan = width1*2*np.pi/wavelength*n1costheta1
    Ms = np.zeros([X.shape[0],X.shape[1],4])
    D = np.exp(-1j*deltan)
    Dc = D.conjugate()
    #[D*1, D*rnn1s//D.c*rnn1a, D.c]/tnn1s
    Ms = np.array([[D,D*rnn1s],[Dc*rnn1s,Dc]])/tnn1s
    Ms = np.rollaxis(np.rollaxis(Ms,3),3)
    Mp = np.array([[D,D*rnn1p],[Dc*rnn1p,Dc]])/tnn1p
    Mp = np.rollaxis(np.rollaxis(Mp,3),3)
    return Ms,Mp
    
    
def fourierPropUp(X,Y,layers,NObj,wavelength):
    '''X and Y are NxM (usually square)'''
    # A up is 10
    # at each pixel
    Z = np.zeros_like(X)
    #makes NxMx2x2
    Umax = np.max(X)*2./wavelength
    Vmax = np.max(Y)*2./wavelength
    dl = 2./Umax
    dm = 2./Vmax
    l = np.linspace(-1./np.sqrt(2),1./np.sqrt(2),X.shape[0])
    m = np.linspace(-1./np.sqrt(2),1./np.sqrt(2),Y.shape[1])
    L,M = np.meshgrid(l,m)
    #in space this is true where n0=1
    n0sintheta0 = np.sqrt(L**2 + M**2)
    
    Aprev = np.ones_like(L)
    layerIdx = 0
    while layderIdx < len(layers.keys()-1):
        Z = layers[layerIdx+1]['height']
        n2 = NObj.compute_n(X,Y,Z-layers[layerIdx]['width']/2.)
        n1 = NObj.compute_n(X,Y,Z+layers[layerIdx+1]['width']/2.)
        Mns,Mnp = makeTransferMatrix(X,Y,n1,n2,layers[1]['width'],n0sintheta0,wavelength)
        Us = np.dot(Mns,Aprev)
        Up = np.dot(Mnp,Aprev)
        
    print Mns.shape
    
x = np.linspace(0,1000,100)
y = np.linspace(0,1000,100)
X,Y = np.meshgrid(x,y)
wavelength = 1.
width = 100*wavelength#fresnelSep(wavelength,2000)
numLayers = int(10000./width)
print("Number of layers:",numLayers)
layers = makeLayers(width,numLayers)
print("Width",width)
NObj = noAtmosphere()
fourierPropUp(X,Y,layers,NObj,wavelength)
    


# In[23]:

def plotIntensity():
    xvec = np.array([0,0.5,1])
    yvec = np.array([0,0,0])
    zvec = np.array([0,0,0])
    lvec = np.linspace(-0.5,0.5,100)
    L,M = np.meshgrid(lvec,lvec)
    NObj = noAtmosphere()
    wavelength = 1.
    width = 10000*wavelength#fresnelSep(wavelength,2000)
    numLayers = int(10000./width)
    numLayers = 10
    print("Number of layers:",numLayers)
    F1 = fresnelZone(wavelength,width)
    print ("Fresnel zone:",F1)
    layers = makeLayers(width,numLayers)
    print("Width",width)
    frame = 0
    while frame < 20:
        x0 = [frame*width*0.01]#*np.cos(c0.spherical.lon.rad+0.1)*np.sin(np.pi/2-c0.spherical.lat.rad)]
        y0 = [0.]#*np.sin(c0.spherical.lon.rad)*np.sin(np.pi/2-c0.spherical.lat.rad)]
        z0 = [width]#*np.cos(np.pi/2-c0.spherical.lat.rad)]
        a = [0.1]
        bx=[width/2.]
        by=[width/2.]
        bz=[width/2.]
        params = np.array([x0,y0,z0,a,bx,by,bz]).transpose()
        NObj = NObject(params)
        ts1,tp1 = propagate(xvec[0],yvec[0],zvec[0],layers,wavelength,L,M,NObj)
        f = plt.figure()
        plt.imshow(np.angle(ts1),origin='lower')
        plt.colorbar(label='phase (rad)')
        plt.xlabel('l')
        plt.ylabel('m')
        print ("saving:",frame)
        f.savefig('figs/ts1_phase_{0:04d}.png'.format(frame))
        #plt.show()
        plt.close()
        #ts2,tp2 = propagate(xvec[1],yvec[1],zvec[1],layers,wavelength,L,M,NObj)
        frame += 1


# In[80]:

help(np.tensordot)


# In[70]:

a = np.array([4])
a.shape.append(4)


# In[ ]:



