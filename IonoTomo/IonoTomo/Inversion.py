
# coding: utf-8

# In[1]:

from Geometry import *
import numpy as np
#import lmfit as lf
from lmfit import Parameters, minimize

def getAllDecendants(octTree):
    '''Get all lowest chilren on tree.'''
    out = []
    if octTree.hasChildren:
        for child in octTree.children:
            out = out + getAllDecendants(child)
        return out
    else:
        return [octTree]
    
def generateModelFromOctree(octTree,numRays,propName='Ne'):
    '''Generate '''
    voxels = getAllDecendants(octTree)        
    N = len(voxels)
    G = np.zeros([numRays,N])
    m = np.zeros(N)
    Cm = np.zeros(N)
    x = np.zeros([N,3])
    if propName not in voxels[0].properties.keys():
        #zero model if no property
        i = 0
        while i < N:
            vox = voxels[i]
            for j in vox.lineSegments.keys():
                G[j,i] = vox.lineSegments[j].sep   
            x[i,:] = vox.centroid
            i += 1
        return G,Cm,m,x
    i = 0
    while i < N:
        vox = voxels[i]
        for j in vox.lineSegments.keys():
            G[j,i] = vox.lineSegments[j].sep   
        m[i] = vox.properties[propName][1]
        Cm[i] = vox.properties[propName][2]
        x[i,:] = vox.centroid
        i += 1
    return G,Cm,m,x
        
def setOctTreeModel(octTree,m,Cm,propName='Ne',propType='extensive'):
    '''Set the model in the octTree. 
    Assumes the model is derived from the same octTree and
    Cm is the diagonal of the covariance.'''
    voxels = getAllDecendants(octTree)
    N = len(voxels)
    i = 0
    while i < N:
        vox = voxels[i]
        vox.properties[propName] = [propType,m[i],Cm[i]]
        vox.lineSegments = {}
        i += 1
    
def forwardProblem(rays,octTree):
    '''Perform the forward problem.'''
    
    pass
    
def noRefractionTECInversion(antennaLocs,antennaTEC, sourcePos):
    '''Invert the 3d TEC based on total measurments on the ground (after factor)
    antennaLocs are in uvw coords of shape
    antennaTEC are NxM'''
    pass

def makeRaysFromSourceAndReciever(recievers,sources):
    rays = []
    count = 0
    for r in recievers:
        for s in sources:
            rays.append(LineSegment(r,s,id=count))
            count += 1
    return rays

def LinearSolution(dobs,G,Cd,Cmprior,mprior):
    '''Assumes d = int(G * m)'''
    #forward problem
    print("Doing forward problem")
    #d = np.log(G.dot(np.exp(mprior)))
    d = G.dot(mprior)
    print("Calculating residuals:")
    residuals = dobs - d
    Gt = G.transpose()
    #smooth and adjoint
    print("Calculating smoothing matrix")
    smooth = np.linalg.inv(G.dot(Cmprior).dot(Gt) + Cd)
    #print(smooth)
    print("Calculating adjoint")
    adjoint = Cmprior.dot(Gt).dot(smooth)
    #print(adjoint)
    print("updating model")
    m = mprior + adjoint.dot(residuals)
    print("updating covariance")
    Cm = Cmprior - adjoint.dot(G).dot(Cmprior)
    return m,Cm  
    
def compute3dExponentialCovariance(sigma2,L,x):
    N = x.shape[0]
    C = np.zeros([N,N])
    i = 0
    while i < N:
        C[i,i] = sigma2
        j = i+1
        while j < N:
            d = x[i,:] - x[j,:]
            C[i,j] = sigma2*np.exp(-np.linalg.norm(d)/L)
            C[j,i] = C[i,j]
            j += 1
        i += 1
    C[C<epsFloat] = 0.
    return C

def ExampleLinearSolution():
    levels = 3
    fileName = "octTree_{0}levels.npy".format(levels)
    baseWidth = 1.
    height = 1000.
    try:
        octTree = loadOctTree(fileName)
        G,Cm,m,x = generateModelFromOctree(octTree,0,propName='Ne')
    except:
        octTree = OctTree([0,0,height/2.],dx=baseWidth,dy=baseWidth,dz=height)
        subDivideToDepth(octTree,levels)
        saveOctTree(fileName,octTree)
        G,Cm,m,x = generateModelFromOctree(octTree,0,propName='Ne')
        
    
    mexact = np.zeros_like(m)
    M = 2
    print("Generating {0} component model".format(M))
    for i in range(M):
        loc = np.array([np.random.uniform(low = -baseWidth/2.,high = baseWidth/2.),
               np.random.uniform(low = -baseWidth/2.,high = baseWidth/2.),
               np.random.uniform(low = height/2.,high = height)])
        mexact += 0.3 + np.random.uniform()*np.exp(-np.sum((x - loc)**2,axis=1)/(height/8.)**2)
    modelVar = 0.1
    print("Computing model 3d exponential covariance")
    Cmexact = compute3dExponentialCovariance(modelVar,baseWidth/2.,x)
    print("Setting octTree with model.")
    setOctTreeModel(octTree,mexact,np.diag(Cmexact),propName='Ne',propType='extensive')
    M = 5
    print("Generating {0} recievers".format(M))
    recievers = []
    for i in range(M):
        recievers.append([np.random.uniform(low = -baseWidth/2.,high = baseWidth/2.),
               np.random.uniform(low = -baseWidth/2.,high = baseWidth/2.),
               -epsFloat])
    M = 10
    print("Generating {0} sources".format(M))
    sources = []
    for i in range(M):
        sources.append([np.random.uniform(low = -baseWidth/2.,high = baseWidth/2.),
               np.random.uniform(low = -baseWidth/2.,high = baseWidth/2.),
               height])
    rays = makeRaysFromSourceAndReciever(recievers,sources)
    print("Propagating {0} rays".format(len(rays)))
    for ray in rays:
        forwardRay(ray,octTree)
    print("Pulling ray propagations.")
    G,CmVar,mexact,x = generateModelFromOctree(octTree,len(rays),propName='Ne')
    
    #generate model
    print("Computing fake observations")
    dobs = []
    for i in range(20):
        dobs.append(G.dot(np.random.multivariate_normal(mean=mexact, cov = Cmexact)))
        #print(dobs[-1])
    dobs = np.array(dobs)
    Cd = np.cov(dobs.transpose())
    dobs = np.mean(dobs,axis=0)
    #print (dobs)
    mprior = mexact/0.5
    print("Solving for model from rays:")
    print("solution with Cm exact input")
    m,Cm = LinearSolution(dobs,G,Cd,Cmexact,mprior)
    import pylab as plt
    plt.plot(m,label='res')
    plt.plot(mexact,label='ex')
    plt.plot(mprior,label='pri')
    plt.legend(frameon=False)
    plt.show()
    plot3OctTreeXZ(octTree,ax=None)
    plotOctTree3D(octTree,model=m)
    f,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True)
    ax1.imshow(Cm)
    ax2.imshow(Cmexact)
    plt.show()
    print("solution with Cm diagonal")
    m,Cm = LinearSolution(dobs,G,Cd,np.diag(np.diag(Cmexact)),mprior)
    import pylab as plt
    plt.plot(m,label='res')
    plt.plot(mexact,label='ex')
    plt.plot(mprior,label='pri')
    plt.legend(frameon=False)
    plt.show()
    plotOctTreeXZ(octTree,ax=None)
    plotOctTree3D(octTree,model=m)
    f,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True)
    ax1.imshow(Cm)
    ax2.imshow(Cmexact)
    plt.show()

def transformCov2Log(Cmprior_linear,mprior_linear,K):
    '''Transform covariance matrix from linear model to log model using:
    cov(y1,y2) = <y1y2> - <y1><y2>
    with,
    y = log(x/K)
    thus,
    <y1y2> ~ y1y2 + 0.5*(var(x1)y1''y2 +var(x2)y2''y1) + cov(x1,x2)y1'y2' 
    = log(x1/K)log(x2/K) - 0.5*(var(x1)log(x2/K)/x1**2 +var(x2)log(x1/K)/x2**2) + cov(x1,x2)/x1/x2 
    and,
    <y1> ~ y1 + 0.5*var(x1)y1''
    = log(x1/K) - 0.5*var(x1)/x1**2
    '''
    mn = np.log(mprior_linear/K)
    Cmprior = np.zeros_like(Cmprior_linear)
    i = 0
    while i < mn.shape[0]:
        j = i
        while j < mn.shape[0]:
            Cmprior[i,j] = mn[i]*mn[j] - 0.5*(Cmprior_linear[i,i]*mn[j]/mprior_linear[i]**2 + Cmprior_linear[j,j]*mn[i]/mprior_linear[j]**2) + Cmprior_linear[i,j]/mprior_linear[i]/mprior_linear[j] - (mn[i] - 0.5*Cmprior_linear[i,i]/mprior_linear[i]**2)*(mn[j] - 0.5*Cmprior_linear[j,j]/mprior_linear[j]**2)
            Cmprior[j,i] = Cmprior[i,j]
            j += 1
        i += 1
    return Cmprior

def mcTransformCov2Log(m_linear,Cm_linear,K):
    
    res = []
    i = 0
    while i < 10:
        res.append(np.log(np.abs(np.random.multivariate_normal(mean=m_linear, cov = Cm_linear))/K))
        i += 1
    Cm_log = np.cov(np.transpose(res))
    print(Cm_log)
    m_log = np.mean(res,axis=0)
    return m_log,Cm_log

def mcTransformCov2Lin(m_log,Cm_log,K):
    
    res = []
    i = 0
    while i < 10:
        res.append(K*np.exp(np.random.multivariate_normal(mean=m_log, cov = Cm_log)))
        i += 1
    Cm_lin = np.cov(np.transpose(Cm_log))
    m_lin = np.mean(res,axis=0)
    return m_lin,Cm_lin

def transformCov2Linear(Cmprior_log,model_log,K):
    '''Transform covariance matrix from log model to ling model using:
    cov(x1,y2) = <y1y2> - <y1><y2>
    with,
    y = K*exp(x)
    thus,
    <y1y2> ~ y1y2 + 0.5*(var(x1)y1''y2 +var(x2)y2''y1) + cov(x1,x2)y1'y2' 
    = K*exp(x1)K*exp(x2)*(1 + 0.5*(var(x1) +var(x2)) + cov(x1,x2)) 
    and,
    <y1> ~ y1 + 0.5*var(x1)y1''
    = K*exp(x1)(1 + 0.5*var(x1))
    '''
    mn = K*np.exp(model_log)
    Cmprior = np.zeros_like(Cmprior_log)
    i = 0
    while i < mn.shape[0]:
        j = i
        while j < mn.shape[0]:
            Cmprior[i,j] = mn[i]*mn[j] * (1 + 0.5*(Cmprior_log[i,i] + Cmprior_log[j,j]) + Cmprior_log[i,j] - (1 + 0.5*Cmprior_log[i,i])*(1 + 0.5*Cmprior_log[j,j]))
            Cmprior[j,i] = Cmprior[i,j]
            j += 1
        i += 1
    return Cmprior
    
def SteepestDescent(octTree,rays,dobs,Cd,Cmprior,mprior):
    '''Assumes d = log(K*int(G * exp(m))) and that input is linear versions'''
    def updater(x,G):
        eps = np.zeros(x.shape[0])
        i = 0
        while i< x.shape[0]:
            if np.sum(G[:,i]) > 0:
                    eps[i] = 0.1
            else:
                eps[i] = 0.01
            i += 1
        return eps
    
    iter = 0
    mn = mprior
    Cmprior = Cmprior
    while iter < 10:
        #forward problem
        print("Setting octTree with model_{0}".format(iter))
        setOctTreeModel(octTree,mn,np.diag(Cmprior),propName='Ne',propType='extensive')
        print("Propagating {0} rays".format(len(rays)))
        for ray in rays:
            forwardRay(ray,octTree)
        print("Pulling ray propagations.")
        G,CmVar,mexact,x = generateModelFromOctree(octTree,len(rays),propName='Ne')
        print("Doing forward problem")
        d = G.dot(mn)
        print("Calculating residuals, Sum:")
        residuals = d - dobs
        print(np.sum(residuals**2))
        #print(residuals.shape)
        print("Calculating weighting residuals")
        weightedRes = np.linalg.pinv(Cd).dot(residuals)
        #print(weightedRes,np.linalg.solve(Cd,residuals))
        #next part should be changed
        #Gt.Cd^-1.(d-dobs)
        Gt = G.transpose()
        #smooth and adjoint
        print("Calculating adjoint")
        dm = Cmprior.dot(Gt).dot(weightedRes)
        print("updating model")
        mn = mn - updater(x,G)*(dm + mn - mprior)
        iter += 1
    
    print("updating covariance")
    print("Calculating smoothing matrix")
    smooth = np.linalg.pinv(G.dot(Cmprior).dot(Gt) + Cd)
    print("Calculating adjoint")
    adjoint = Cmprior.dot(Gt).dot(smooth)
    Cm = Cmprior - adjoint.dot(G).dot(Cmprior)
    return mn,Cm

def ionosphereModel(h,dayTime=True):
    Nf1 = 4*np.exp((h-300)/100.)/(1 + np.exp((h-300)/100.))**2
    if dayTime:
        Ne = 10**(-0.5)*4*np.exp((h-85.)/50.)/(1 + np.exp((h-85.)/50.))**2
        return Nf1 + Ne
    return Nf1

def constructIonosphereModel(maxBaseline):
    '''initialize with 1/m^3 at 300km +- 150km'''

    fileName = "ionosphereModel.npy"
    height = 1000.#km
    octTree = OctTree([0,0,height/2.],dx=maxBaseline,dy=maxBaseline,dz=height)
    #level 2 - all
    subDivide(subDivide(octTree))
    voxels = getAllDecendants(octTree)
    for vox in voxels:
        #level 3 - 250 to 750
        if (vox.centroid[2] > 250) and (vox.centroid[2] < 750):
            subDivide(vox)
        #level 4 - 250 to 500
        if (vox.centroid[2] > 250) and (vox.centroid[2] < 500):
            subDivide(vox)
    G,Cm,m,x = generateModelFromOctree(octTree,0,propName='Ne')
    i = 0
    while i < x.shape[0]:
        h = x[i,2]
        m[i] = ionosphereModel(h,dayTime=False)
        i += 1
    setOctTreeModel(octTree,m,np.ones_like(m)*0.1,propName='Ne',propType='extensive')
    saveOctTree(fileName,octTree)
    #plotOctTreeXZ(octTree,ax=None)
    #plotOctTree3D(octTree,model=m)
    return octTree
    
def residuals(params_,G,x,nComp,dobs=None):
    '''params contain the model'''
    params = params_.valuesdict()
    N = x.shape[0]
    m = np.zeros(N)
    i = 0
    while i < nComp:
        loc = np.array([params["x{0}".format(i)],params["y{0}".format(i)],params["z{0}".format(i)]])
        d = x - loc
        m += params["a{0}".format(i)] * np.exp(-np.sum(d*d,axis=1) / params["s{0}".format(i)] )
        i += 1
    d = G.dot(m)
    if dobs is None:
        return d
    res = d - dobs
    print (np.sum(res))
    return res

def initHomogeneousModel(G,dobs):
    return np.sum(dobs)/np.sum(G)
    
if __name__=='__main__':
    print("Constructing ionosphere model")
    maxBaseline = 100
    octTree = constructIonosphereModel(maxBaseline)
    M = 40
    print("Generating {0} recievers".format(M))
    recievers = []
    for i in range(M):
        recievers.append([np.random.uniform(low = -maxBaseline/2.,high = maxBaseline/2.),
               np.random.uniform(low = -maxBaseline/2.,high = maxBaseline/2.),
               -epsFloat])
    M = 15
    print("Generating {0} sources".format(M))
    sources = []
    for i in range(M):
        sources.append([np.random.uniform(low = -maxBaseline/2.,high =maxBaseline/2.),
               np.random.uniform(low = -maxBaseline/2.,high = maxBaseline/2.),
               1000.])
    rays = makeRaysFromSourceAndReciever(recievers,sources)
    print("Propagating {0} rays".format(len(rays)))
    for ray in rays:
        forwardRay(ray,octTree)
    print("Pulling ray propagations.")
    G,mVar_lin,mexact_lin,x = generateModelFromOctree(octTree,len(rays),propName='Ne')
    #ExampleLinearSolution()
    modelVar = np.mean(mVar_lin)
    print("Computing model 3d exponential covariance")
    Cmexact_lin = compute3dExponentialCovariance(modelVar,20.,x)
    #K = np.mean(mexact_lin)
    #mexact_log,Cmexact_log = mcTransformCov2Log(mexact_lin,Cmexact_lin,K)
    #generate model
    print("Computing fake observations")
    dobs = []
    for i in range(10):
        #dobs.append(np.log(G.dot(K*np.exp(np.random.multivariate_normal(mean=mexact_log, cov = Cmexact_log)))))
        dobs.append(G.dot(np.abs(np.random.multivariate_normal(mean=mexact_lin, cov = Cmexact_lin))))
        #print(dobs[-1])
    dobs = np.array(dobs)
    Cd = np.cov(dobs.transpose())
    #print(dobs)
    dobs = np.mean(dobs,axis=0)
    #dobs = G.dot(mexact_lin)
    #print (dobs)
    mprior = np.ones_like(mexact_lin)*initHomogeneousModel(G,dobs)
    mprior = np.random.multivariate_normal(mean=mexact_lin, cov = Cmexact_lin)
    Cmprior = Cmexact_lin
    print("Solving for model from rays:")
    #m,Cm = LinearSolution(dobs,G,Cd,Cmexact,mprior)
    m,Cm = SteepestDescent(octTree,rays,dobs,Cd,Cmprior,mprior)
    #params = vec2Parameters(mprior)
    #params = InitParameters(5)
    #res = minimize(residuals, params, args=(G,x,5), kws={"dobs":dobs})
    #print (res)
    #m,Cm = LinearSolution(dobs,G,Cd,Cmprior,mprior)
    CmCm = Cm.dot(np.linalg.inv(Cmexact_lin))
    R = np.eye(CmCm.shape[0]) - CmCm
    print("Resolved by dataSet:{0}, resolved by a priori:{1}".format(np.trace(R),np.trace(CmCm)))
    import pylab as plt
    plt.plot(m,label='res')
    plt.plot(mexact_lin,label='ex')
    plt.plot(mprior,label='pri')
    plt.legend(frameon=False)
    plt.show()
    plotOctTreeXZ(octTree,ax=None)
    plotOctTree3D(octTree,model=m)
   


# In[2]:

f,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True)
ax1.imshow(Cm)
ax2.imshow(Cmexact_lin)
plt.show()



# In[20]:

m,Cm = LinearSolution(dobs,G,Cd,Cmexact,mprior)
CmCm = Cm.dot(np.linalg.inv(Cmexact))
R = np.eye(CmCm.shape[0]) - CmCm
print("Resolved by dataSet:{0}, resolved by a priori:{1}".format(np.trace(R),np.trace(CmCm)))
import pylab as plt
plt.plot(m,label='res')
plt.plot(mexact,label='ex')
plt.plot(mprior,label='pri')
plt.legend(frameon=False)
plt.show()


# In[9]:

import lmfit


# In[ ]:



