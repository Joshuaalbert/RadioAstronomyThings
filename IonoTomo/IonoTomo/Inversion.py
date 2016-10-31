
# coding: utf-8

# In[22]:

from Geometry import *
import numpy as np

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
    #forward problem
    print("Doing forward problem")
    d = G.dot(mprior)
    print("Calculating residuals:")
    residuals = dobs - d
    print(residuals)
    Gt = G.transpose()
    #smooth and adjoint
    print("Calculating smoothing matrix")
    smooth = np.linalg.inv(G.dot(Cmprior).dot(Gt) + Cd)
    print("Calculating adjoint")
    adjoint = Cmprior.dot(Gt).dot(smooth)
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
    plotOctTreeXZ(octTree,ax=None)
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
    
def SteepestDescent(octTree,rays,dobs,Cd,Cmprior,mprior):
    iter = 0
    mn = mprior
    while iter < 100:
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
        print("Calculating weighting residuals")
        weightedRes = np.linalg.pinv(Cd).dot(residuals)
        #print(weightedRes,np.linalg.solve(Cd,residuals))
        #next part should be changed
        Gt = G.transpose()
        #smooth and adjoint
        print("Calculating adjoint")
        dm = Cmprior.dot(Gt).dot(weightedRes)
        print("updating model")
        mn = mn - 0.01*(dm + mn - mprior)
        iter += 1
    
    print("updating covariance")
    print("Calculating smoothing matrix")
    smooth = np.linalg.pinv(G.dot(Cmprior).dot(Gt) + Cd)
    print("Calculating adjoint")
    adjoint = Cmprior.dot(Gt).dot(smooth)
    Cm = Cmprior - adjoint.dot(G).dot(Cmprior)
    return mn,Cm  

def constructIonosphereModel(maxBaseline,):
    levels = 3
    fileName = "ionosphereModel_{0}levels.npy".format(levels)
    baseWidth = 1.
    height = 1000.#km
    try:
        octTree = loadOctTree(fileName)
        G,Cm,m,x = generateModelFromOctree(octTree,0,propName='Ne')
    except:
        octTree = OctTree([0,0,height/2.],dx=baseWidth,dy=baseWidth,dz=height)
        subDivideToDepth(octTree,levels)
        saveOctTree(fileName,octTree)
        G,Cm,m,x = generateModelFromOctree(octTree,0,propName='Ne')
    
    
if __name__=='__main__':
    #ExampleLinearSolution()
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
    M = 3
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
    M = 10
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
    mprior = mexact*0.5
    print("Solving for model from rays:")
    #m,Cm = LinearSolution(dobs,G,Cd,Cmexact,mprior)
    m,Cm = SteepestDescent(octTree,rays,dobs,Cd,Cmexact,mprior)
    CmCm = Cm.dot(np.linalg.inv(Cmexact))
    R = np.eye(CmCm.shape[0]) - CmCm
    print("Resolved by dataSet:{0}, resolved by a priori:{1}".format(np.trace(R),np.trace(CmCm)))
    import pylab as plt
    plt.plot(m,label='res')
    plt.plot(mexact,label='ex')
    plt.plot(mprior,label='pri')
    plt.legend(frameon=False)
    plt.show()
    plotOctTreeXZ(octTree,ax=None)
    plotOctTree3D(octTree,model=m)
   


# In[13]:

f,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True)
ax1.imshow(Cm)
ax2.imshow(Cmexact)
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


# In[ ]:



