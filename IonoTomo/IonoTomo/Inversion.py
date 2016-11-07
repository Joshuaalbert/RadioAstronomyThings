
# coding: utf-8

# In[1]:

from Geometry import *
import numpy as np
    
def generateModelFromOctree(octTree,numRays):
    '''Generate model '''
    voxels = getAllDecendants(octTree)        
    N = len(voxels)
    G = np.zeros([numRays,N])
    m = np.zeros(N)
    Cm = np.zeros(N)
    x = np.zeros([N,3])
    if 'ne' not in voxels[0].properties.keys():
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
        m[i] = vox.properties['ne'][1]
        Cm[i] = vox.properties['ne'][2]
        x[i,:] = vox.centroid
        i += 1
    return G,Cm,m,x

def electronDensity2RefractiveIndex(ne,frequency=120e6):
    #eCharge = 1.60217662e-19#C = F.V = W.s/V^2.V = kg.m^2/s^2/V
    #epsilonPerm = 8.854187817e-12#F/m = kg.m/s^2/V^2
    #eMass = 9.10938215e-31#kg
    #constant = eCharge**2*4*pi/eMass
    constant = 56.3*56.3#Hz^2 m^3 lightman p 226
    n = np.sqrt(1. - constant*ne/frequency**2)
    dndne = constant/frequency**2/n/2.
    return n,dndne
        
        
def setOctTreeElectronDensity(octTree,ne,neVar,frequency=120e6):
    '''Set the model in the octTree. 
    Assumes the model is derived from the same octTree and
    Cm is the diagonal of the covariance.'''
    voxels = getAllDecendants(octTree)
    N = len(voxels)
    i = 0
    while i < N:
        vox = voxels[i]
        vox.properties['ne'] = ['intensive',ne[i],neVar[i]]
        vox.properties['Ne'] = ['extensive',ne[i]*vox.volume,neVar[i]*vox.volume]
        n,dndne = electronDensity2RefractiveIndex(vox.properties['ne'][1],frequency)
        vox.properties['n'] = ['intensive',n,dndne**2*vox.properties['ne'][2]]
        vox.lineSegments = {}
        i += 1

def setOctTreeElectronNumber(octTree,Ne,NeVar,frequency = 120e6):
    '''Set the model in the octTree. 
    Assumes the model is derived from the same octTree and
    Cm is the diagonal of the covariance.'''
    voxels = getAllDecendants(octTree)
    N = len(voxels)
    i = 0
    while i < N:
        vox = voxels[i]
        vox.properties['ne'] = ['intensive',Ne[i]/vox.volume,NeVar[i]/vox.volume]
        vox.properties['Ne'] = ['extensive',Ne[i],NeVar[i]]
        n,dndne = electronDensity2RefractiveIndex(vox.properties['ne'][1],frequency)
        vox.properties['n'] = ['intensive',n,dndne*vox.properties['ne'][2]]
        vox.lineSegments = {}
        i += 1        

def makeRaysFromSourceAndReciever(recievers=None,directions=None,sources=None,maxBaseline = 100.,height=1000.,numSources=15,numRecievers=40):
    """make rays"""
    #make recievers
    if recievers is None:
        numRecievers = 40
        print("Generating {0} recievers".format(numRecievers))
        recievers = []
        for i in range(numRecievers):
            recievers.append(np.array([np.random.uniform(low = -maxBaseline/4.,high = maxBaseline/4.),
                   np.random.uniform(low = -maxBaseline/4.,high = maxBaseline/4.),
                   -epsFloat]))
            
    if sources is None:
        print("Generating {0} sources".format(numSources))
        sources = []
        for i in range(numSources):
            sources.append(np.array([np.random.uniform(low = -maxBaseline/4.,high =maxBaseline/4.),
                   np.random.uniform(low = -maxBaseline/4.,high = maxBaseline/4.),
                   height]))
    if directions is None:
        numDirections = numSources
        directions = []
        print("Generating {} directions".format(numDirections))
        for i in range(numDirections):
            mag = np.linalg.norm(sources[i])
            #direction cosines
            directions.append(sources[i]/mag)
    rays = []
    count = 0
    for r in recievers:
        for d in directions:
            rays.append(Ray(r,d,id=count))
            count += 1
    return rays

def compute3dExponentialCovariance(sigma,L,x):
    '''exponential covariance model'''
    N = x.shape[0]
    Cm = np.zeros([N,N])
    if np.size(sigma) == N:
        i = 0
        while i < N:
            Cm[i,i] = sigma[i]**2
            j = i+1
            while j < N:
                d = x[i,:] - x[j,:]
                Cm[i,j] = sigma[i]*sigma[j]*np.exp(-np.linalg.norm(d)/L)
                Cm[j,i] = C[i,j]
                j += 1
            i += 1
    else:
        sigma2 = sigma*sigma
        i = 0
        while i < N:
            Cm[i,i] = sigma2
            j = i+1
            while j < N:
                d = x[i,:] - x[j,:]
                Cm[i,j] = sigma2*np.exp(-np.linalg.norm(d)/L)
                Cm[j,i] = Cm[i,j]
                j += 1
            i += 1
    Cm[Cm<epsFloat] = 0.
    return Cm

def ionosphereModel(x,dayTime=True,bump=False):
    h = x[2]
    Nf1 = 4*np.exp((h-300)/100.)/(1 + np.exp((h-300)/100.))**2
    res = Nf1
    if dayTime:#also E layer
        Ne = 0.3*4*np.exp((h-85.)/50.)/(1 + np.exp((h-85.)/50.))**2
        res += Ne
    if bump:
        res += 0.5*np.exp(-np.sum((x - np.array([30,30,500]))**2)/50.**2)
        res += 0.2*np.exp(-np.sum((x - np.array([-30,-30,200]))**2)/50.**2)
    return res

def constructIonosphereModel(maxBaseline,height):
    '''initialize with 1/m^3 at 300km +- 150km'''

    fileName = "ionosphereModel.npy"
    octTree = OctTree([0,0,height/2.],dx=maxBaseline,dy=maxBaseline,dz=height)
    #level 3 - all
    subDivide(subDivide(octTree))
    #subDivide(octTree)
    subDivide(octTree)
    voxels = getAllDecendants(octTree)
    voxels = []
    for vox in voxels:
        #level 4 - 250 to 750
        if (vox.centroid[2] > 250) and (vox.centroid[2] < 750):
            subDivide(vox)
        #level 5 - 250 to 500
        if (vox.centroid[2] > 250) and (vox.centroid[2] < 500):
            subDivide(vox)
    G,Cm,m,x = generateModelFromOctree(octTree,0)
    i = 0
    while i < x.shape[0]:
        m[i] = ionosphereModel(x[i,:],dayTime=True,bump=True)
        i += 1
    setOctTreeElectronDensity(octTree,m,np.ones_like(m)*0.05**2)
    saveOctTree(fileName,octTree)
    #plotOctTreeXZ(octTree,ax=None)
    #plotOctTree3D(octTree,model=m)
    return octTree

def gradientCheck(mprior,G):
    eps = 7./4. - 3./4. - 1.
    eps = epsFloat
    N = np.size(mprior)
    M = G.shape[0]
    K = np.mean(mprior)
    mlog = np.log(mprior/K)
    mForward = K*np.exp(mlog)
    g0 = G.dot(mForward)
    J = G*mForward
    Jexact = np.zeros([M,N])
    i = 0
    while i < N:
        mlog_old = mlog[i]
        mlog[i] += eps
        mForward = K*np.exp(mlog)
        g = G.dot(mForward)
        Jexact[:,i] = (g - g0)/eps
        #print(Jexact[:,i])
        mlog[i] = mlog_old
        i += 1
    import pylab as plt
    plt.imshow(J-Jexact)
    plt.colorbar()
    plt.show()
    return J,Jexact

def initHomogeneousModel(G,dobs):
    return np.sum(dobs)/np.sum(G)

def transformCov2Log(Cm_linear,K):
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
    Update using newer tecnique 
    '''
    #K = np.mean(K)
    #Cm_log = np.log(1 + Cm_linear/np.outer(mean_linear,mean_linear))
    Cm_log = np.log(1 + Cm_linear/K**2)
    return Cm_log

def transformCov2Linear(Cm_log,K):
    '''Invert the log transform
    '''
    return (np.exp(Cm_log) - 1.)*K**2

def mayaviPlot(x,m,maxNumPts=None):
    '''Do a density plot'''
    from mayavi import mlab
    
    X = x[:,0]
    Y = x[:,1]
    Z = x[:,2]
    xyz = np.vstack([X,Y,Z])
    #kde = stats.gaussian_kde(xyz)
    #density = kde(xyz)

    # Plot scatter with mayavi
    figure = mlab.figure('DensityPlot')
    figure.scene.disable_render = True
    pts = mlab.points3d(X,Y,Z, m, scale_mode='vector', scale_factor=1.)
    
    #if maxNumPts is not None:
    #    mask = pts.glyph.mask_points
    #    mask.maximum_number_of_points = x.size
    #    mask.on_ratio = 1
    #    pts.glyph.mask_input_points = True

    #figure.scene.disable_render = False 
    
    mlab.axes()
    mlab.show()
    #turn X,Y,Z,m into mesh and values
    
    #grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
    #mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))

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
        print(Cd,weightedRes)
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



def BerrymanSol(G,dobs,mprior=None,mu = 0.0,Cd=None,Cm = None):
    '''Solve d=G.m minimizing misfit:
    (dobs-d)^t.W1.(dobs-d) + mu (m - mprior)^t.W2.(m-mprior)
    with the berryman choice of W1, W2.
    G is mxn, m - num rays, n - num cells'''
    m = G.shape[0]
    n = G.shape[1]

    
    if Cd is not None:
        L = Cd +  np.diag(np.sum(G,axis=1))
    else:
        #row sums, length of path i
        L = np.diag(np.sum(G,axis=1))
    if Cm is not None:
        C = np.linalg.pinv(Cm + np.diag(np.sum(G,axis=0)))
    else:
        #col sums, legnth of all rays in cell j (coverage)
        C = np.diag(np.sum(G,axis=0))  
    Linv = np.linalg.pinv(L)  
    Cinv = np.linalg.pinv(C)
    #m-vec choice
    u = np.ones(m)
    #n-vec
    v = Cinv.dot(G.transpose()).dot(u)
    #v = np.ones(n)
    sigma0 = u.transpose().dot(dobs)/(u.transpose().dot(L).dot(u))
    if mprior is None:
        #weight for mean background m0 = (u^t.L.W3.dobs/u^t.L.W3.L.u)v
        #W3 = inv(L)
        #W3 = Linv
        #mprior = u.transpose().dot(L).dot(W3).dot(dobs)/(u.transpose().dot(L).dot(W3).dot(L).dot(u))*v
        mprior = sigma0*v
    #W1 = Linv
    #D = np.sqrt(C)
    #A = np.sqrt(W1).dot(G).dot(inv(D))
    Linv12 = sqrtm(Linv)
    Cinv12 = sqrtm(Cinv)
    A = Linv12.dot(G).dot(Cinv12)
    AtA = A.transpose().dot(A)
    print("eigen val solve At.A",AtA)
    #sorted in ascending order
    sing,eigv = np.linalg.eigh(AtA)
    #Zj = xj^t.A^t.Linv12.dobs
    zb = sqrtm(C).dot(mprior)
    dz = np.zeros(n)
    
    adjoint = A.transpose().dot(Linv12).dot(dobs)
    i = len(sing) - 2
    while i >= 0:
        Zj = eigv[:,i].transpose().dot(adjoint)
        #print(Zj)
        
        if np.isnan(sing[i]) or sing[i] < 1e-5:
            print("rank: {0}".format(len(sing)-1-i))
            break
        
        dz += Zj*eigv[:,i]/(sing[i]+mu)
        i -= 1
    #compare with 
    #zcomp = np.linalg.pinv(AtA).dot(Cinv12).dot(G.transpose()).dot(Linv).dot(dobs)
    z = dz + zb
    m = Cinv12.dot(z)
    return np.abs(m)
    
def MetropolisSolution(G,dobs,Cd,Cmprior,mprior):
    postDist = []
    iter = 0
    T0 = 1.
    size = 1000
    Cdinv_ = np.linalg.pinv(Cd)
    mML = mprior
    Cm = Cmprior
    while iter < 100:
        print("Metropolis iter-{0}".format(iter))
        T = T0/(1 + iter)
        Cdinv = Cdinv_/T
        Cmsample = Cmprior*T
        count = 0
        mCandidate = np.copy(mML)
        d = (G.dot(mCandidate) - dobs)
        Li = np.exp(-d.transpose().dot(Cdinv).dot(d)/2.)
        while count < 100:
            print("New sample batch: {0}".format(count))
            #sample prior
            msample = np.abs(np.random.multivariate_normal(mean=mML, cov = Cmsample))
            # posterior distribution
            #forward problems
            i = 0
            while i < len(mML):
                mCandidate[i] = msample[i]
                d = (G.dot(mCandidate) - dobs)
                Lj = np.exp(-d.transpose().dot(Cdinv).dot(d)/2.)
                if Lj > Li:
                    Li = Lj
                    count += 1
                else:
                    if np.random.uniform() < Lj/Li:
                        Li = Lj
                        count += 1
                    else:
                        mCandidate[i] = mML[i]
                i += 1
        postDist.append(mCandidate)
        mML = mCandidate#np.mean(postDist,axis=0)
        iter += 1
    Cm = np.cov(postDist,rowvar=0)
    return mML,Cm
            
def LMSol(G,mprior,Cd,Cm,dobs,mu=1.,octTree=None):
    """Assume the frechet derivative is,
    G(x) = exp"""

    K = np.mean(mprior)
    mlog = np.log(mprior/K)
    
    Cm_log = transformCov2Log(Cm,K)#np.log(1. + Cm/K**2)#transformCov2Log(Cm,mprior)
    #Cdinv = np.linalg.pinv(Cd)
    C = np.sum(G>0,axis=0)
    C = C/np.max(C)
    C[C==0] = np.min(C[C>0])/2.
    #C = np.sum(G,axis=0)
    #C = C/np.max(C)
    res = 1
    iter = 0
    while res > 1e-8 and iter < 1000:
        #forward transform
        mForward = K*np.exp(mlog)
        g = G.dot(mForward)
        J = G*mForward
        #residuals g - dobs -> -dm
        res = g - dobs
        
        #A1 = J.transpose().dot(Cdinv)
        #Cmlog_inv = A1.dot(J) + mu*Cm_log
        #dm,resi,rank,s = np.linalg.lstsq(Cmlog_inv,A1.dot(res))
        #S = mu Cd + J.Cm.J^t
        #S = int Ri Rj k^2 exp(m(x) + m(x')) sigma^2 exp(-|x-x'|/L) + Cd
        
        #K int dV Cm(x,x') J(x') del(i)
        P1 = Cm_log.dot(J.transpose())
        smooth = np.linalg.pinv(mu*Cd + J.dot(P1))
        dm = P1.dot(smooth).dot(res)
        res = np.sum(dm**2)/np.sum(mlog**2)
        print("Iter-{0} res: {1}".format(iter,res))
        #converage learn propto length of rays in cells
        
        mlog -= dm*C
        iter += 1
    
    return K*np.exp(mlog), transformCov2Linear(Cm_log,K)
        
if __name__=='__main__':
    print("Constructing ionosphere model")
    maxBaseline = 100.
    height=1000.
    octTree = constructIonosphereModel(maxBaseline,height)
    rays = makeRaysFromSourceAndReciever(maxBaseline = maxBaseline,height=height)
    print("Propagating {0} rays".format(len(rays)))
    for ray in rays:
        forwardRay(ray,octTree)
    print("Pulling ray propagations.")
    G,mVar,mexact,x = generateModelFromOctree(octTree,len(rays))
    print("Computing model 3d exponential covariance")
    Cmprior = compute3dExponentialCovariance(np.sqrt(np.mean(mVar)),30.,x)
    #generate simple initial starting point
    print("Setting a priori model")
    mprior = []
    i = 0
    while i < x.shape[0]:
        mprior.append(ionosphereModel(x[i,:],dayTime=False,bump=False))
        i += 1
    mprior = np.array(mprior)
    #mprior = np.ones_like(mexact)*initHomogeneousModel(G,dobs)
    #mprior = np.random.multivariate_normal(mean=mexact, cov = Cmprior)
    print("Computing observation covariance")
    dobs = []
    for i in range(10):
        dobs.append(G.dot(np.abs(np.random.multivariate_normal(mean=mexact, cov = Cmprior))))
    dobs = np.array(dobs)
    Cd = np.cov(dobs.transpose())
    dobs = G.dot(mexact)
    print("Solving for model from rays:")
    #m,Cm = LinearSolution(dobs,G,Cd,Cmprior,mprior)
    #m,Cm = MetropolisSolution(G,dobs,Cd,Cmprior,mprior)
    #m = BerrymanSol(G,dobs,mprior=None,Cd=Cd,Cm=None,mu=0.00)
    #m,Cm = SteepestDescent(octTree,rays,dobs,Cd,Cmprior,mprior)
    m,Cm = LMSol(G,mprior,Cd,Cmprior,dobs,mu=1.0)
    CmCm = Cm.dot(np.linalg.inv(Cmprior))
    R = np.eye(CmCm.shape[0]) - CmCm
    print("Resolved by dataSet:{0}, resolved by a priori:{1}".format(np.trace(R),np.trace(CmCm)))
    import pylab as plt
    plt.plot(m,label='res')
    plt.plot(mexact,label='ex')
    plt.plot(mprior,label='pri')
    C = np.sum(G>0,axis=0)
    C = C < 3
    plt.scatter(np.arange(len(m))[C],m[C])
    plt.legend(frameon=False)
    plt.show()
    plotOctTreeXZ(octTree,ax=None)
    plotOctTree3D(octTree,model=m,rays=False)
    if False:
        f,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True)
        ax1.imshow(Cm)
        ax2.imshow(Cmexact_lin)
        plt.show()

        plt.hist(np.sum(G>0,axis=0),bins=100)
        plt.show()
        print(G)


