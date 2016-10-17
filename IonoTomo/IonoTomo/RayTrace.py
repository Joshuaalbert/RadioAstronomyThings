
# coding: utf-8

# In[7]:

from Geometry import *
from itertools import combinations

def makeOctTree(eastWidth,northWidth,pointingHeight,alt,az,minSep):
    '''Define an OctTree with pointing and given dimensions
    minSep - minimum seperation between sources in radians'''
    minCellSize = np.tan(minSep)*pointingHeight
    center = np.array([0,0,pointingHeight/2.])
    masterOctTree = OctTree(center,eastWidth,northWidth,pointingHeight)
    while masterOctTree.minCellSize()[0] > minCellSize:
        masterOctTree.subDivide()
    
    

def makePlanes(height,width,du,dw):
    planes = []
    h = 0
    while h < height:
        planes.append(Plane(np.array([0,0,h]),normal=np.array([0,0,1])))
        h += dw

    h = 0
    while h < width:
        planes.append(Plane(np.array([0,h-width/2.,0]),normal=np.array([0,1,0])))
        h += du

    h = 0
    while h < width:
        planes.append(Plane(np.array([h-width/2.,0,0]),normal=np.array([1,0,0])))
        h += du
    return planes

def buildOctTree(center,dx,dy,dz):
    voxels = []
    for center in centers:
        voxel.append(Voxel(center=center,dx=dx,dy=dy,dz=dz))
    

def point2index(point,du,dw):
    nu = np.floor(point[0]/du)
    nv = np.floor(point[1]/du)
    nw = np.floor(point[2]/dw)
    return nu,nv,nw    
    
def getClosestApproach(segs):
    '''Get the closest approach between all segs'''
    closest = {}
    outSegs = []
    i = 0
    while i < len(segs):
        seg1 = segs[i]
        sep = np.inf 
        j = 0
        while j < len(segs):
            seg2 = segs[j]
            if i == j:
                j += 1
                continue
            if np.alltrue(seg1.origin == seg2.origin):
                j += 1
                continue
            res,seg = intersectRayRay(seg1,seg2)
            if res:
                sep = 0
                closest[seg1] = LineSegment(seg,seg)#seg is a point
                outSegs.append(closest[seg1])
                break
            if seg.sep < sep:
                sep = seg.sep
                closest[seg1] = seg
                outSegs.append(closest[seg1])
            j += 1
        i += 1
            
    return closest,outSegs
            
                
def tracer(sources,recievers):
    segs = []
    for s in sources:
        for r in recievers:
            seg = LineSegment(r,s)
            segs.append(seg)
    #get closest approach
    connectDict,connectingSegs = getClosestApproach(segs)
    print connectingSegs
    #build OctTree
    mean = (np.mean(sources,axis=0) + np.mean(recievers,axis=0))/2
    lims = np.max([np.max(sources,axis=0),np.max(recievers,axis=0)],axis=0)-np.min([np.min(sources,axis=0),np.min(recievers,axis=0)],axis=0)
    print mean,lims
    octTree =  OctTree(center=mean,dx=lims[0],dy = lims[1], dz = lims[2])
    octTree.subDivide().subDivide().subDivide()
    print("minimum cellsize: ",octTree.minCellSize())
    #plotProj(segs,octTree.getAllBoundingPlanes()[0],np.array([0,1,0]),connectingSegs=connectingSegs)
    #build model
    for seg in segs:
        rayLengths = []
        #intersect with segment a ray!
        #populate the voxels with the length of each ray in them
        res,point,neighbours = octTree.intersectRay(seg)
        if res:#hit a voxel
            if point[2] >= seg.eval(seg.sep)
            
def plotProj(lineSegs,planes,projDir,connectingSegs=None):
    '''Project the lineSegs and planes into projDir and show'''
    import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    projPlane = Plane([0,0,0],normal=projDir)
    xdir,ydir,zdir = gramSchmidt(projDir)
    R = np.array([xdir,ydir,zdir])
    f = plt.figure()
    ax = plt.subplot(111,projection='3d')
    count = 0
    for seg in lineSegs:
        count += 1
        x1 = R.dot(seg.origin)
        x2 = R.dot(seg.eval(seg.sep))
        ax.plot([x1[0],x2[0]],[x1[1],x2[1]],[x1[2],x2[2]],label='seg-{0}'.format(count))
    for seg in connectingSegs:
        x1 = R.dot(seg.origin)
        x2 = R.dot(seg.eval(seg.sep))
        ax.plot([x1[0],x2[0]],[x1[1],x2[1]],[x1[2],x2[2]],lw=2)
    ax.set_xlabel('Dir {0}'.format(xdir))
    ax.set_ylabel('Dir {0}'.format(ydir))
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    for p in planes:
        res,ray  = intersectPlanePlane(p,projPlane)
        if res:
            tmax = np.sqrt((xmax-xmin)**2 + (ymax - ymin)**2)
            ray = Ray(R.dot(ray.origin),R.dot(ray.dir))
            ax.plot([ray.eval(-tmax)[0],ray.eval(tmax)[0]],[ray.eval(-tmax)[1],ray.eval(tmax)[1]],c='black')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    plt.legend(frameon=False)
    plt.show()
        

if __name__=='__main__':
    height = 100.

    r = [np.array([0,0,0]),np.array([1.2,0,0]),np.array([2.1,0,0])]
    s = [np.array([0,1,height]),np.array([1,0,height]),np.array([2.3,0,height])]

    tracer(s,r)


# In[ ]:




# In[ ]:



