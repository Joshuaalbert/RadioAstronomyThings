
# coding: utf-8

# In[1]:

from Geometry import *
from itertools import combinations
import pylab as plt

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

def plotProj(lineSegs,planes,projDir,connectingSegs=None):
    '''Project the lineSegs and planes into projDir and show'''
    import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    projPlane = Plane([0,0,0],normal=projDir)
    xdir,ydir,zdir = gramSchmidt(projDir)
    R = np.array([xdir,ydir,zdir])
    f = plt.figure()
    ax = plt.subplot(111)#,projection='3d')
    count = 0
    for seg in lineSegs:
        count += 1
        x1 = R.dot(seg.origin)
        x2 = R.dot(seg.eval(seg.sep))
        ax.plot([x1[0],x2[0]],[x1[1],x2[1]],label='seg-{0}'.format(count))
    if connectingSegs is not None:
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
    return ax
    #plt.show()


def propagateRay(ray,octTree,ax=None):
    '''Propagate ray through octTree until it leaves the boundaries'''
    #from outside to octTree
    if ax is None:
        ax = plt.subplot(111)
    print(ray,octTree)
    inside,entryPoint,entryPlaneIdx = octTree.intersectRay(ray)
    if not inside:
        print('failed to hit',octTree)
        return
    vox = octTree.intersectPoint(entryPoint)
    n1 = 1
    entryRay = ray
    while inside:
        n2 = vox.properties['n'][1]
        normal = -vox.boundingPlanes[entryPlaneIdx].n
        rayProp = snellsLaw(n1,n2,entryRay,normal,entryPoint)
        res,exitPoint,exitPlaneIdx = vox.propagateRay(rayProp)
        ax.plot([exitPoint[0],entryPoint[0]],[exitPoint[2],entryPoint[2]])
        if not res:
            print("something went wrong and ",rayProp," didn't exit ",vox)
            return
        vox.lineSegments[rayProp.id] = np.linalg.norm(exitPoint-entryPoint)
        print(vox.lineSegments)
        #resolve the next vox to hit
        this = vox
        unresolved = True
        while unresolved:
            if this.parent is None:
                inside = False
                break
                
            nextVoxIdx = this.getOtherSide(exitPlaneIdx)
            if nextVoxIdx >= 0:#hits a sibling of this
                nextVox = this.parent.children[nextVoxIdx]
                res,entryPoint,entryPlaneIdx = nextVox.intersectRay(rayProp)
                if not res:
                    print("failed to hit sibling",nextVox)
                    plt.show()
                    return
                vox = nextVox.intersectPoint(entryPoint)
                n1 = n2
                entryRay = rayProp
                unresolved = False
                continue
            else:#nextVoxIdx is -planeIdx-1 of parent
                this = this.parent
                exitPlaneIdx = -(nextVoxIdx+1)
    return exitPoint,vox.boundingPlanes[exitPlaneIdx]
                
def tracer(sources,recievers):
    '''Create an array of ray lengths'''
    
    segs = []
    count = 1
    ids = []
    for s in sources:
        for r in recievers:
            seg = LineSegment(r,s,id=count)
            segs.append(seg)
            ids.append(count)
            count += 1
    #get closest approach
    #connectDict,connectingSegs = getClosestApproach(segs)
    #print connectingSegs
    #build OctTree
    mean = (np.mean(sources,axis=0) + np.mean(recievers,axis=0))/2
    lims = np.max([np.max(sources,axis=0),np.max(recievers,axis=0)],axis=0)-np.min([np.min(sources,axis=0),np.min(recievers,axis=0)],axis=0)
    #print mean,lims
    octTree =  OctTree(center=mean,dx=lims[0],dy = lims[1], dz = lims[2])
    octTree.subDivide().subDivide().subDivide()
    print("minimum cellsize: ",octTree.minCellSize())
    ax = plotProj(segs,octTree.getAllBoundingPlanes()[0],np.array([0,1,0]))
    for ray in segs:
        propagateRay(ray,octTree,ax=ax)
    plt.show()
    #build model
    #get all children
    #allChildren = octTree.getAllDecendants()
    
    #G,m = octTree.generateModel(ids)
    #print (np.sum(G),m)
            



if __name__=='__main__':
    height = 100.

    r = [np.array([0,0,0])]
    s = [np.array([0.1,1,height]),np.array([1,0,height]),np.array([2.3,0,height])]

    tracer(s,r)


# In[2]:

plt.show()


# In[ ]:



