
# coding: utf-8

# In[1]:

from Geometry import *
from itertools import combinations


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

def buildVoxels(centers,dx,dy,dz):
    voxels = []
    for center in centers:
        voxel.append(Voxel(center=center,dx=dx,dy=dy,dz=dz))

def point2index(point,du,dw):
    nu = np.floor(point[0]/du)
    nv = np.floor(point[1]/du)
    nw = np.floor(point[2]/dw)
    return nu,nv,nw    
    
def tracer(sources,recievers,planes):
    segs = []
    for s,r in zip(sources,recievers):
        seg = LineSegment(r,s)
        segs.append(seg)
    #plot x vs y
    plotProj(segs,planes,np.array([0,1,0]))
    #build voxels
    voxels = buildVoxels(planes)
    #build model
    for seg in segs:
        rayLengths = []
        while seg.sep > 0:
            #ind = point2index(seg.origin)
            dist = []
            points = []
            i = 0
            while i < len(planes):
                res,point = intersectRayPlane(seg,planes[i])
                if res:
                    points.append(point)
                    dist.append(np.linalg.norm(point-seg.origin))
                i += 1
            closest = np.argmin(dist)
            rayLength.append(dist[closest])
            
            seg = LineSegment(points[closest],seg.eval(seg.sep))
            
def plotProj(lineSegs,planes,projDir):
    import pylab as plt
    projPlane = Plane([0,0,0],normal=projDir)
    xdir,ydir,zdir = gramSchmidt(projDir)
    R = np.array([xdir,ydir,zdir])
    f = plt.figure()
    ax = plt.subplot(111)
    count = 0
    for seg in lineSegs:
        count += 1
        x1 = R.dot(seg.origin)
        x2 = R.dot(seg.eval(seg.sep))
        ax.plot([x1[0],x2[0]],[x1[1],x2[1]],label='seg-{0}'.format(count))
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
    width = 15.
    dw = height/10.
    du = width/10.
    s = [np.array([5,0,0]),np.array([2,1,0]),np.array([0,1,0])]
    r = [np.array([0,1,height]),np.array([1,6,height]),np.array([2,-1,height])]
    planes = makePlanes(height,width,du,dw)
    buildVoxels(planes)
    tracer(s,r,planes)
    
    
    planexy1 = Plane(np.array([0,0,0]),normal=np.array([0,0,1]))
    planexz1 = Plane(np.array([0,0,0]),normal=np.array([0,1,0]))
    planeyz1 = Plane(np.array([0,0,0]),normal=np.array([1,0,0]))
    planexy2 = Plane(np.array([1,1,1]),normal=np.array([0,0,1]))
    planexz2 = Plane(np.array([1,1,1]),normal=np.array([0,1,0]))
    planeyz2 = Plane(np.array([1,1,1]),normal=np.array([1,0,0]))
    print Voxel(planexy1,planexz1,planeyz1,planexy2,planexz2,planeyz2)


# In[ ]:




# In[ ]:



