
# coding: utf-8

# In[1]:

'''Contains geometry algorithms'''
import numpy as np
from itertools import combinations

epsFloat = (7/3. - 4/3. - 1)*10

def rot(dir,theta):
    '''create the rotation matric about unit vector dir by theta radians.
    Appear anticlockwise when dir points toward observer.
    and clockwise when pointed away from observer. 
    **careful when visualizing with hands**'''
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,-dir[2]*s,dir[1]*s],
                     [dir[2]*s,c,-dir[0]*s],
                     [-dir[1]*s,dir[0]*s,c]]) + (1-c)*np.outer(dir,dir)

def rotx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1,0,0],
                    [0,c,-s],
                    [0,s,c]])
def roty(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,0,s],
                    [0,1,0],
                    [-s,0,c]])

def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,-s,0],
                    [s,c,0],
                    [0,0,1]])

def rotAxis(R):
    '''find axis of rotation for R using that (R-I).u=(R-R^T).u=0'''
    u = np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    #|u| = 2 sin theta
    return u/np.linalg.norm(u)
    
def rotAngle(R):
    '''Given a rotation matix R find the angle of rotation.
    Use that Tr(R) = 1 + 2 cos(theta)'''
    return np.arccos((R[0,0] + R[1,1] + R[2,2] - 1.)/2.)

def localENU2uvw(enu,alt,az,lat):
    '''Given a local ENU vector, rotate to uvw coordinates'''
    #first get hour angle and dec from alt az and lat
    ha,dec = altAz2hourangledec(alt,az,lat)
    return rotx(lat).dot(rotz(-ha)).dot(rotx(lat-dec)).dot(enu)

def itrs2uvw(ha,dec,lon,lat):
    #ha,dec = altAz2hourangledec(alt,az,lat)
    R = rotx(-lat).dot(rotz(-ha)).dot(rotx(np.pi/2.-dec)).dot(rotz(np.pi/2. + lon))
    udir = R.dot(np.array([1,0,0]))
    vdir = R.dot(np.array([0,1,0]))
    wdir = R.dot(np.array([0,0,1]))
    return np.array([udir,vdir,wdir])
    udir = rotz(lon-ha).dot(roty(- dec - lat)).dot(np.array([0,1,0]))
    vdir = rotz(lon-ha).dot(roty(- dec - lat)).dot(np.array([0,0,1]))
    wdir = rotz(lon-ha).dot(roty(- dec - lat)).dot(np.array([1,0,0]))
    #print np.linalg.norm(udir),np.linalg.norm(vdir),np.linalg.norm(wdir)
    return np.array([udir,vdir,wdir])

class Ray(object):
    def __init__(self,origin,direction):
        self.origin = np.array(origin)
        self.dir = np.array(direction)/np.sqrt(np.dot(direction,direction))
    def eval(self,t):
        '''Return the location along the line'''
        return self.origin + t*self.dir
    def __repr__(self):
        return "Ray: origin: {0} -> dir: {1}".format(self.origin,self.dir)
    
class LineSegment(Ray):
    def __init__(self,p1,p2):
        self.sep = np.linalg.norm(np.array(p2)-np.array(p1))
        super(LineSegment,self).__init__(p1,p2 - p1)
        if np.alltrue(p1==p2):
            print("Point not lineSegment")        
    def inBounds(self,t):
        return (t>0) and (t <= self.sep)
    def __repr__(self):
        return "LineSegment: p1: {0} -> p2 {1}, sep: {2}".format(self.eval(0),self.eval(self.sep),self.sep)

class Plane(object):
    def __init__(self,p1,p2=None,p3=None,p4=None,normal=None):
        '''If normal is defined and surface defines a part of a convex polyhedron,
        take normal to point inside the poly hedron.
        ax+by+cz+d=0'''
        if normal is None:
            if p2 is None and p3 is None:
                print("Not enough information")
                return
            a = p1[1]*(p2[2] - p3[2]) + p2[1]*(p3[2] - p1[2]) + p3[1]*(p1[2] - p2[2])
            b = p1[2]*(p2[0] - p3[0]) + p2[2]*(p3[0] - p1[0]) + p3[2]*(p1[0] - p2[0])
            c = p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])
            d = -p1[0]*(p2[1]*p3[2] - p3[1]*p2[2]) - p2[0]*(p3[1]*p1[2] - p1[1]*p3[2]) - p3[0]*(p1[1]*p2[2] - p2[1]*p1[2])
            self.n = np.array([a,b,c])
            self.d = d/np.linalg.norm(self.n)
            self.n = self.n/np.linalg.norm(self.n)
        else:
            self.n = np.array(normal)
            self.n = self.n/np.linalg.norm(self.n)
            self.d = -self.n.dot(p1)
    def normalSide(self,p):
        s = self.n.dot(p) + self.d
        if s > epsFloat:
            return True
        if s < -epsFloat:
            return False
        return None
    def onPlane(self,p):
        return (np.abs(self.n.dot(p) + self.d) < epsFloat)
    def __repr__(self):
        return "Plane: normal: {0}, d=-n.p {1}".format(self.n,self.d)
      

        
                    
def projLineSegPlane(line,plane):
    '''Project a lineseg onto a plane.'''
    proj = np.eye(np.size(line.origin)) - np.outer(plane.n,plane.n)
    x1 = proj.dot(line.origin)
    x2 = proj.dot(line.eval(line.sep))
    return LineSegment(x1,x2)

def projRayPlane(line,plane):
    '''Project a ray onto a plane.'''
    proj = np.eye(np.size(line.origin)) - np.outer(plane.n,plane.n)
    x1 = proj.dot(line.origin)
    dir = proj.dot(line.dir)
    return Ray(x1,dir)    

def gramSchmidt(dir):
    '''Get an ortho system of axes'''
    raxis = np.cross(dir,np.array([0,0,1]))
    mag = np.linalg.norm(raxis) 
    if mag == 0:
        zdir = np.array([0,0,1])
        xdir = np.array([1,0,0])
        ydir = np.array([0,1,0])
    else:
        R = rot(raxis,np.arcsin(mag/np.linalg.norm(dir)))
        zdir = dir
        xdir = (np.eye(3) - np.outer(zdir,zdir)).dot(R.dot(np.array([1,0,0])))
        xdir /= np.linalg.norm(xdir)
        ydir = (np.eye(3) - np.outer(zdir,zdir)- np.outer(xdir,xdir)).dot(R.dot(np.array([0,1,0])))
        ydir /= np.linalg.norm(ydir)
    return xdir,ydir,zdir    

def intersectPointRay(point,ray):
    diff = point - ray.origin
    t = diff.dot(ray.dir)
    p = ray.eval(t)
    if diff.dot(diff) - t**2 > 0:
        return False, LineSegment(point,p)
    else:
        return True, p
    
def intersectRayRay(ray1,ray2):
    '''returns type,res
    type = -1 parallel rays
    0 -> intersection point
    1 -> LineSegment'''
    A = np.cross(ray1.dir,ray2.dir)
    Amag = np.sqrt(A.dot(A))
    if Amag < epsFloat:
        print("ray1 and ray2 are parallel")
        return -1,None
    B = ray2.origin - ray1.origin
    M = np.array([B,ray2.dir,A]).transpose()
    t1 = np.linalg.det(M)/Amag
    M[:,1] = ray1.dir
    t2 = np.linalg.det(M)/Amag
    p1 = ray1.eval(t1)
    p2 = ray2.eval(t2)
    c = p2 - p1
    d = np.sqrt(c.dot(c))
    if (d < epsFloat):
        return True, p1
    else:
        return False,LineSegment(p1,p2)
    
def intersectRayPlane(ray,plane):
    '''Intersection of ray and plane'''
    proj = plane.n.dot(ray.dir)
    if (proj < epsFloat):
        return False, None
    else:
        t = (-plane.d - plane.n.dot(ray.origin))/proj
        return True,ray.eval(t) 

def intersectLineSegmentPlane(lineSeg,plane):
    '''Intersection of segment and plane'''
    proj = plane.n.dot(lineSeg.dir)
    if (proj < epsFloat):
        return False, None
    else:
        t = (-plane.d - plane.n.dot(lineSeg.origin))/proj
        return lineSeg.inBounds(t),lineSeg.eval(t) 
    
def intersectPlanePlane(plane1,plane2):
    '''calculate intersection of 2 planes'''

    n1n2 = plane1.n.dot(plane2.n)#3 + 3
    if n1n2 > 1 - epsFloat:
        print "Parallel planes"
        return False, None
    n1n1 = plane1.n.dot(plane1.n)#3 + 3
    n2n2 = plane2.n.dot(plane2.n)#3 + 3
    det = n1n1*n2n2 - n1n2*n1n2#2 + 1
    c1 = (plane2.d*n1n2 - plane1.d*n2n2)/det#3 + 1
    c2 = (plane1.d*n1n2 - plane2.d*n1n1)/det#3 + 1
    u = np.cross(plane1.n,plane2.n)#6 + 3
    ray = Ray(c1*plane1.n + c2*plane2.n,u)#6 + 3
    return True,ray#35 + 18
    
    u = np.cross(plane1.n,plane2.n)#6 + 3
    uMag = np.linalg.norm(u)#
    if uMag < epsFloat:
        print "Parallel planes"
        return False,None
    u /= uMag
    i = np.argmax(u)
    den = u[i]
    if i == 0:
        ray = Ray(np.array([0,(plane1.n[2]*plane2.d - plane2.n[2]*plane1.d)/den,
                            (plane2.n[1]*plane1.d - plane1.n[1]*plane2.d )/den]),u)#6 + 2
        return True,ray#12 + 5
    if i == 1:
        ray = Ray(np.array([(plane1.n[2]*plane2.d - plane2.n[2]*plane1.d)/den,0,
                            (plane2.n[0]*plane1.d - plane1.n[0]*plane2.d )/den]),u)
        return True,ray
    if i == 2:
        ray = Ray(np.array([(plane1.n[1]*plane2.d - plane2.n[1]*plane1.d)/den,
                            (plane2.n[0]*plane1.d - plane1.n[0]*plane2.d )/den,0]),u)
        return True,ray

def intersectPlanePlanePlane(plane1,plane2,plane3):
    n23 = np.cross(plane2.n,plane3.n)
    n31 = np.cross(plane3.n,plane1.n)
    n12 = np.cross(plane1.n,plane2.n)
    n123 = plane1.n.dot(n23)
    if (n123 == 0):
        return False, None
    else:
        return True, -(plane1.d*n23 + plane2.d*n31 + plane3.d*n12)/n123
    
def centroidOfPlanes(planes):
    '''Will often fail if not properly oriented planes'''
    rays = []
    i = 0
    while i < len(planes):
        j = i+1
        while j<len(planes):
            res,ray = intersectPlanePlane(planes[i],planes[j])
            if res:
                rays.append(ray)
            j += 1
        i += 1
    centroid = np.array([0,0,0])
    count = 0
    i = 0 
    while i<len(rays):
        j = i+1
        while j < len(rays):
            res,point = intersectRayRay(rays[i],rays[j])
            if res:
                centroid += point
                count += 1
            j += 1
        i += 1

    return (count >= 5), centroid/count

def coplanarPoints(points):
    if len(points) < 3:
        print "Not enough points to test"
        return False,None
    if len(points) == 3:
        return True, Plane(*points)
    plane = Plane(*points[:3])
    i = 3
    while i < len(points):
        if not plane.onPlane(points[i]):
            return False, None
        i += 1
    return True,plane

def cosLineLine(line1,line2):
    return line1.dir.dot(line2.dir)
    
def midPointLineSeg(seg):
    return seg.eval(seg.sep/2.)

class BoundedPlane(Plane):
    '''assumes convex hull of 4 points'''
    def __init__(self,vertices):
        if len(vertices) != 4:
            print("Not enough vertices")
            return
        res,plane = coplanarPoints(vertices)
        if res:
            super(BoundedPlane,self).__init__(*vertices)
            
            self.centroid = np.mean(vertices,axis=0)
            #first triangle
            edge01 = LineSegment(vertices[0],vertices[1])
            edge12 = LineSegment(vertices[1],vertices[2])
            edge20 = LineSegment(vertices[2],vertices[0])
            
            centroid1 = (vertices[0] + vertices[1] + vertices[2])/3.
            centerDist = np.linalg.norm(centroid1 - vertices[3])
            if np.linalg.norm(midPointLineSeg(edge01) - vertices[3]) < centerDist:
                #reject 01
                self.edges = [edge12,edge20,LineSegment(vertices[0],vertices[3]),
                         LineSegment(vertices[1],vertices[3])]
            if np.linalg.norm(midPointLineSeg(edge12) - vertices[3]) < centerDist:
                #reject 12
                self.edges = [edge01,edge20,LineSegment(vertices[1],vertices[3]),
                         LineSegment(vertices[2],vertices[3])]
            if np.linalg.norm(midPointLineSeg(edge20) - vertices[3]) < centerDist:
                #reject 21
                self.edges = [edge01,edge12,LineSegment(vertices[2],vertices[3]),
                         LineSegment(vertices[0],vertices[3])]   
        else:
            print('not coplanar',vertices)
                
    def inHull(self,point):
        '''Determine if point in hull of plane'''
        res = self.onPlane(point)
        if res:
            for e in self.edges:
                t = point - e.origin
                s = self.centroid - e.origin
                if -t.dot(s) + t.dot(e.dir)*s.dot(e.dir) < -epsFloat:
                    return False
            return True

def planesOfCuboid(center,dx,dy,dz):
    '''return bounding planes of cuboid'''
    planes = []
    planes.append(Plane(np.array(center) - np.array([dx/2.,0,0]),normal=np.array([1,0,0])))
    planes.append(Plane(np.array(center) - np.array([-dx/2.,0,0]),normal=np.array([-1,0,0])))
    planes.append(Plane(np.array(center) - np.array([0,dy/2.,0]),normal=np.array([0,1,0])))
    planes.append(Plane(np.array(center) - np.array([0,-dy/2.0,0]),normal=np.array([0,-1,0])))
    planes.append(Plane(np.array(center) - np.array([0,0,dz/2.]),normal=np.array([0,0,1])))
    planes.append(Plane(np.array(center) - np.array([0,0,-dz/2.]),normal=np.array([0,0,-1])))
    return planes

def boundPlanesOfCuboid(center,dx,dy,dz):
    '''return bounding planes of cuboid'''
    planes = []
    
    planes.append(BoundedPlane([np.array(center) + np.array([-dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,-dy/2.,dz/2.])]))
    planes.append(BoundedPlane([np.array(center) + np.array([dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,-dz/2.])]))
    planes.append(BoundedPlane([np.array(center) + np.array([-dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,-dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,-dz/2.])]))
    planes.append(BoundedPlane([np.array(center) + np.array([-dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,dz/2.])]))
    planes.append(BoundedPlane([np.array(center) + np.array([dx/2.,dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,-dz/2.])]))
    planes.append(BoundedPlane([np.array(center) + np.array([-dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([-dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,-dz/2.])]))
    return planes

class Voxel(object):
    def __init__(self,center=None,dx=None,dy=None,dz=None,boundingPlanes=None):
        '''Create a volume out of bounding planes'''
        try:
            boundingPlanes = boundPlanesOfCuboid(center,dx,dy,dz)
        except:
            pass
            
        if boundingPlanes is not None:
            self.vertices = []
            planeTriplets = combinations(boundingPlanes,3)
            for triplet in planeTriplets:
                
                res,point = intersectPlanePlanePlane(*triplet)
                if res:
                    self.vertices.append(point)
            if len(self.vertices)<8:
                print("Planes don't form voxel",len(self.vertices))
                self.volume = 0
                return
            self.centroid = np.mean(self.vertices,axis=0)
            sides = np.max(self.vertices,axis=0) - np.min(self.vertices,axis=0)
            self.dx = sides[0]
            self.dy = sides[1]
            self.dz = sides[2]
            self.volume = self.dx*self.dy*self.dz
        self.boundingPlanes = boundingPlanes
    def intersectRay(self,ray):
        points = []
        for plane in self.boundingPlanes:
            res,point = intersectRayPlane(ray,plane)
            if res:
                res = plane.inHull(point)
                if res:
                    points.append(point)
        return len(points)>0,points           
    def __repr__(self):
        return "Voxel: Center {0}\nVertices:\t{2}".format(self.centroid,self.vertices)
    
class OctTree(Voxel):
    def __init__(self,center=None,dx=None,dy=None,dz=None,boundingPlanes=None):
        super(OctTree,self).__init__(center,dx,dy,dz,boundingPlanes)
        self.children = []
    def subDivide(self):
        '''Make eight voxels to partition this voxel'''
        if len(self.children)==8:
            for child in self.children:
                child.subDivide()
        else:
            self.children = []
            dx = self.dx/2.
            dy = self.dy/2.
            dz = self.dz/2.
            self.children.append(OctTree(self.centroid - np.array([dx/2.,dy/2.,dz/2.]),dx,dy,dz))
            self.children.append(OctTree(self.centroid - np.array([dx/2.,-dy/2.,-dz/2.]),dx,dy,dz))
            self.children.append(OctTree(self.centroid - np.array([dx/2.,dy/2.,-dz/2.]),dx,dy,dz))
            self.children.append(OctTree(self.centroid - np.array([dx/2.,-dy/2.,dz/2.]),dx,dy,dz))
            self.children.append(OctTree(self.centroid - np.array([-dx/2.,dy/2.,dz/2.]),dx,dy,dz))
            self.children.append(OctTree(self.centroid - np.array([-dx/2.,-dy/2.,-dz/2.]),dx,dy,dz))
            self.children.append(OctTree(self.centroid - np.array([-dx/2.,dy/2.,-dz/2.]),dx,dy,dz))
            self.children.append(OctTree(self.centroid - np.array([-dx/2.,-dy/2.,dz/2.]),dx,dy,dz))
        return self
    def intersectRay(self,ray):
        points = []
        for plane in self.boundingPlanes:
            res,point = intersectRayPlane(ray,plane)
            if res:
                res = plane.inHull(point)
                if res:
                    points.append(point)
        return len(points)>0,points   
    def __repr__(self):
        return "OctTree: center {0} children {1}".format(self.centroid,len(self.children)==8)
        
if __name__ == '__main__':    
    

    octTree = OctTree(np.array([0,0,0]),1,1,1)
    print octTree
    octTree.subDivide().subDivide()
    print octTree.children[0].children
    exit(0)
    rays = []
    planes = []
    for i in range(6):
        rays.append(Ray(np.random.uniform(size=3),np.random.uniform(size=3)))
        planes.append(Plane(rays[-1].origin,normal=rays[-1].dir))
    ray1 = Ray(np.array([0,4,0]),np.array([1,0,0]))
    ray2 = Ray(np.array([1,0,0]),np.array([0,1,0]))
    ray3 = Ray(np.array([1,1,1]),np.array([0,0,1]))
    plane1 = Plane(ray1.origin,normal=ray1.dir)
    plane2 = Plane(ray2.origin,normal=ray2.dir)
    plane3 = Plane(ray3.origin,normal=ray3.dir)
    vox = Voxel(planesOfCuboid([0,0,0],1,1,1))
    #print vox.dz
    print intersectPlanePlanePlane(plane1,plane2,plane3)
    res,ray1 = intersectPlanePlane(plane1,plane2)
    res,ray2 = intersectPlanePlane(plane2,plane3)
    print intersectRayRay(ray1,ray2)
    print "floating point prec. times a few",epsFloat
    print intersectRayRay(ray1,ray2)
    print plane1
    print intersectRayPlane(ray2,plane1)
    print intersectPlanePlane(plane1,plane2)
    print(roty(np.pi/2.).dot(np.array([1,0,0])))


# In[ ]:



