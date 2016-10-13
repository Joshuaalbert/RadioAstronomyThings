
# coding: utf-8

# In[13]:

import numpy as np

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
            print self.n
        else:
            self.n = np.array(normal)
            self.n = self.n/np.linalg.norm(self.n)
            self.d = -self.n.dot(p1)
    def __repr__(self):
        return "Plane: normal: {0}, d=-n.p {1}".format(self.n,self.d)
    
class BoundedPlane(Plane):
    '''assumes convex hull of 4 points'''
    def __init__(self,vertices):
        if len(vertices) != 4:
            print("Not enough vertices")
            return
        super(BoundedPlane,self).__init__(*vertices)

if __name__ == '__main__':    
    print "Building bounded plane with squarly arranged vertices:"
    print Plane(np.array([-1,-1,-1]),
                np.array([1,-1,-1]),
                np.array([1,-1,1]),
                np.array([-1,-1,1]))
    print BoundedPlane([np.array([-1,-1,-1]),
                        np.array([1,-1,-1]),
                        np.array([1,-1,1]),
                        np.array([-1,-1,1])])


# In[ ]:




# In[ ]:



