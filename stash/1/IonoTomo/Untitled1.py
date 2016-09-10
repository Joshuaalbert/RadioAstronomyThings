
# coding: utf-8

# In[1]:

import numpy as np
from astropy.coordinates import *
from astropy.units import *
from astropy.time import *

s = SkyCoord(ra=45*deg,dec=45*deg)
eloc = EarthLocation(0*m,0*m,6356752*m)
aa = AltAz(obstime=Time(1234,format='gps'),location=eloc)
strans = s.transform_to(aa) #bring to altaz frame
x = np.cos(strans.alt.rad)*np.sin(strans.az.rad)#points to E
y = np.cos(strans.alt.rad)*np.cos(strans.az.rad)#points to N
z = np.sin(strans.alt.rad)#points to zenith
print x,y,z
print strans.cartesian.xyz
print "x and y are swapped!"


# In[ ]:



