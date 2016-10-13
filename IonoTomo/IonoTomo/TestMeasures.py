
# coding: utf-8

# In[ ]:

import pyrap.measures as pm
from pyrap.quanta import quantity as q

m = pm.measures()
julianSeconds = 4.93258e9
jD = julianSeconds/86400. - 2./24.

time = q('{0}d'.format(jD))
time2 = m.epoch('TAI',time)

m.do_frame(time2)
m.do_frame(m.observatory('GMRT'))
m.do_frame(m.direction('J2000',q([-2.74393],'rad'),q(0.532485,'rad')))
print "gmrt",m.observatory('GMRT')
x = q([1.65701e6,1.65702e6],'m')
y = q([5.79858e6,5.79822e6],'m')
z = q([2.07328e6,2.07326e6],'m')
pos = m.position('itrf',x,y,z)
print "pos",pos
posb = m.asbaseline(pos)
print "posb:",posb
sb = m.baseline('J2000',x,y,z)
print "sb:",sb
out = m.uvw('itrf',x,y,z)
print "out",out







