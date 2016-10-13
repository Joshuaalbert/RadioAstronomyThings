
# coding: utf-8

# In[1]:

import numpy as np

def makeFakeCatalog(ra,dec,scatter=0.5,n=100):
    i = 0
    catalog = []
    while i  < n:
        dir = "J2000 {0:.8f}deg {1:.8f}deg".format(np.random.normal(loc=ra,scale=scatter),np.random.normal(loc=dec,scale=scatter))
        flux = np.random.uniform()*1.5
        shape = 'S'
        catalog.append([dir,flux,shape,])
        i += 1
    return catalog

def makeCompList(catalog,freq="150MHz"):
    os.system('rm -rf point.cl')
    cl.done()
    for row in catalog:
        print row
        dir = row[0]#"J2000 10h00m00.08s -30d00m02.0s"
        flux = row[1]#1.0
        shape = row[2]#S point or G gaussian
        if shape == 'S':
            cl.addcomponent(dir=dir,
                            flux=flux,
                            fluxunit='Jy',
                            freq=freq, shape="point")
        else:
            bmaj = row[3]#"1arcmin"
            bmin = row[4]#"1arcmin"
            bpa = row[5]#"40deg"
            cl.addcomponent(dir=dir,
                            flux=flux,
                            fluxunit='Jy',
                            freq=freq, shape="Gaussian",
                           majoraxis = bmaj,
                           minoraxis = bmin,
                           positionangle = bpa)
        cl.rename('point.cl')
    cl.done()

catalog = makeFakeCatalog(45,45,scatter=0.5,n=100)
makeCompList(catalog,freq="150MHz")

# In CASA
default("simobserve")
project = "FITS_list"
complist = 'point.cl'
compwidth = '10MHz'
direction = "J2000 3h00m00.0s 45d00m00.0s"
obsmode = "int"
antennalist = '/net/geleen/data2/albert/owncloud/RadioAstronomyThings/IonoTomo/lofar.cycle0.hba.antenna.cfg'
totaltime = "360s"
mapsize = "1deg"
simobserve()
#
#default("simanalyze")
#project = "FITS_list"
#vis="FITS_list.alma.cycle0.compact.ms"
#imsize = [256,256]
#imdirection = "J2000 10h00m00.0s -30d00m00.0s" 
#cell = '0.1arcsec'
#niter = 5000
#threshold = '10.0mJy/beam'
#analyze = True
#simanalyze()



# In[ ]:



