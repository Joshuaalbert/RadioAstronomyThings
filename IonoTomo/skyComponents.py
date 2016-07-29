
# coding: utf-8

# In[23]:

'''
We represent a sky model as:
id,ra,dec,S(nu=nu0),spectralindex=-0.73
ICRS frame
'''
import numpy as np
import astropy.coordinates as ac
import astropy.units as au
import os

class skyModel(object):
    def __init__(self,fileName = ""):
        self.skyModel = None
        self.nu0 = 150e6
        if os.path.isfile(fileName):
            try:
                self.loadSkyModel(fileName)
            except:
                print("Could")
    def getSource(self,id):
        if self.skyModel.shape[0]>id:
            row = self.skyModel[id,:]
            if id != row[0]:
                #something wrong
                return None
            ra = row[1]
            dec = row[2]
            icrsLoc = ac.ICRS(ra=ra*au.deg,dec=dec*au.deg)
            return icrsLoc,row[3],row[4]
    def getFullSky(self):
        icrsLocs = ac.ICRS(ra=self.skyModel[:,1]*au.deg,dec =self.skyModel[:,2]*au.deg)
        return icrsLocs,self.skyModel[:,3],self.skyModel[:,4]
    
    def addSource(self,icrsLoc,S,alpha1,nu0=None):
        ra = icrsLoc.ra.deg
        dec = icrsLoc.dec.deg
        if nu0 is not None:
            self.nu0 = nu0
        if self.skyModel is None:
            self.skyModel = np.array([[0,ra,dec,S,alpha1]])
        else:
            id = self.skyModel.shape[0]
            self.skyModel = np.append(self.skyModel,[[id,ra,dec,S,alpha1]],axis=0)
    def loadSkyModel(self,filename):
        '''load skymodel from file.'''
        self.skyModel = np.genfromtxt(filename,comments='#',delimiter=',',names=True)
        self.nu0 = float(self.skyModel.dtype.names[3].split('Hz')[0].split('S')[1])
        
        
    def saveSkyModel(self,filename):
        '''Save skymodel to file.'''
        np.savetxt(filename,self.skyModel,fmt='%-5d,%5.10f,%5.10f,%5.10f,%+5.5f',delimiter=',',header="id,ra,dec,S({0}Hz),alpha1".format(int(self.nu0)),comments='#')

if __name__=='__main__':
    src1 = ac.ICRS(ra=0*au.deg,dec=0*au.deg),1,-0.5,160e6
    SM = skyModel()
    SM.addSource(*src1)
    SM.saveSkyModel('testSM.csv')
    SM.loadSkyModel('testSM.csv')
    
        


# In[ ]:



