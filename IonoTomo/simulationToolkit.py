
# coding: utf-8

# In[ ]:

'''
Simulation toolkit for Ionospheric Tomography Package (IonoTomo)
This contains the meat of math.
Created by Joshua G. Albert - albert@strw.leidenuniv.nl
'''
#reload()
import numpy as np
from sys import stdout
import json
import h5py
import os
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import skyComponents
from Logger import Logger
from Atmosphere import Atmosphere

class RadioArray(object):
    def __init__(self,logger,arrayFile = None,name = None,msFile=None,numAntennas=0,earthLocs=None):
        self.log = logger
        self.locs = []
        if arrayFile is not None:
            self.arrayFile = arrayFile
            self.loadArrayFile(arrayFile)
    def loadArrayFile(self,arrayFile):
        '''Loads a csv where each row is x,y,z in geocentric coords of the antennas'''
        d = np.genfromtxt(arrayFile)
        i = 0
        locs = []
        while i < d.shape[0]:
            earthLoc = ac.SkyCoord(x=d[i,0]*au.m,y=d[i,1]*au.m,z=d[i,2]*au.m,frame='itrs')
            locs.append(earthLoc)
            i += 1
        self.addLocs(locs)
    def saveArrayFile(self,arrayFile):
        pass
    def loadMsFile(self,msFIle):
        '''Get antenna positions from ms, array name, frequency'''
        pass

    def addLocs(self,locs):
        '''Add antenna locations. Each location is an EarthLocation.'''
        i = 0
        while i < len(locs):
            self.locs.append(locs[i])
            i += 1
        self.calcCenter()
    def calcCenter(self):
        '''calculates the centroid of the array based on self.locs returns the EarthLocation of center'''
        r0 = np.array([0,0,0])*au.m
        i = 0
        while i < len(self.locs):
            locgc = self.locs[i].earth_location.geocentric
            r0 += np.array([locgc[0].to(au.m).value,
                              locgc[1].to(au.m).value,
                              locgc[2].to(au.m).value])*au.m
            i += 1
        r0 /= float(len(self.locs))
        self.center = ac.SkyCoord(x=r0[0],y=r0[1],z=r0[2],frame='itrs')
        self.log("Center of array: {0}".format(self.center))
        return self.center
    
    def getCenter(self):
        try:
            return self.center
        except:
            self.calcCenter()
            return self.center
        
class SkyModel(object):
    def __init__(self,skyModelFile=None):
        if skyModelFile is not None:
            self.skyModelFile = skyModelFile
            self.loadSkyModel(skyModelFile)
    def loadSkyModel(self,skyModelFile):
        pass
        
        


class Simulation(Logger):
    def __init__(self,simConfigJson=None,logFile=None,help=False,**args):
        super(Simulation,self).__init__(logFile=logFile)
        #logger.__init__(self,logFile=logFile)
        if help:
            self.getAttributes(help)
            exit(0)
        self.speedoflight = 299792458.
        self.simConfigJson = simConfigJson
        self.loadSimConfigJson(simConfigJson)
        if self.simConfigJson is None:
            self.simConfigJson = "SimulationConfig.json"
        #overwrite possible json params with non-None args now
        self.initializeSimulation(**args)
        #create working directory
        try:
            os.makedirs(self.workingDir)
            self.workingDir = os.path.abspath(self.workingDir)
        except:
            self.workingDir = os.path.abspath(self.workingDir)
            self.log("Working directory already exists (beware of overwrite!): {0}".format(self.workingDir)) 
        #start simulation
        self.startSimulation()

    def getAttributes(self,help = False):
        '''Store the definition of attributes "name":[type,default]'''
        self.attributes = {#'maxLayer':[int,0],
                  #'minLayer':[int,0],
                  'skyModelFile':[str,""],
                  'obsStart':[lambda s:at.Time(s,format='isot'),"2000-01-01T00:00:00.000"],
                  'sampleTime':[float,1.],
                  'obsLength':[float,600.],
                  'pointing':[np.array,np.array([0.,0.])],
                  'layerHeights':[np.array,np.array([])],
                  #'maxTime':[int,0],
                  #'minTime':[int,0],
                  'frequency':[float,150e6],
                  'wavelength':[float,0.214],
                  'arrayFile':[str,""],
                  'workingDir':[str,"./output"],
                  #'dataFolders':[list,[]],
                  'dataFolder':[str,""],
                  'precomputed':[bool,False]}#will be replaced with list for parallel processing
        if help:
            self.log("Attribute:[type, default]")
            self.log(self.attributes)
        return self.attributes
    def initializeSimulation(self,**args):
        '''Set up variables here that will hold references throughout'''
        attributes = self.getAttributes()
        for attr in attributes.keys():
            #if in args and non-None then overwrite what is there
            if attr in args.keys():
                if args[attr] is not None:#see if attr already inside, or else put default
                    #setattr(self,attr,getattr(self,attr,attributes[attr][1]))
                    try:
                        setattr(self,attr,attributes[attr][0](args[attr]))
                        #self.log("Set: {0} -> {1}".format(attr,attributes[attr][0](args[attr])))
                    except:
                        self.log("Could not convert {0} into {1}".format(args[attr],attributes[attr][0]))
            else:
                #already set of setting to default
                setattr(self,attr,getattr(self,attr,attributes[attr][0](attributes[attr][1])))
                #self.log("Set: {0} -> {1}".format(attr,getattr(self,attr)))
                
    def startSimulation(self):
        '''Set things to get simulation on track'''

        #set current sim directory dataFolder, and save the settings
        if self.dataFolder == "":
            i = 0
            while self.dataFolder == "":
                dataFolder = "{0}/{1}".format(self.workingDir,i)
                try:
                    os.makedirs(dataFolder)
                    self.dataFolder = os.path.abspath(dataFolder)
                except:
                    self.log("data folder already exists (avoiding overwrite!): {0}".format(dataFolder))
                if i > 1000:
                    self.log("Many data folders. May be a lock causing failed mkdir!")
                    exit(1)
                i += 1
        if not os.path.isdir(self.dataFolder):
            try:
                os.makedirs(self.dataFolder)
            except:
                self.log("Failed to create {0}: ".format(self.dataFolder))
                exit(1)
            self.dataFolder = os.path.abspath(self.dataFolder)
        self.log("Using data folder: {0}".format(self.dataFolder))
        simConfigJson = "{0}/{1}".format(self.dataFolder,self.simConfigJson.split('/')[-1])
        if os.path.isfile(simConfigJson):
            self.log("Found config file in data folder!")
            self.loadSimConfigJson(simConfigJson)
        else:
            self.saveSimConfigJson(simConfigJson)
        #set skymodel
        self.skyModel = SkyModel(self.skyModelFile)
        #set array
        if self.arrayFile is not None:
            self.radioArray = RadioArray(self.log,arrayFile = self.arrayFile)
        else:
            self.radioArray = RadioArray(self.log)
        #set frequency
        try:
            self.setFrequency(self.frequency)
        except:
            self.setWavelength(self.wavelength)
        #time
        self.timeSlices = []
        nsample = np.ceil(self.obsLength/self.sampleTime)
        i = 1
        while i <= nsample:
            self.timeSlices.append(at.Time(self.obsStart.gps + i*self.sampleTime, format = 'gps'))
            i += 1
        
        self.atmosphere = Atmosphere(self.layerHeights)#shall be it's own thing eventually
        self.log("Starting computations... please wait.")
        self.compute()
        self.log("Finished computations... enjoy.")
        
    def restart(self):
        self.log("Resetting simulation...")
        self.startSimulation()
        #change things 
                
    def log(self,message):
        stdout.write("{0}\n".format(message))
        stdout.flush()
    def loadSimConfigJson(self,simConfigJson):
        '''extract sim config from json, and then call initializeSimulation'''
        if simConfigJson is None:
            return
        try:
            f = open(simConfigJson,'r')
        except:
            self.log("No file: {0}".format(simConfigJson))
            exit(1)
        jobject = json.load(f)
        f.close()
        self.log("Loaded from {0}:\n{1}".format(simConfigJson,json.dumps(jobject,sort_keys=True, indent=4, separators=(',', ': '))))
        self.initializeSimulation(**jobject)
        
    def saveSimConfigJson(self,simConfigJson):
        '''Save config in a json to load later.'''
        #self.log("Saving configuration in {0}".format(simConfigJson))
        try:
            jobject = {}
            attributes = self.getAttributes()
            for attr in attributes.keys():
                jobject[attr] = getattr(self,attr,attributes[attr][1])
                if attributes[attr][0] == np.array:#can't store np.array
                    jobject[attr] = list(jobject[attr])
                if attributes[attr][0] == at.core.Time:#can't store np.array
                    jobject[attr] = jobject[attr].isot
            try:
                f = open(simConfigJson,'w+')
                json.dump(jobject,f,sort_keys=True, indent=4, separators=(',', ': '))
                self.log("Saved configuration in: {0}".format(simConfigJson))
                self.log("Stored:\n{0}".format(json.dumps(jobject,sort_keys=True, indent=4, separators=(',', ': '))))
            except:
                self.log("Can't open file: {0}".format(simConfigJson))
        except:
            self.log("Could not get configuration properly.")
    def setWavelength(self,wavelength):
        '''set lambda in m'''
        self.wavelength = wavelength
        self.frequency = self.speedoflight/self.wavelength
    def setFrequency(self,frequency):
        '''set nu in Hz'''
        self.frequency = frequency
        self.wavelength = self.speedoflight/self.frequency
    def getWavelength(self):
        '''return lambda in m'''
        return self.wavelength
    def getFrequency(self):
        '''return nu in Hz'''
        return self.frequency
    def getMinLayer(self):
        '''return the minimum layer, usually zero'''
        return 0
    def getMaxLayer(self):
        return len(self.layerHeights)
    def getMinTime(self):
        return 0
    def getMaxTime(self):
        return len(self.timeSlices)
    def getVisibilities(self,timeIdx,layerIdx):
        '''Returns visibilities <E*E>, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getIntensity(self,timeIdx,layerIdx):
        '''Returns intensity <E.E>, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getTau(self,timeIdx,layerIdx):
        '''Returns optical thickness or transfer function, tau, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getElectronDensity(self,timeIdx,layerIdx):
        '''Returns electron density, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getRefractiveIndex(self,timeIdx,layerIdx):
        '''Returns refractive index at frequency, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getLayerHeight(self,layerIdx):
        '''Get layer height in km'''
        return self.layerHeights[layer]
    def getTimeSlice(self,timeIdx):
        '''return time in seconds since start of simulation'''
        return self.timeSlices[time]
    def compute(self):
        '''Given the parameters simulate the ionotomo, or load if precomputed'''
        if self.precomputed:
            try:
                self.loadResults()
                return
            except:
                self.log("Failed to load precomputed results.")
                exit(1)
        #save before running
        #simConfigJson = "{0}/{1}".format(self.dataFolder,self.simConfigJson.split('/')[-1])
        #self.saveSimConfigJson(simConfigJson)
        timeIdx = 0
        while timeIdx < len(self.timeSlices):
            self.updateLayers(timeIdx)
            layerIdx = len(self.layerHeights) - 1
            #start from the sky model
            prevLayer = self.getSkyLayer(timeIdx)
            while layerIdx >= 0:
                thisLayer = self.getLayer(layerIdx)
                r1 = thisLayer.points()
                r2 = prevLayer.points()
                tau = thisLayer.getTau()
                prevLayer = thisLayer
                layerIdx += 1
            timeIdx += 1
    
    def getLayer(self,layerIdx):
        return self.layers[layerIdx]
        
    def loadResults(self):
        '''Load the results from hdf5 file.'''
        assert False,"loadResults not implemented yet"
    
    def getSkyLayer(self,timeIdx):
        '''Return the Layer object associated with the sky at time index'''
        time = self.timeSlices[timeIdx]
        return self.skyModel
        
    def updateLayers(self,timeIdx):
        '''Wrapper that can call anything that updates the layers 
        and electron content.
        e.g. this could call integration timesteps from a SPH simulation
        of a turbulent atmosphere. Or an analytic toy model.'''
        layerIdx = 0
        while layerIdx < len(self.layerHeights):
            layerIdx += 1
            pass
            
        
