
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

class layer(object):
    def __init__(self):
        self.processedTimes = []
    def processed(self,time):
        if time in self.processedTimes:
            return True
        else:
            return False

class simulation(object):
    def __init__(self,simConfigJson=None,help=False,**args):
        if help:
            self.getAttributes(help)
            exit(0)
        self.speedoflight = 299792458.
        self.simConfigJson = simConfigJson
        self.loadSimConfigJson(simConfigJson)
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
                  'layerHeights':[np.array,np.array([])],
                  #'maxTime':[int,0],
                  #'minTime':[int,0],
                  'timeSlices':[np.array,np.array([])],
                  'frequency':[float,150e6],
                  'wavelength':[float,0.214],
                  'array':[str,""],
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
                setattr(self,attr,getattr(self,attr,attributes[attr][1]))
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
                    self.log("data folder already exists (avoiding overwrite!): {0}".format(self.dataFolder))
                i += 1
        if os.path.isdir(self.dataFolder):
            self.log("Using data folder: {0}".format(self.dataFolder))
            simConfigJson = "{0}/{1}".format(self.dataFolder,self.simConfigJson.split('/')[-1])
            if os.path.isfile(simConfigJson):
                self.log("Found config file in data folder!")
                self.loadSimConfigJson(simConfigJson)
        else:
            os.makedirs(self.dataFolder)
            self.dataFolder = os.path.abspath(self.dataFolder)
            
        
        try:
            self.setFrequency(self.frequency)
        except:
            self.setWavelength(self.wavelength)
        #sort and reduce times and layers
        sortedTime = []
        sortedLayers = []
        for timeSlice,layerHeight in zip(self.timeSlices,self.layerHeights):
            if timeSlice not in sortedTime:
                sortedTime.append(timeSlice)
            if layerHeight not in sortedLayers:
                sortedLayers.append(layerHeight)
        self.timeSlices = np.sort(np.array(sortedTime))
        self.layerHeights = np.sort(np.array(sortedLayers))
                
        self.log("Starting computations... please wait.")
        self.compute()
        self.log("Finished computations... enjoy.")
        
    def restart(self):
        self.log("Reseting simulation...")
        self.curDataFolderIdx += 1
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
        try:
            jobject = {}
            attributes = self.getAttributes()
            for attr in attributes.keys():
                jobject[attr] = getattr(self,attr,attributes[attr][1])
                if attributes[attr][0] == np.array:#can't store np.array
                    jobject[attr] = list(jobject[attr])
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
    def getVisibilities(self,time,layer):
        '''Returns visibilities <E*E>, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getIntensity(self,time,layer):
        '''Returns intensity <E.E>, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getTau(self,time,layer):
        '''Returns optical thickness or transfer function, tau, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getElectronDensity(self,time,layer):
        '''Returns electron density, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getRefractiveIndex(self,time,layer):
        '''Returns refractive index at frequency, (x1,x2,'units'), (y1,y2,'units')'''
        return (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'))
    def getLayerHeight(self,layer):
        '''Get layer height in km'''
        return self.layerHeights[layer]
    def getTimeSlice(self,time):
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
        simConfigJson = "{0}/{1}".format(self.dataFolder,self.simConfigJson.split('/')[-1])
        self.saveSimConfigJson(simConfigJson)
        
    def loadResults(self):
        '''Load the results from hdf5 file.'''
        pass

