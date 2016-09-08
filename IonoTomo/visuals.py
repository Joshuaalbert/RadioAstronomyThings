
# coding: utf-8

# In[2]:

'''
Create the visualization gui for PropSim
Author: Joshua G. Albert albert@strw.leidenuniv.nl
'''
#reload(simulationToolkit)
import SimulationToolkit

#%matplotlib inline
import numpy as np
import pylab as plt
from sys import stdout
from matplotlib.widgets import RadioButtons, Cursor, Slider, Button
from TextBoxWidget import TextBox
get_ipython().magic(u'matplotlib.inline')

class visual(object):
    def __init__(self,simConfigJson=None,logFile=None,help=False,**args):
        '''Get the args by passing help=True'''
        if help:
            SimulationToolkit.Simulation(help=True)
            exit(0)
        self.simTk = SimulationToolkit.Simulation(simConfigJson=simConfigJson,logFile=logFile,**args)
        self.log = self.simTk.log
        self.curWavelength = self.simTk.getWavelength()
        self.curFrequency = self.simTk.getFrequency()
        self.speedoflight = self.simTk.speedoflight
        self.minLayer = self.simTk.getMinLayer()
        self.maxLayer = self.simTk.getMaxLayer()
        self.curLayer = self.maxLayer#sky 
        self.minTime = self.simTk.getMinTime()
        self.maxTime = self.simTk.getMaxTime()
        self.curTime = self.minTime

        self.dataSelection = ['Visibilities <E*E>',
                              'Intensity <E.E>',
                              'Optical Thickness',
                              'Electron Density',
                              'Refractive Index']
        self.dataSelectionCallers = {'Visibilities <E*E>':self.simTk.getVisibilities,
                                     'Intensity <E.E>':self.simTk.getIntensity,
                                     'Optical Thickness':self.simTk.getTau,
                                     'Electron Density':self.simTk.getElectronDensity,
                                     'Refractive Index':self.simTk.getRefractiveIndex}
    def log(self,message):
        stdout.write("{0}\n".format(message))
        stdout.flush()
        
    def createGui(self):
        self.fig = None
        self.infoDisplayAxes = None
        #layers infoax
        self.layerSliderAxes = None
        self.layerInfoAxes = None
        
        self.layerBottomAxes = None
        self.layerDownAxes = None
        self.layerTopAxes = None
        self.layerUpAxes = None

        self.wavelengthAxes = None
        self.frequencyAxes = None
        self.timeSliderAxes = None
        self.timeInfoAxes = None
        self.timeStartAxes = None
        self.timePrevAxes = None
        self.timeEndAxes = None
        self.timeNextAxes = None
        self.dataSelectionAxes = None
        self.imageAxes = None
        self.colorbarAxes = None
        

        self.resetGui()
    def getAspect(self):
        '''Return image aspect for equidistant coordiantes'''
        #get xlim, ylim
        #return ylim/xlim
        return 1.
    def resizeCb(self,event):
        self.figW = event.width/self.dpi
        self.figH = event.height/self.dpi
        #print self.figW,self.figH
        self.resetGui()
    def resetGui(self):
        '''Builds the gui or rescales things if not yet built'''
        if self.fig is None:
            self.figH = 6#in
            self.figW = 6#in
            self.dpi = float(90)#pt/in
            self.fig = plt.figure(figsize=[self.figW,self.figH],dpi=self.dpi,facecolor='grey')
            plt.connect('resize_event',self.resizeCb)
        self.padding = 0.02#rel
        self.font = 12#pt
        self.lineHeight = self.font*1.2/(self.figH*self.dpi)#rel
        self.infoDisplayWidth = min(self.font*23/(self.figW*self.dpi),1./3.)#rel
        self.infoDisplayHeight = 1. - 3*self.padding - self.lineHeight#rel
        self.infoDisplayBL = (self.padding, self.padding)#rel
        self.imageYlabelMargin = self.font*4.5/(self.figW*self.dpi)#rel
        self.imageXlabelMargin = self.font*2.5/(self.figH*self.dpi)#rel
        self.imageAxesWidth = 1. - 3*self.padding - self.infoDisplayWidth - self.imageYlabelMargin#rel
        self.imageAxesAspect = self.getAspect()#*self.figW/self.figH#h/w
        self.imageAxesHeight = self.imageAxesWidth*self.imageAxesAspect#rel
        self.dataSelectionWidth = self.imageAxesWidth + self.imageYlabelMargin#rel
        self.dataSelectionHeight = 1. - 4*self.padding - self.lineHeight - self.imageAxesHeight - self.imageXlabelMargin#rel
        self.imageAxesBL = (2*self.padding + self.infoDisplayWidth + self.imageYlabelMargin,2*self.padding + self.dataSelectionHeight + self.imageXlabelMargin)
        self.dataSelectionBL = (self.imageAxesBL[0] - self.imageYlabelMargin,self.padding)
        self.colorbarHeight = self.lineHeight#rel
        self.colorbarWidth = self.imageAxesWidth
        self.colorbarBL = (self.imageAxesBL[0], 1-self.padding - self.colorbarHeight)

        self.layerSliderWidth = self.infoDisplayWidth - self.padding - 3*self.font/(self.figW*self.dpi)*1.2
        self.layerSliderHeight = self.font*1.2/(self.figH*self.dpi)#rel
        self.layerSliderBL = (self.infoDisplayBL[0]+self.padding,self.infoDisplayBL[1]+self.padding)
        self.layerInfoWidth = self.infoDisplayWidth - 2*self.padding
        self.layerInfoHeight = self.layerSliderHeight
        self.layerInfoBL = (self.layerSliderBL[0],self.layerSliderBL[1]+self.layerSliderHeight+self.padding)
        
        self.layerBottomHeight = self.layerInfoHeight*1.2
        self.layerBottomWidth = (self.layerInfoWidth-self.padding)/2.
        self.layerBottomBL = (self.layerSliderBL[0],self.layerInfoBL[1] + self.layerInfoHeight + self.padding)
        self.layerDownHeight = self.layerBottomHeight
        self.layerDownWidth = self.layerBottomWidth
        self.layerDownBL = (self.layerBottomBL[0]+self.layerBottomWidth+self.padding,self.layerBottomBL[1])
        
        self.layerTopHeight = self.layerBottomHeight
        self.layerTopWidth = self.layerBottomWidth
        self.layerTopBL = (self.layerBottomBL[0],self.layerBottomBL[1] + self.layerBottomHeight + self.padding)
        self.layerUpHeight = self.layerBottomHeight
        self.layerUpWidth = self.layerBottomWidth
        self.layerUpBL = (self.layerDownBL[0],self.layerTopBL[1])
        
        self.wavelengthHeight = self.layerBottomHeight
        self.wavelengthWidth = self.layerBottomWidth
        self.wavelengthBL = (self.layerDownBL[0],self.layerTopBL[1] + self.layerTopHeight + self.padding)
        self.frequencyHeight = self.layerBottomHeight
        self.frequencyWidth = self.layerBottomWidth
        self.frequencyBL = (self.wavelengthBL[0],self.wavelengthBL[1] + self.wavelengthHeight + self.padding)
        
        self.timeSliderWidth = self.layerSliderWidth
        self.timeSliderHeight = self.layerSliderHeight
        self.timeSliderBL = [self.layerSliderBL[0],self.frequencyBL[1] + self.frequencyHeight + self.padding]
        self.timeInfoWidth = self.layerInfoWidth
        self.timeInfoHeight = self.timeSliderHeight
        self.timeInfoBL = [self.timeSliderBL[0], self.timeSliderBL[1] + self.timeSliderHeight + self.padding]
        
        self.timeStartHeight = self.timeInfoHeight*1.2
        self.timeStartWidth = (self.timeInfoWidth-self.padding)/2.
        self.timeStartBL = (self.timeSliderBL[0],self.timeInfoBL[1] + self.timeInfoHeight + self.padding)
        self.timePrevHeight = self.timeStartHeight
        self.timePrevWidth = self.timeStartWidth
        self.timePrevBL = (self.timeStartBL[0]+self.timeStartWidth+self.padding,self.timeStartBL[1])
        
        self.timeEndHeight = self.timeStartHeight
        self.timeEndWidth = self.timeStartWidth
        self.timeEndBL = (self.timeStartBL[0],self.timeStartBL[1] + self.timeStartHeight + self.padding)
        self.timeNextHeight = self.timeStartHeight
        self.timeNextWidth = self.timeStartWidth
        self.timeNextBL = (self.timePrevBL[0],self.timeEndBL[1])
        #print self.layerSliderBL,self.layerInfoBL,self.layerBottomBL,self.layerDownBL,self.layerTopBL,self.layerUpBL
        if self.colorbarAxes is None:
            self.colorbarAxes = plt.axes([self.colorbarBL[0],self.colorbarBL[1],self.colorbarWidth,self.colorbarHeight])
            self.colorbarAxes.patch.set_facecolor('silver')
            self.colorbarAxes.set_xticks([])
            self.colorbarAxes.set_yticks([])
        else:
            self.colorbarAxes.set_position([self.colorbarBL[0],self.colorbarBL[1],self.colorbarWidth,self.colorbarHeight],which='both')
        
        if self.infoDisplayAxes is None:
            self.infoDisplayAxes = plt.axes([self.infoDisplayBL[0],self.infoDisplayBL[1],self.infoDisplayWidth,self.infoDisplayHeight])
            self.infoDisplayAxes.patch.set_facecolor('silver')
            self.infoDisplayAxes.set_xticks([])
            self.infoDisplayAxes.set_yticks([])
            #self.infoDisplayAxes
            #set things in there
        else:
            self.infoDisplayAxes.set_position([self.infoDisplayBL[0],self.infoDisplayBL[1],self.infoDisplayWidth,self.infoDisplayHeight],which='both')
        
        if self.layerSliderAxes is None:
            self.layerSliderAxes = plt.axes([self.layerSliderBL[0],self.layerSliderBL[1],self.layerSliderWidth,self.layerSliderHeight])
            self.layerSliderAxes.patch.set_facecolor('silver')
            self.layerSliderAxes.set_frame_on(False)
            self.layerSliderAxes.set_xticks([])
            self.layerSliderAxes.set_yticks([])
            self.layerSlider = Slider(self.layerSliderAxes,"", self.minLayer, self.maxLayer, valinit=self.curLayer, valfmt='%d', closedmin=True, closedmax=True, slidermin=None, slidermax=None, dragging=True)
            self.layerSlider.on_changed(self.layerSliderCb)
        else:
            self.layerSliderAxes.set_position([self.layerSliderBL[0],self.layerSliderBL[1],self.layerSliderWidth,self.layerSliderHeight],which='both')
        
        if self.layerInfoAxes is None:
            self.layerInfoAxes = plt.axes([self.layerInfoBL[0],self.layerInfoBL[1],self.layerInfoWidth,self.layerInfoHeight])
            self.layerInfoAxes.patch.set_facecolor('silver')
            self.layerInfoAxes.set_frame_on(False)
            self.layerInfoAxes.set_xticks([])
            self.layerInfoAxes.set_yticks([])
            self.layerInfoBn = Button(self.layerInfoAxes,"Layer: {0} [{1}km]".format(self.curLayer,self.getCurLayerHeight()), color='none',hovercolor='none')
        else:
            self.layerInfoAxes.set_position([self.layerInfoBL[0],self.layerInfoBL[1],self.layerInfoWidth,self.layerInfoHeight],which='both')
        
        if self.layerBottomAxes is None:
            self.layerBottomAxes = plt.axes([self.layerBottomBL[0],self.layerBottomBL[1],self.layerBottomWidth,self.layerBottomHeight])
            self.layerBottomAxes.patch.set_facecolor('silver')
            self.layerBottomAxes.set_frame_on(True)
            self.layerBottomAxes.set_xticks([])
            self.layerBottomAxes.set_yticks([])
            self.layerBottomBn = Button(self.layerBottomAxes,"Bottom", color='grey',hovercolor='white')
            self.layerBottomBn.on_clicked(self.layerBottomBnCb)
        else:
            self.layerBottomAxes.set_position([self.layerBottomBL[0],self.layerBottomBL[1],self.layerBottomWidth,self.layerBottomHeight],which='both')
        
        if self.layerDownAxes is None:
            self.layerDownAxes = plt.axes([self.layerDownBL[0],self.layerDownBL[1],self.layerDownWidth,self.layerDownHeight])
            self.layerDownAxes.patch.set_facecolor('silver')
            self.layerDownAxes.set_frame_on(True)
            self.layerDownAxes.set_xticks([])
            self.layerDownAxes.set_yticks([])
            self.layerDownBn = Button(self.layerDownAxes,"Down", color='grey',hovercolor='white')
            self.layerDownBn.on_clicked(self.layerDownBnCb)
        else:
            self.layerDownAxes.set_position([self.layerDownBL[0],self.layerDownBL[1],self.layerDownWidth,self.layerDownHeight],which='both')
        
        if self.layerTopAxes is None:
            self.layerTopAxes = plt.axes([self.layerTopBL[0],self.layerTopBL[1],self.layerTopWidth,self.layerTopHeight])
            self.layerTopAxes.patch.set_facecolor('silver')
            self.layerTopAxes.set_frame_on(True)
            self.layerTopAxes.set_xticks([])
            self.layerTopAxes.set_yticks([])
            self.layerTopBn = Button(self.layerTopAxes,"Top", color='grey',hovercolor='white')
            self.layerTopBn.on_clicked(self.layerTopBnCb)
        else:
            self.layerTopAxes.set_position([self.layerTopBL[0],self.layerTopBL[1],self.layerTopWidth,self.layerTopHeight],which='both')
        
        if self.layerUpAxes is None:
            self.layerUpAxes = plt.axes([self.layerUpBL[0],self.layerUpBL[1],self.layerUpWidth,self.layerUpHeight])
            self.layerUpAxes.patch.set_facecolor('silver')
            self.layerUpAxes.set_frame_on(True)
            self.layerUpAxes.set_xticks([])
            self.layerUpAxes.set_yticks([])
            self.layerUpBn = Button(self.layerUpAxes,"Up", color='grey',hovercolor='white')
            self.layerUpBn.on_clicked(self.layerUpBnCb)
        else:
            self.layerUpAxes.set_position([self.layerUpBL[0],self.layerUpBL[1],self.layerUpWidth,self.layerUpHeight],which='both')
        
        if self.wavelengthAxes is None:
            self.wavelengthAxes = plt.axes([self.wavelengthBL[0],self.wavelengthBL[1],self.wavelengthWidth,self.wavelengthHeight])
            self.wavelengthAxes.patch.set_facecolor('White')
            self.wavelengthAxes.set_frame_on(True)
            self.wavelengthAxes.set_xticks([])
            self.wavelengthAxes.set_yticks([])
            self.wavelengthTb = TextBox(self.wavelengthAxes,r"$\lambda$ (m)", initial = '%.3f'%(self.curWavelength),color='white',hovercolor='white', label_pad = self.padding)
            self.wavelengthTb.on_submit(self.wavelengthTbCb)
        else:
            self.wavelengthAxes.set_position([self.wavelengthBL[0],self.wavelengthBL[1],self.wavelengthWidth,self.wavelengthHeight],which='both')
        
        if self.frequencyAxes is None:
            self.frequencyAxes = plt.axes([self.frequencyBL[0],self.frequencyBL[1],self.frequencyWidth,self.frequencyHeight])
            self.frequencyAxes.patch.set_facecolor('White')
            self.frequencyAxes.set_frame_on(True)
            self.frequencyAxes.set_xticks([])
            self.frequencyAxes.set_yticks([])
            self.frequencyTb = TextBox(self.frequencyAxes,r"$\nu$ (MHz)", initial = "%.3f"%(1e-6*self.curFrequency),color='white',hovercolor='white', label_pad = self.padding)
            self.frequencyTb.on_submit(self.frequencyTbCb)
        else:
            self.frequencyAxes.set_position([self.frequencyBL[0],self.frequencyBL[1],self.frequencyWidth,self.frequencyHeight],which='both')
        
        if self.timeSliderAxes is None:
            self.timeSliderAxes = plt.axes([self.timeSliderBL[0],self.timeSliderBL[1],self.timeSliderWidth,self.timeSliderHeight])
            self.timeSliderAxes.patch.set_facecolor('silver')
            self.timeSliderAxes.set_frame_on(False)
            self.timeSliderAxes.set_xticks([])
            self.timeSliderAxes.set_yticks([])
            self.timeSlider = Slider(self.timeSliderAxes,"", self.minTime, self.maxTime, valinit=self.curTime, valfmt='%d', closedmin=True, closedmax=True, slidermin=None, slidermax=None, dragging=True)
            self.timeSlider.on_changed(self.timeSliderCb)
        else:
            self.timeSliderAxes.set_position([self.timeSliderBL[0],self.timeSliderBL[1],self.timeSliderWidth,self.timeSliderHeight],which='both')
        
        if self.timeInfoAxes is None:
            self.timeInfoAxes = plt.axes([self.timeInfoBL[0],self.timeInfoBL[1],self.timeInfoWidth,self.timeInfoHeight])
            self.timeInfoAxes.patch.set_facecolor('silver')
            self.timeInfoAxes.set_frame_on(False)
            self.timeInfoAxes.set_xticks([])
            self.timeInfoAxes.set_yticks([])
            self.timeInfoBn = Button(self.timeInfoAxes,"Time: {0} [{1}s]".format(self.curTime,self.getCurTimeSeconds()), color='none',hovercolor='none')
        else:
            self.timeInfoAxes.set_position([self.timeInfoBL[0],self.timeInfoBL[1],self.timeInfoWidth,self.timeInfoHeight],which='both')

        if self.timeStartAxes is None:
            self.timeStartAxes = plt.axes([self.timeStartBL[0],self.timeStartBL[1],self.timeStartWidth,self.timeStartHeight])
            self.timeStartAxes.patch.set_facecolor('silver')
            self.timeStartAxes.set_frame_on(True)
            self.timeStartAxes.set_xticks([])
            self.timeStartAxes.set_yticks([])
            self.timeStartBn = Button(self.timeStartAxes,"Start", color='grey',hovercolor='white')
            self.timeStartBn.on_clicked(self.timeStartBnCb)
        else:
            self.timeStartAxes.set_position([self.timeStartBL[0],self.timeStartBL[1],self.timeStartWidth,self.timeStartHeight],which='both')
        
        if self.timePrevAxes is None:
            self.timePrevAxes = plt.axes([self.timePrevBL[0],self.timePrevBL[1],self.timePrevWidth,self.timePrevHeight])
            self.timePrevAxes.patch.set_facecolor('silver')
            self.timePrevAxes.set_frame_on(True)
            self.timePrevAxes.set_xticks([])
            self.timePrevAxes.set_yticks([])
            self.timePrevBn = Button(self.timePrevAxes,"Prev", color='grey',hovercolor='white')
            self.timePrevBn.on_clicked(self.timePrevBnCb)
        else:
            self.timePrevAxes.set_position([self.timePrevBL[0],self.timePrevBL[1],self.timePrevWidth,self.timePrevHeight],which='both')
        
        if self.timeEndAxes is None:
            self.timeEndAxes = plt.axes([self.timeEndBL[0],self.timeEndBL[1],self.timeEndWidth,self.timeEndHeight])
            self.timeEndAxes.patch.set_facecolor('silver')
            self.timeEndAxes.set_frame_on(True)
            self.timeEndAxes.set_xticks([])
            self.timeEndAxes.set_yticks([])
            self.timeEndBn = Button(self.timeEndAxes,"End", color='grey',hovercolor='white')
            self.timeEndBn.on_clicked(self.timeEndBnCb)
        else:
            self.timeEndAxes.set_position([self.timeEndBL[0],self.timeEndBL[1],self.timeEndWidth,self.timeEndHeight],which='both')
        
        if self.timeNextAxes is None:
            self.timeNextAxes = plt.axes([self.timeNextBL[0],self.timeNextBL[1],self.timeNextWidth,self.timeNextHeight])
            self.timeNextAxes.patch.set_facecolor('silver')
            self.timeNextAxes.set_frame_on(True)
            self.timeNextAxes.set_xticks([])
            self.timeNextAxes.set_yticks([])
            self.timeNextBn = Button(self.timeNextAxes,"Next", color='grey',hovercolor='white')
            self.timeNextBn.on_clicked(self.timeNextBnCb)
        else:
            self.timeNextAxes.set_position([self.timeNextBL[0],self.timeNextBL[1],self.timeNextWidth,self.timeNextHeight],which='both')

        if self.imageAxes is None:
            self.imageAxes = plt.axes([self.imageAxesBL[0],self.imageAxesBL[1],self.imageAxesWidth,self.imageAxesHeight])
            self.imageAxes.patch.set_facecolor('black')
            #self.imageAxes.set_xticks([])
            #self.imageAxes.set_yticks([])
            self.imageAxesCur = Cursor(self.imageAxes,horizOn=True, vertOn=True, color='red',lw=1)
        else:
            self.imageAxes.set_position([self.imageAxesBL[0],self.imageAxesBL[1],self.imageAxesWidth,self.imageAxesHeight],which='both')
                
        if self.dataSelectionAxes is None:
            self.dataSelectionAxes = plt.axes([self.dataSelectionBL[0],self.dataSelectionBL[1],self.dataSelectionWidth,self.dataSelectionHeight])
            self.dataSelectionAxes.patch.set_facecolor('silver')
            self.dataSelectionRB = RadioButtons(self.dataSelectionAxes,self.dataSelection)
            for lab in self.dataSelectionRB.labels:
                lab.set_fontsize=self.font
            self.dataSelectionRB.on_clicked(self.updateDataSelector)
            self.updateDataSelector(self.dataSelection[0])
        else:
            self.dataSelectionAxes.set_position([self.dataSelectionBL[0],self.dataSelectionBL[1],self.dataSelectionWidth,self.dataSelectionHeight],which='both')
        
        self.fig.canvas.draw()
        
    def updateDataSelector(self,label):
        '''Sets self.dataCaller which is a function which takes time,layer and returns (2D array, (xaxis,'units'), (yaxis,'units')'''
        dataCaller = self.dataSelectionCallers[label]
        if dataCaller == None:
            self.log("No attached data caller for: {0}".format(label))
            self.dataCaller = lambda time,layer: (np.random.uniform(size=[10,10]),(-1,1,'rand'),(-1,1,'rand'),(0,1))#(2D array, (xaxis,'units'), (yaxis,'units')
            self.curDataSelected = None
        else:
            self.dataCaller = dataCaller
            self.curDataSelected = label
        self.updateData()
    def layerSliderCb(self,sliderVal):
        self.curLayer = int(sliderVal)
        self.layerInfoBn.label.set_text("Layer: {0} [{1}km]".format(self.curLayer,self.getCurLayerHeight()))
        plt.draw()
        self.updateData()
    def timeSliderCb(self,sliderVal):
        self.curTime = int(sliderVal)
        self.timeInfoBn.label.set_text("Time: {0} [{1}s]".format(self.curTime,self.getCurTimeSeconds()))
        plt.draw()
        self.updateData()
    def layerBottomBnCb(self,event):
        #event.button is button used
        self.curLayer = 0
        self.layerSlider.set_val(self.curLayer)
        self.updateData()
    def layerDownBnCb(self,event):
        #event.button is button used
        self.curLayer = max(self.curLayer-1,0)
        self.layerSlider.set_val(self.curLayer)
        self.updateData()
    def layerTopBnCb(self,event):
        #event.button is button used
        self.curLayer = self.maxLayer
        self.layerSlider.set_val(self.curLayer)
        self.updateData()
    def layerUpBnCb(self,event):
        #event.button is button used
        self.curLayer = min(self.curLayer+1,self.maxLayer)
        self.layerSlider.set_val(self.curLayer)
        self.updateData()
    def timeStartBnCb(self,event):
        #event.button is button used
        self.curTime = 0
        self.timeSlider.set_val(self.curTime)
        self.updateData()
    def timePrevBnCb(self,event):
        #event.button is button used
        self.curTime = max(self.curTime-1,0)
        self.timeSlider.set_val(self.curTime)
        self.updateData()
    def timeEndBnCb(self,event):
        #event.button is button used
        self.curTime = self.maxTime
        self.timeSlider.set_val(self.curTime)
        self.updateData()
    def timeNextBnCb(self,event):
        #event.button is button used
        self.curTime = min(self.curTime+1,self.maxTime)
        self.timeSlider.set_val(self.curTime)
        self.updateData()
    def wavelengthTbCb(self,text):
        '''Used to update simulation to recompute, but thats too much'''
        self.wavelengthTb.text_disp.set_text("%.3f"%(self.curWavelength))
        self.fig.canvas.draw()
        return
        try:
            curWavelength = float(text)
            
        except:
            self.log("{0} is not a float".format(text))
            self.wavelengthTb.text_disp.set_text("%.3f"%(self.speedoflight/self.curFrequency))
        if curWavelength == 0:
            self.log("Wavelength cannot be zero")
            return
        self.curWavelength = curWavelength
        self.simTk.setWavelength(self.curWavelength)
        self.curFrequency = self.simTk.getFrequency()
        self.wavelengthTb.text_disp.set_text("%.3f"%(self.curWavelength))
        self.frequencyTb.text_disp.set_text("%.3f"%(1e-6*self.curFrequency))
        self.fig.canvas.draw()
        self.simTk.restart()
        self.updateData()
    def frequencyTbCb(self,text):
        self.frequencyTb.text_disp.set_text("%.3f"%(1e-6*self.curFrequency))
        self.fig.canvas.draw()
        return
        try:
            curFrequency = float(text)*1e6
        except:
            self.log("{0} is not a float".format(text))
            self.frequencyTb.text_disp.set_text("%.3f"%(1e-6*self.speedoflight/self.curWavelength))
        if curFrequency == 0:
            self.log("Frequency cannot be zero")
            return
        self.curFrequency = curFrequency
        self.simTk.setFrequency(self.curFrequency)
        self.curWavelength = self.simTk.getWavelength()
        self.wavelengthTb.text_disp.set_text("%.3f"%(self.curWavelength))
        self.frequencyTb.text_disp.set_text("%.3f"%(1e-6*self.curFrequency))
        self.fig.canvas.draw()
        self.simTk.restart()
        self.updateData()

    def getCurLayerHeight(self):
        '''returns height of current layer in km.'''
        #use simTK eventually to get the height
        return self.simTk.getLayerHeight(self.curLayer)
    def getCurTimeSeconds(self):
        '''returns time of current frame in seconds.'''
        #use simTK eventually to get the height
        return self.simTk.getTimeSlice(self.curTime)
    def updateData(self):
        '''this is called when layer, time, wavelength, or data caller are changed which draws the new image'''
        #self.log("Updating {0} at {1} and {2}".format(self.curDataSelected,self.curTime,self.curLayer))
        img,xaxis,yaxis,caxis = self.dataCaller(self.curTime,self.curLayer)
        x1,x2,xunits = xaxis[0],xaxis[1],xaxis[2]
        y1,y2,yunits = yaxis[0],yaxis[1],yaxis[2]
        self.imageAxes.clear()
        self.dataImage = self.imageAxes.imshow(img,origin='lower',extent=(x1,x2,y1,y2),interpolation='nearest')
        self.imageAxes.set_xlabel(xunits)
        self.imageAxes.set_ylabel(yunits)
        self.imageAxesCur.disconnect_events()
        self.imageAxesCur = Cursor(self.imageAxes,horizOn=True, vertOn=True, color='red',lw=1)
        self.colorbarAxes.clear()
        self.colorbar = self.fig.colorbar(self.dataImage,cax=self.colorbarAxes,orientation='horizontal')
        self.fig.canvas.draw()
gui = visual('SimulationConfig.json',logFile='logs/0.log')
gui.createGui()
plt.show()


# In[ ]:

for j in range(4):
    sums = []
    for i in range(1000):
        sums.append(np.sum(gui.simTk.atmosphere.cells[j]['electronDensity'][:,:,i]))
    plt.plot(sums)
plt.show()
#gui.simTk.tau[0][1]-gui.simTk.tau[0][2]


# In[2]:

gui.simTk.atmosphere.cells[0]['electronDensity']


# In[ ]:



