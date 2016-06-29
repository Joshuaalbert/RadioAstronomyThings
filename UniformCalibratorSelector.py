import sys
import os
sys.path.append('/software/LOFAR/PyBDSM-1.8.3/LOFAR/build/gnu_opt/installed/lib64/python2.7/site-packages')
import lofar.bdsm
from lofar.bdsm.image import Image
import numpy as np

from scipy.special import binom
import pylab as plt
from matplotlib.patches import Circle

from astropy import units as u
from astropy.coordinates import SkyCoord

from multiprocessing import Pool
import itertools
import time

from astropy.coordinates import SkyCoord
#Minimize Uniformity
def NU(arg):
	'''L2 non-uniformity of the spacings between the calibrators. See Albert 2016.
	arg is a nest tuple for multiprocessing. This could be sped up definitely by precomputing spacings, 
	essentially getting rid of the nest while loops.'''
	cals = arg[0]#idicies of calibrators to calculate over
	subarg = arg[1]#nest tuple
	x = subarg[0]#ra of all calibrators
	y = subarg[1]#dec of all calibrators
	numClusters = subarg[2]#number of calibrators (I suppose I could have used len(cals))
	if numClusters == 1:#otherwise you get divide by zero
		nonuni = numClusters**2/(numClusters**4*(numClusters**2 - 2*numClusters + 3)**2/4.)
		#print "NU:",nonuni
		return nonuni

	#get nyquist sampling size
	maxU = 0.
	maxV = 0.
	i = 0
	while i < numClusters:
		ic = cals[i]#Index of calibrator
		j = 0
		while j < numClusters:
			jc = cals[j]#Index of cal_j
			#print x[ic] - x[jc],y[ic] - y[jc]
			maxU = max(np.abs(x[ic] - x[jc]),maxU)
			maxV = max(np.abs(y[ic] - y[jc]),maxV)
			j += 1
		i += 1
	dU_ = 2./maxU
	dV_ = 2./maxV
	vecU_ = np.linspace(-numClusters*dU_,numClusters*dU_,2*numClusters+1)
	vecV_ = np.linspace(-numClusters*dV_,numClusters*dV_,2*numClusters+1)
	#S_uv defined in Albert 2016.
	S_uv = np.ones([np.size(vecU_),np.size(vecV_)])*numClusters**2
	U_,V_ = np.meshgrid(vecU_,vecV_)
	ip = 0
	while ip < numClusters:
		ipc = cals[ip]
		jp = 0
		while jp < numClusters:
			jpc = cals[jp]
			i = 0
			while i < ip:
				ic = cals[i]
				j = 0
				while j < jp:
					jc = cals[j]
					#was too lazy to efficiently code this loop, precomputing things may be possible
					S_uv += 2.*np.cos((x[ic]-x[jc] - x[ipc]+x[jpc])*U_ + (y[ic]-y[jc] - y[ipc]+y[jpc])*V_)
					j += 1
				i += 1
			jp += 1
		ip += 1
	S_mu = np.mean(S_uv)
	#print numClusters
	nonuni = np.sum(np.abs(S_uv - S_mu)**2)/(numClusters**4*(numClusters**2 - 2*numClusters + 3)**2/4.)
	#print "NU:",nonuni
	return nonuni

def chooseGroupSize(K,ncpu=1,timeFactor=32*11.7/3.17e6,maxTime=None,minGroupSize=5,plot=False):
	'''Chooses the optimal group size to search for uniform calibrators that maximizes group size, and search depth 
	but performs within the required time.
	K - number of calibrators to search for
	ncpu - number of threads that can be run
	timeFactor - to convert complexity to time (I calibrated on Leiden Paracluster)
		Calibrated as for big enough groupSize0,searchDepth0 as Ncpu*time(groupSize0,searchDepth0)[seconds]/computations(groupSize0,searchDepth0)
	maxTime - time in seconds to let it search
	minGroupSize - have at least 5 per searchGroup
	plot - (false) if true will plot results'''
	n = [2,3,4]
	groupSize = 1#general
	if maxTime is None:
		maxTime = np.inf
	G,N,C = [],[],[]
	G_,N_,C_ = [],[],[]
	while groupSize < K:
		computeSize = (groupSize*(groupSize - 1)/2)**2
		for n in [2,3,4]:
			if (K % groupSize) < minGroupSize + n:#remainder will be less than 5 (so not good uniformity)
				groupSize += 1
				continue

			searchSize = groupSize + n
			nCr = binom(searchSize,groupSize)
			computeTime = timeFactor*computeSize*nCr/float(ncpu)
			G_.append(groupSize)
			N_.append(n)
			C_.append(computeTime)
			if computeTime < maxTime:
				G.append(groupSize)
				N.append(n)
				C.append(computeTime)
		groupSize += 1
	
	if len(G) == 0:#try lower constraint
		resG,resN = chooseGroupSize(K,ncpu=ncpu,timeFactor=timeFactor,maxTime=maxTime,minGroupSize=minGroupSize - 1)
		return resG,resN
	N = np.array(N)
	G = np.array(G)
	maxInd = np.argmax(G)
	resG = G[maxInd]
	resN = np.max(N[G==resG])
	if plot:
		plt.scatter(G_,C_,c=N_)
		plt.xlabel('GroupSize')
		plt.ylabel('Est. ComputeTime (seconds)')
		plt.colorbar(label='SearchDepth')
		plt.show()
	print "Search size and depth:",resG,resN
	return resG,resN

def make_directions_file(fitsfile,facetfile='factor_directions.txt',xc=None,yc=None,minSep=30.,minFlux=0.6,numClusters=30,maxSize=None,fov=5,ncpu=32,thresh_pix=100.,thresh_isl=3.,maxTime = 60.,overwrite=False):
	'''
	fitsfile - deepest highres image of field
	facetfile - firections file to produce
	xc,yc - field location ICRS (deg, deg), or else will use mean of calibrator's locations
	fov - field of view in deg
	minSep - arcmin, the minimum seperation between calibrators, merge below
	minFlux - flux cut
	ncpu - number of cpu's to use
	numClusters - number of facets (a cluster may contain several calibrators)
	maxSize - maximum size of source to consider a calibrator (None is don't care)
	thresh_pix - SNR required to recognize a source in bsdm, 100 is good usually
	thresh_isl - boundry of source, 3 is fine
	maxTime - max time to solve uniformity in seconds
	overwrite - overwrite the catalog generated by pybdsm, set false unless you change fitsfile or thresh_pix
	
	Things that could be changed for the better: If you want to feed in your own list of possible calibrators its possible.
	You can do any filtering on that data. The main thing that this does is minimize non-uniformity via an approximated combinatorial search.
	That means if you want to use model components from instrument tables or whatever, you just need to mesh this code with that code.
	This also orders the directions by decreasing background rms (as calculated by pybdsm) which is desireable.
	'''

	#Use pybdsm to get a potential list 
	_img = Image({'filename':fitsfile,'thresh_pix':thresh_pix,'thresh_isl':thresh_isl})
	if (os.path.exists(fitsfile+'.gaul') and overwrite) or not os.path.exists(fitsfile+'.gaul'):
		_img.process()
		_img.write_catalog(outfile=fitsfile+'.gaul', format='ascii', srcroot=None, catalog_type='gaul',  bbs_patches=None, incl_chan=False, incl_empty=False, clobber=True,force_output=False, correct_proj=True, bbs_patches_mask=None)
	try:
		D = np.genfromtxt(fitsfile+'.gaul',comments='#',names=True,skip_header=5)
	except:
		print "No catalog:",fitsfile+'.gaul'
		return

	maxId = int(np.max(D['Isl_id']))
	compactThresh = 1
	compactIsl = []
	while len(compactIsl) <= numClusters+3:#at least as many islands as desired facets+3
		compactIsl = []
		mask = D['Isl_id'] != D['Isl_id']#initial to False
		for i in range(maxId):
			if np.sum(D['Isl_id'] == i) < compactThresh:#add to list of good islands
				compactIsl.append(i)
				mask += D['Isl_id'] == i #XOR
		compactThresh += 1
	print "Taking minimum number of gaussians per calibrator to be:",compactThresh - 1
	print "Clustering may still fail if minSep is too large"

	#Define calibrator properties
	#Within FOV
	#Select islands
	print "Grouping islands into calibrators"
	sizes = []# island sizes
	rms = []
	fluxes = []# island fluxes
	x,y = [],[]#island positions
	if xc is None:
		xc = np.mean(D['RA'])
	if yc is None:
		yc = np.mean(D['DEC'])
	for i in compactIsl:
		maski = D['Isl_id'] == i
		s1 = np.sum(D['Total_flux'][maski]*np.sqrt(D['Min'][maski]**2 + D['Maj'][maski]**2))/np.sum(D['Total_flux'])#sizes, heuristic flux weighted gaussian size.
		s2 = s1#lower limit is s1...
		i = 0
		while i < np.size(D['RA'][maski])-1:#get maximum distance between gaussians in isl
			s2 = max(s2,np.max(np.sqrt((D['RA'][maski][i] - D['RA'][maski])**2 + (D['DEC'][maski][i] - D['DEC'][maski])**2)))#update max distance
			i += 1
		weightedRA = np.sum(D['Total_flux'][maski]*D['RA'][maski])/np.sum(D['Total_flux'][maski])
		weightedDEC = np.sum(D['Total_flux'][maski]*D['DEC'][maski])/np.sum(D['Total_flux'][maski])
		if np.sqrt((weightedRA - xc)**2 + (weightedDEC - yc)**2) < fov/2.:
			sizes.append(s2)
			fluxes.append(np.sum(D['Isl_Total_flux'][maski]))
			#Positions, heuristic flux weighted ra,dec.
			x.append(weightedRA)
			y.append(weightedDEC)
			rms.append(np.max(D['Resid_Isl_rms'][maski]))
	sizes = np.array(sizes)
	fluxes = np.array(fluxes)
	rms = np.array(rms)
	x,y = np.array(x),np.array(y)
	
	#Apply seperation mask.
	print "Filtering by seperation"
	sepFilteredCals = []
	fluxSortMask = np.argsort(fluxes)[::-1]
	collectMask = fluxes==fluxes#which ones to collect
	ind = 0
	while ind < np.size(fluxes):
		i = fluxSortMask[ind]#go by brightest
		if not collectMask[i]:#if already collected move on
			ind += 1
			continue
		dist_mask = np.sqrt((x[i] - x)**2 + (y[i] - y)**2) < minSep/60.#Those that are close
		#merge the ones which are true
		#Only keep brightest one in there
		for j in fluxSortMask:#Could skip because this one should be brightest?
			if dist_mask[j] and collectMask[j]:
				if j not in sepFilteredCals:
					sepFilteredCals.append(j)
					break
		#don't collect these again
		collectMask = np.bitwise_not(np.bitwise_or(np.bitwise_not(collectMask),dist_mask))
		ind += 1
	sepFilteredCals = np.array(sepFilteredCals)
	print "Number of clusters after filtering by serpations:",len(sepFilteredCals)
	numClusters = len(sepFilteredCals)

	bi = sepFilteredCals	
	#bi = np.argsort(fluxes)[::-1]#sorry got lazy at naming at this point
	print "Island Fluxes:",fluxes[bi],x[bi],y[bi]
	print "Maximizing spatial seperation uniformity"
	#time factor appropriate for paracluster cpu's
	initSearchSize,nDepth = chooseGroupSize(numClusters,ncpu=ncpu,timeFactor=32*11.7/3.17e6,maxTime=maxTime)
	searchSize = min(initSearchSize,numClusters-nDepth)
	p = Pool(ncpu)
	group = []
	while searchSize > 0:#len(group) < numClusters:
		#print len(group),len(bi)
		print "grouping first:",searchSize
		searchGroup = bi[:searchSize+nDepth]
		calComb = itertools.combinations(searchGroup,searchSize)
		t1 = time.time()
		NU_Grouping = p.map(NU,itertools.product(calComb,[[x,y,searchSize]]))
		print "Time for",searchSize,"is",time.time()-t1
		#print NU_Grouping
		min_NU_i = np.argmin(NU_Grouping)
		calComb = itertools.combinations(searchGroup,searchSize)
		c = 0
		while c <= min_NU_i:
			cals = calComb.next() 
			c += 1
		new_bi = []
		#aggregate
		for cal in bi:
			if cal not in cals:
				new_bi.append(cal)
			else:
				group.append(cal)
		bi = np.array(new_bi)
		searchSize = min(initSearchSize,numClusters-len(group))
		if len(bi) < searchSize+nDepth:
			searchSize = len(bi) - nDepth
		#print len(bi)

		#searchGroup = bi[:searchSize+2]
		#calComb = itertools.combinations(searchGroup,searchSize)	
	
	cals = np.array(group)
	#np.savez('groupings.npz',NU_Grouping,searchGroup,x,y,numClusters)
	#NU_Grouping = []
	#i = 0
	#for calSel in calComb:
	#	NU_Grouping.append(NU([calSel,[x,y,numClusters]]))
	#	print "Grouping:",i,"NU:",NU_Grouping[i]
	#	i += 1
	#min_NU_i = np.argmin(NU_Grouping)
	#cals = calComb[min_NU_i]
	x_ = x[cals]
	y_ = y[cals]
	rms_ = rms[cals]
	f_ = fluxes[cals]
	s_ = sizes[cals]
	plt.scatter(-x_,y_,c=f_)
	plt.scatter(-x,y,c=fluxes,marker='+')
	plt.xlim([-xc-fov/2.*1.2,-xc+fov/2.*1.2])
	plt.ylim([yc-fov/2.*1.2,yc + fov/2.*1.2])
	circle = Circle((-xc,yc),radius = fov/2.,facecolor='none')
	plt.gca().add_patch(circle)
	plt.colorbar()
	plt.savefig("Facet_locations.pdf",format='pdf')
	plt.show()

	os.system("rm %s;sleep 1"%(facetfile))

	#writing to file
	try:
		f = open(facetfile,'w+')
	except:
		print "Overwriting:",facetfile
		os.system("rm %s;sleep 1"%(facetfile))
		f = open(facetfile,'w+')

	f.write('# name position atrous_do mscale_field_do cal_imsize solint_ph solint_amp dynamic_range region_selfcal region_field peel_skymodel outlier_source cal_radius_deg cal_flux\n')
	f.write('#Generated by UniformCalibratorSelector.py - author: Joshua G. Albert\n')
	ri = np.argsort(rms_)[::-1]
	j = 0
	while j < np.size(x_):	
		i = ri[j]#decreasing rms
		c = SkyCoord(ra=x_[i]*u.degree, dec=y_[i]*u.degree).to_string('hmsdms')
		RA = c.split(' ')[0]		
		DEC = c.split(' ')[0]
		f.write('facet_patch_%d %s,%s empty empty 0 0 0 LD empty empty empty False %.14f %.14f\n'%(i,RA,DEC,s_[i],f_[i]))
		j += 1
	f.close()

	print "Directions file:",facetfile
	return facetfile
	
if __name__ == '__main__':
	#image = '/home/albert/para12_1/products/goods-n-initsubtract40/0/results/initsubtract/field/L137859_SBgr000-10_uv.dppp.dpppaverage.wsclean_high2-image.fits'
	image = '/net/para12/data1/albert/products/goods-n-prefactor-results/images/combinedImages_weighed/CombinedImage.fits'	
#	image = '/net/para37/data2/L281008_SB190-199.2ch8s.wsclean_high2-image.fits'
	c = SkyCoord('12h36m55.000000s +62d14m15.00000s')#getxc,yc
	xc,yc = c.ra.deg,c.dec.deg#not consistent with pybdsm for some reason
	make_directions_file(image,minSep=15,maxTime = 5*60.,overwrite=False)
