import pylab as plt
import aplpy
import numpy as np
import pyfits
import astropy

def get_aips_beam(fits):
	radiohead = pyfits.getheader(fits)
	try:
		h = radiohead['HISTORY']
		for line in h:
			if 'BMAJ' in line:
			#	print line
				l = line.split('BPA=')
				bpa = float(l[1])
				l = l[0].split('BMIN=')
				bmin = float(l[1])
				l = l[0].split('BMAJ=')
				bmaj = float(l[1])
				break
		beam=(bmaj,bmin,bpa)#deg
	except:
		bpa = radiohead['BPA']
		bmin = radiohead['BMIN']
		bmaj = radiohead['BMAJ']
		beam=(bmaj,bmin,bpa)#deg
        print fits
        print "Beam:",bmaj*3600.,bmin*3600,bpa
        print "BEAM AREA:",(bmaj*3600*bmin*3600)*np.pi/4.*np.log(2)
	return beam

def plot_cross(ra,dec, size):
	lines = []
	for r,d in zip(ra,dec):
		lines.append(np.array([[r,d+size/8.],[r,d+size/2.]]))#top
		lines.append(np.array([[r,d-size/8.],[r,d-size/2.]]))#bottom
		lines.append(np.array([[r-size/8.,d],[r-size/2.,d]]))#left
		lines.append(np.array([[r+size/8.,d],[r+size/2.,d]]))#right
	return lines


a2kpc=6.231
r500kpc=1010.
#center of bcg
bcg = (289.271167, -33.522389)

imgdir = '/net/para34/data1/albert/casa/images/paper/'
#sys.path.mkdirs(imgdir)

tgssfits = 'TGSS_plckg004-19_150.fits'
radio610fits = 'plckg004-19_610_multiscale.fits'
radio320fits = 'plckg004-19_320.fits'
radio150fits = 'plckg004-19_150_multiscale.fits'
#radio320fits = '../spec_map_scripts/plck004-19_320_21arcsec.fits'
#radio150fits = '../spec_map_scripts/plck004-19_150_21as_69cut.fits'
#xrayfits='xmms_plckg004-19.fits'
xrayfits='Image_total_norm_corr.fits'
optfits = 'gmos_plckg004-19.fits'
optimg='PLCK_G004.5-19.5.tiff'

do150 = True
do320 = True
do610opt = True

#Xray base
def plotXray():
	fig = aplpy.FITSFigure(xrayfits)
	fig.set_theme('pretty')
	fig.show_colorscale(cmap='terrain_r',stretch='sqrt',vmin=0.7,vmax=25,aspect='equal',smooth=1)
	fig.recenter(bcg[0],bcg[1],radius = (r500kpc+180.)/a2kpc/3600.)
	fig.ticks.set_color('black')
	fig.add_grid()
	fig.grid.set_color('black')

	#add r500, and scalebar
	fig.show_circles(bcg[0],bcg[1],radius = r500kpc/a2kpc/3600.,color='black',linewidth=2)
	fig.add_scalebar(r500kpc/a2kpc/3600./2.,label='500 kpc',corner='top right',frame=False,color='black',alpha=1.0,linewidth=2.)
	gmos_data = np.genfromtxt('redshifts_fors2mxu_gmos.cat',comments='#',usecols=[1,2,7])
	ra = gmos_data[:,0]
	dec = gmos_data[:,1]
	z = gmos_data[:,2]
	mask = (z>0.50)*(z<0.53)
	fig.show_markers(ra,dec,layer='spec_src',edgecolor='black',facecolor='none',marker='+',s=200,linewidth=1.5,alpha=1.0)
        #fig.show_regions('xrayCon.reg')
        fig.show_contour(xrayfits,levels=np.array([0,16.0251,32.0502,48.0752,64.1003]),colors='grey',linewidths=1.5)
	return fig

#Xray base
def plotOpt():
	c = astropy.coordinates.SkyCoord('+19h17m02s -33d30m40.4s',astropy.coordinates.ICRS)
	fig = aplpy.FITSFigure(optfits)
	fig.set_theme('pretty')
	fig.show_rgb(optimg)
	fig.recenter(c.ra,c.dec,radius = 60./3600.)
	fig.ticks.set_color('white')
	#fig.add_grid()
	#fig.grid.set_color('white')

	#add r500, and scalebar
	#fig.show_circles(bcg[0],bcg[1],radius = r500kpc/a2kpc/3600.,color='black',linewidth=2)
	fig.add_scalebar(100./a2kpc/3600.,label='100 kpc',corner='top right',frame=False,color='white',alpha=1.0,linewidth=2.)
	gmos_data = np.genfromtxt('redshifts_fors2mxu_gmos.cat',comments='#',usecols=[1,2,7])
	ra = gmos_data[:,0]
	dec = gmos_data[:,1]
	z = gmos_data[:,2]
	mask = (z>0.50)*(z<0.53)
	fig.show_markers(ra,dec,layer='spec_src',edgecolor='cyan',facecolor='none',marker='o',s=200,linewidth=1.5,alpha=1.0)
	return fig

def plotAnn(fig,color="black",ann=None):
	ra = ['+19h16m59.586s','+19h17m01.085s','+19h17m11.181s','+19h17m15.776s','+19h17m06.86s','+19h17m04.64s','+19h17m09.49s','+19h17m08.053s','+19h17m07.276s','+19h17m18s']
	dec = ['-33d30m24.99s','-33d31m09.10s','-33d32m05.99s','-33d32m24.44s','-33d31m15.6s','-33d33m44.9s','-33d33m21.64s','-33d31m55.52s','-33d31m35.21s','-33d32m00s']
	text = ["NW2",'NW1','E','X1','H','S','C','C1','C2','X2']
	for r,d,t in zip(ra,dec,text):
		c = astropy.coordinates.SkyCoord("%s %s"%(r,d),astropy.coordinates.ICRS)
		x = c.ra.deg
		y = c.dec.deg
		print x,y,t
                if (ann is None) or ((ann is not None) and (t in ann)):
                    if t != 'C':
                        fig.add_label(x, y, t, style="normal", color=color, weight="bold", size=18,horizontalalignment='left')

#do 150
if do150:
	fig = plotXray()
	plotAnn(fig)
	beam150 = get_aips_beam(radio150fits)
	fig.show_beam(beam150[0],beam150[1],beam150[2],corner='bottom left',frame=True,color='red')
	sens = 1.4e-3
	fig.show_contour(radio150fits,levels=np.array([5,10,20,40,80,160])*sens,colors='red',linewidths=1.5)

	beam610 = get_aips_beam(radio610fits)
	fig.show_beam(beam610[0],beam610[1],beam610[2],corner='bottom left',frame=False,color='blue')
	sens = 90e-6
	fig.show_contour(radio610fits,levels=np.array([5,10,20,40,80,160])*sens,colors='blue',linewidths=1.5)

	fig.save('radio150-xray.pdf')


if do320:
	fig = plotXray()
	plotAnn(fig)
	beam320 = get_aips_beam(radio320fits)
	fig.show_beam(beam320[0],beam320[1],beam320[2],corner='bottom left',frame=True,color='red')
	sens = 120e-6
	fig.show_contour(radio320fits,levels=np.array([5,10,20,40,80,160])*sens,colors='red',linewidths=1.5)

	beam610 = get_aips_beam(radio610fits)
	fig.show_beam(beam610[0],beam610[1],beam610[2],corner='bottom left',frame=False,color='blue')
	sens = 90e-6
	fig.show_contour(radio610fits,levels=np.array([5,10,20,40,80,160])*sens,colors='blue',linewidths=1.5)

	fig.save('radio320-xray.pdf')

if do610opt:
	fig = plotOpt()
	plotAnn(fig,color="white",ann=['NW1','NW2'])
	beam610 = get_aips_beam(radio610fits)
	fig.show_beam(beam610[0],beam610[1],beam610[2],corner='bottom left',frame=True,color='red')
	sens = 90e-6
	fig.show_contour(radio610fits,levels=np.array([5,10,20,40,80,160])*sens,colors='red',linewidths=1.5)

	fig.save('radio610-opt.pdf')



#contours


#xray = aplpy.FITSFigure(xrayfits,figure=fig)
#xray.show_colorscale(cmap='gist_heat')
#xray.recenter(bcg[0],bcg[1],radius = r500kpc/a2kpc/3600.)


#beam
#Set ticklabels

plt.show()
#plt.show()

