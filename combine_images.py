import numpy as np
import pylab as plt
from scipy.stats import norm#is it poisson
import glob,os
#import BeamDeconvolution

import numpy as np
from scipy.optimize import minimize

def ecliptic2quadratic(xc,yc,bmaj,bmin,pa,k=np.log(2)):
	'''a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = k
	pa in deg'''

	#unrotated solution
	a0 = k/(bmaj/2.)**2
	c0 = k/(bmin/2.)**2
	theta = (pa + 90.)*np.pi/180.
	#Rotated Solution
	cos2 = np.cos(theta)**2
	sin2 = np.sin(theta)**2
	A = (a0*cos2 + c0*sin2)
	C = (c0*cos2 + a0*sin2)
	B = (a0 - c0 )*np.sin(2.*theta)
	#Now move center
	D = -2.*A*xc - B*yc
	E = -2.*C*yc - B*xc
	F = A*xc**2 + B*xc*yc + C*yc**2
	return A,B,C,D,E,F

def quad2ecliptic(A,B,C,k=np.log(2)):
	A /= k
	B /= k
	C /= k
	D1 = np.sqrt(A**2-2*A*C+B**2+C**2)
	A0 = (-D1 + A + C)/2.
	C0 = (D1 + A + C)/2.
	D2 = D1 + A - C
	D3 = 2*D2*D1
	if D3 < 0 or A0 < 0 or C0 < 0:
	#	print "No real ecliptic coordinates from quadratic"
		return None
	if (D2 == 0) or (D3 - B*np.sqrt(D3) == 0):
		#print "circle"
		theta = np.pi/2.
	else:
		theta = 2.*np.arctan(B/D2 - np.sqrt(D3)/D2)
	bmaj = 2./np.sqrt(A0)
	bmin = 2./np.sqrt(C0)
	bpa = (theta - np.pi/2.)*180/np.pi#degs
	bpa = np.mod(bpa,180.)-90.
#	while bpa < 0:#to (0,180)
#		bpa += 180.
#	while bpa > 180.:
#		bpa -= 180.
        def chi2(b,ak,bk,ck):
            a,b,c,d,e,f = ecliptic2quadratic(0.,0.,b[0],b[1],b[2])
            return (a-ak)**2 + (b-bk)**2 + (c-ck)**2
        res = minimize(chi2,(bmaj,bmin,bpa),args=(A,B,C),method='Powell')
	return res.x
	

def deconvolve(A1,B1,C1,A2,B2,C2):
	'''Solves analytically G(A1,B1,C1) = convolution(G(A2,B2,C2), G(Ak,Bk,Ck))
	for Ak,Bl,Ck
	Returns None if delta function'''
	D = B1**2 - 2*B1*B2 + B2**2 - 4*A1*C1 + 4* A2* C1 + 4* A1* C2 - 4* A2* C2
	if (np.abs(D) < 10*(1-2./3.-1./3.)):

		#print "Indefinite... invertibles"
		return None#delta function
	if (D<0.):
		#print "Inverse Gaussian, discriminant D:",D
		pass
	Ak = (-A2* B1**2 + A1* B2**2 + 4* A1* A2* C1 - 4* A1* A2* C2)/D
	Bk = (-B1**2 *B2 + B1* B2**2 + 4* A1* B2* C1 - 4* A2* B1* C2)/D
	Ck = (B2**2 *C1 - B1**2 *C2 + 4* A1* C1* C2 - 4* A2* C1* C2)/D

	return Ak,Bk,Ck

def findCommonBeam(beams):
    beams_array = []
    for b in beams:
        beams_array.append([b['major']['value'],b['minor']['value'],b['pa']['value']])
    beams_array = np.array(beams_array)
    #Try convolving to max area one
    Areas = beams_array[:,0]*beams_array[:,1]*np.pi/4./np.log(2.)
    idxMaxArea = np.argsort(Areas)[-1]
    A1,B1,C1,D,E,F = ecliptic2quadratic(0.,0.,beams_array[idxMaxArea,0],beams_array[idxMaxArea,1],beams_array[idxMaxArea,2])
    cb = beams_array[idxMaxArea,:].flatten()
    i = 0
    while i < np.size(Areas):
        print np.size(Areas),i
        if i != idxMaxArea:
            #deconlove
            A2,B2,C2,D,E,F = ecliptic2quadratic(0.,0.,beams_array[i,0],beams_array[i,1],beams_array[i,2])
            Ak,Bk,Ck = deconvolve(A1,B1,C1,A2,B2,C2)
            print Ak,Bk,Ck
            try:
                b = quad2ecliptic(Ak,Bk,Ck,k=np.log(2))
                if b is None:
                    pass
                else:
                    "convolve possible:",b
            except:
                "Failed convolve"
                cb = None
                break
        i += 1
    if cb is None:
        Area_init = Areas[idxMaxArea]
        inc = 1.05#15 iters in area
        works = False
        Area = Area_init
        while Area < 2.*Area_init and not works:
            bmaj_min = np.sqrt(Area*4.*np.log(2)/np.pi)
            bmaj_max = np.sqrt(Area*4.*np.log(2)/np.pi*3.)
            bmaj = np.linspace(bmaj_min,bmaj_max,10)
            pa = np.linspace(-90.,90.,10)
            for bj in bmaj:
                bmin = Area*4.*np.log(2)/np.pi/bj
                for p in pa:
                    cb = (bj,bmin,p)
                    A1,B1,C1,D,E,F = ecliptic2quadratic(0.,0.,cb[0],cb[1],cb[2])
                    i = 0
                    while i < np.size(Areas):
                        #deconlove
                        A2,B2,C2,D,E,F = ecliptic2quadratic(0.,0.,beams_array[i,0],beams_array[i,1],beams_array[i,2])
                        Ak,Bk,Ck = deconvolve(A1,B1,C1,A2,B2,C2)
                        print Ak,Bk,Ck
                        try:
                            b = quad2ecliptic(Ak,Bk,Ck,k=np.log(2))
                            if b is None:
                                pass
                            else:
                                "convolve possible:",b
                        except:
                            "Failed convolve"
                            cb = None
                            break
                        i += 1
                    if cb is not None:
                        work = True

            Area *= inc
    else:
        print "passed",
    return cb
 
def fftGaussian(A,B,C,X,Y):
	D = 4*A*C-B**2
	return 2*np.pi/np.sqrt(D)*np.exp(-4*np.pi/D*(-C*X**2 +B*X*Y -A*Y**2))

def gaussian(A,B,C,X,Y):
	return np.exp(-A*X**2 - B*X*Y - C*Y**2)


def getRms(casaImage,snr=3.,plot=False):
    #one method
    ia.open(casaImage)
    pixel_array = ia.getchunk().flatten()
    ia.close()
    #remove some of tail
    s = np.std(pixel_array)
    pixel_array[np.abs(pixel_array) > snr*s] = np.nan#s includes signal so greater than rms background
    #could repeat
    mu,sigma = norm.fit(pixel_array)#should remove those above 3 sigma first to remove tail
    print "Image Statistics %s: mu = %.2e Jy/beam, sigma = %.2e Jy/beam"%(casaImage,mu,sigma)
    return mu
    #iterative below
#    pixel_array_in = np.copy(pixel_array)
#    s = np.nanstd(pixel_array)
#    while np.max(pixel_array) > snr*s:
#        pixel_array[pixel_array > snr*s] = np.nan
#        s = np.nanstd(pixel_array)
#    if plot:
#        plt.imshow(pixel_array)
#        plt.imshow(pixel_array_in[pixel_array_in > snr*s]*0.)
#        plt.show
#    return s

def stackImages(images,final,weights):
    os.system("cp -r %s %s"%(images[0],final))
    ia.open(images[0])
    array = ia.getchunk()*weights[0]
    ia.close()
    sum = weights[0]
    i = 1
    while i < len(images):
        ia.open(images[i])
        array += ia.getchunk()*weights[i]
        ia.close()
        sum += weights[i]
        i += 1
    ia.open(final)
    ia.putchunk(array/sum)
    ia.close()
    #return array/sum

def getImageInfo(image):
    ia.open(image)
    beam = ia.commonbeam()
    shape = ia.shape()
    ia.close()
    rms = getRms(image)
    return beam,shape,rms

def run(images_glob,output_dir):
    images = glob.glob(images_glob)
    print "Images in:",images
    try:
        os.makedirs(output_dir)
    except:
        print "output dir already exists:",output_dir
    imageInfo = []
    beams = []
    rms = []
    casaImages = []
    idxMaxBeam = 0
    idxMaxSize = 0
    i = 0
    while i < len(images):
        if '.fits' in images[i]:
            casaImages.append(output_dir+'/'+images[i].replace('.fits','.im'))
            if not os.path.exists(casaImages[i]):
                importfits(fitsimage=images[i],imagename = casaImages[i])
            imageInfo.append(getImageInfo(casaImages[i]))
            beams.append(imageInfo[i][0])
            rms.append(imageInfo[i][2])
            if imageInfo[i][0]['minor']['value'] >= imageInfo[idxMaxBeam][0]['minor']['value']:#just minor axis
                idxMaxBeam = i
            if imageInfo[i][1][0] >= imageInfo[idxMaxSize][1][0]:#just x axis size
                idxMaxSize = i
            i += 1
        else:
            print "wrong fits extension format. Use .fits"
            return
    cb = findCommonBeam(beams)
    print "Common beam: ",cb
    print "Regridding to [%d x %d]"%(imageInfo[idxMaxSize][1][0],imageInfo[idxMaxSize][1][1])
    print "Should double that"
    regridImages = []
    regridSmoothedImages = []
    i = 0
    while i < len(images):
        #regrid
        regridImages.append(casaImages[i].replace('.im','-regrid.im'))
        regridSmoothedImages.append(casaImages[i].replace('.im','-regrid-smoothed.im'))
        if not os.path.exists(regridImages[i]):
            imregrid(imagename=casaImages[i],template=casaImages[idxMaxSize],output=regridImages[i],axes=[0,1])
        if not os.path.exists(regridSmoothedImages[i]):
            imsmooth(imagename=regridImages[i],major='%.5farcsec'%(cb[0]),minor='%.5farcsec'%(cb[1]),pa='%.5fdeg'%(cb[2]),outfile=regridSmoothedImages[i],targetres=True,kernel='gauss')
            if not os.path.exists(regridSmoothedImages[i]):
                os.system("cp -r %s %s"%(regridImages[i],regridSmoothedImages[i]))
        i += 1
    finalImage=output_dir+'/CombinedImage.im'
    stackImages(regridSmoothedImages,finalImage,1./rms)
    finalFits = finalImage.replace('.im','.fits')
    exportfits(imagename=finalImage,fitsimage=finalFits)
    print "Combined in:",finalFits

if __name__== '__main__':
    run("*.fits","./combinedImages_weighed3")

