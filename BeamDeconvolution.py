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

            
        

	
def fftGaussian(A,B,C,X,Y):
	D = 4*A*C-B**2
	return 2*np.pi/np.sqrt(D)*np.exp(-4*np.pi/D*(-C*X**2 +B*X*Y -A*Y**2))

def gaussian(A,B,C,X,Y):
	return np.exp(-A*X**2 - B*X*Y - C*Y**2)

