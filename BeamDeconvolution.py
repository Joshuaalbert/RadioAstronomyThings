import numpy as np

def quadratic2elliptic(A,B,C,D=0,E=0,F=-np.log(2)):
    """Invert:
    (A0 cos^2 phi + c0 sin^2 phi)k = A
    (A0-C0)sin 2phi k = B
    (a0 sin^2 phi + c0 cos^2 phi) k = C
    returns bmaj,bmin,bpa[,xc,y if D,E != 0]"""
    if (B**2 - 4*A*C) == 0:
        print "It is parabolic,not elliptic or hyperbolic"
        return None
    if A!=C:#not a circle so should be able to solve second equation
        #A-C = A0 cos^2 phi + c0 sin^2 phi - a0 sin^2 phi - c0 cos^2 phi
        #A-C = A0 (cos^2 phi - sin^2 phi)- c0 (-sin^2 phi  + c0 cos^2 phi) = (A0 - C0) cos 2phi
        #(cos^2 phi - sin^2 phi) = cos 2 phi        
        phi = np.arctan(B/(A-C))/2.#choose your own modulus
    else:#circle
        phi = pi/4.#arbitrary, nonzero cos and sin for the rest
    #rotate x,y to phi to x',y' =  xcos - ysin, xsin + ycos
    #then expand A(x' - xc)^2 + B(x'-xc)(y'-yc) + C(y' - yc)^2 + D(x' - xc) + E(y' - yc) + F = 0
    #then now the coord sys is unrotated relative to the quadratic paramters
    c = np.cos(phi)
    s = np.sin(phi)
    c2 = c*c
    s2 = s*s
    A1 = A*c2 + B*c*s + C*s2
    B1 = 2.*(C-A)*s*c+B*(c2-s2)#should be zero since there's no cross term in the unroated
    #print "Rotated cross term: {0} =? 0".format(B1)
    C1 = A*s2 - B*c*s + C*c2
    D1 = D*c + E*s
    E1 = -D*s + E*c
    if (A1 == 0) or C1 == 0:
        print "degenerate between ellipse and hyperbola"
        return None
    #complete square for x's and y's
    #A1(x-xc)^2 + C1(y-yc)^2 + F = A1xx - 2A1xxc + A1xcxc + C1yy - 2C1yyc + C1ycyc = A1() + D1() + C1() + E1() + A1xcxc + C1ycyc + F =0 
    xc1 = D1/(-2.*A1)
    yc1 = E1/(-2.*C1)#solutions (in rotated frame)
    #now unrotate them
    xc = xc1*c - yc1*s#might be xc1*c + yc1*s
    yc = xc1*s + yc1*c#might be -xc1*s + yc1*c
    #bring the remainder of completed squares to rhs with F
    rhs = -F + D1**2/(4.*A1)  + E1**2/(4.*C1)
    #(x-xc)^2/(bmaj/2)^2 + (y-yc)^2/(bmin/2)^2 = 1
    #A1(x-xc)^2 + C1*y-yc)^2 = rhs
    
    A0 = A1/rhs
    C0 = C1/rhs
    bmaj = np.sign(A0)*2.*np.sqrt(1./np.abs(A0))
    bmin = np.sign(C0)*2.*np.sqrt(1./np.abs(C0))
    if bmaj*bmin < 0:
        print "Hyperbolic solution ;)not what we want here though technically we just inverted"
        return None 
    if bmin > bmaj:
        temp = bmin
        bmin = bmaj
        bmaj = temp
    bpa = phi - np.pi/2.#starts at y
    if E==0 and D==0:
        return bmaj,bmin,bpa*180./np.pi
    return bmaj,bmin,bpa*180./np.pi,xc,yc
        
def elliptic2quadratic(bmaj,bmin,pa,xc=0,yc=0,k=np.log(2)):
    '''a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
    pa in deg
    return A,B,C[,D,E,F if xc,yc!=0]'''

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
    F = A*xc**2 + B*xc*yc + C*yc**2 - 1./k
    if xc==0 and yc==0:
        return A,B,C
    return A,B,C,D,E,F

def deconvolve(A1,B1,C1,A2,B2,C2):
    '''Solves analytically G(A1,B1,C1) = convolution(G(A2,B2,C2), G(Ak,Bk,Ck))
    Returns Ak,Bk,Ck
    A,B,C are quadratic parametrization.
    If you have bmaj,bmin,bpa, then get A,B,C = ecliptic2quadratic(0,0,bmaj,bmin,bpa)
    
    Returns (None,None,None) if solution is delta function'''
    D = B1**2 - 2*B1*B2 + B2**2 - 4*A1*C1 + 4* A2* C1 + 4* A1* C2 - 4* A2* C2
    if (np.abs(D) < 10*(1-2./3.-1./3.)):

        #print "Indefinite... invertibles"
        return (None,None,None)#delta function
    if (D<0.):
        #print "Inverse Gaussian, discriminant D:",D
        pass
    Ak = (-A2* B1**2 + A1* B2**2 + 4* A1* A2* C1 - 4* A1* A2* C2)/D
    Bk = (-B1**2 *B2 + B1* B2**2 + 4* A1* B2* C1 - 4* A2* B1* C2)/D
    Ck = (B2**2 *C1 - B1**2 *C2 + 4* A1* C1* C2 - 4* A2* C1* C2)/D

    return Ak,Bk,Ck

def convolve(A1,B1,C1,A2,B2,C2):
    '''
        Convolves two gaussians with quadratic parametrization:
        A,B,C are quadratic parametrization.
        If you have bmaj,bmin,bpa, then get A,B,C = elliptic2quadratic(0,0,bmaj,bmin,bpa)
        Where g = factor*Exp(-A*X**2 - B*X*Y - C*Y**2)
    '''
    D1 = 4.*A1*C1 - B1**2
    D2 = 4.*A2*C2 - B2**2
    D3 = -2.*B1 * B2 + 4.*A2*C1 + 4.*A1*C2 + D1+D2
    D4 = C2*D1+C1*D2
    #Non-solvable cases
    if (D1*D2*D3*D4 == 0):
        print "Can't convolve..."
        return (None,None,None)
    if (D3 < 0):#always imaginary
        print "D3 < 0, Imaginary solution",D3
        return (None,None,None)
    factor = 2.*np.pi*np.sqrt(D1 + 0j)*np.sqrt(D2 + 0j)/np.sqrt(D3/D4 + 0j)/np.sqrt(D4/(D1*D2) + 0j)
    if np.abs(np.imag(factor)) > 10.*(7./3 - 4./3 - 1.):
        print "Imaginary result somehow..."
        return (None,None,None)
    factor = np.real(factor)
    A = (A2*D1 + A1 * D2)/D3
    B = (B2*D1+B1*D2)/D3
    C = D4/D3
    k = np.log(factor*2.)
    return A,B,C,factor

def findCommonBeam(beams):
    '''Given a list of beams where each element of beams is a dictionary having standard casa format:
    {'major':{'value':<>,'unit':<>},'minor':{'value':<>,'unit':<>},'bpa':{'value':<>,'unit':<>}}

    return the beam parameters '''
    beams_array = []
    for b in beams:
        beams_array.append([b['major']['value'],b['minor']['value'],b['pa']['value']])
    beams_array = np.array(beams_array)
    #Try convolving to max area one
    Areas = beams_array[:,0]*beams_array[:,1]*np.pi/4./np.log(2.)
    idxMaxArea = np.argsort(Areas)[-1]
    A1,B1,C1 = elliptic2quadratic(beams_array[idxMaxArea,0],beams_array[idxMaxArea,1],beams_array[idxMaxArea,2])
    cb = beams_array[idxMaxArea,:].flatten()
    i = 0
    while i < np.size(Areas):
        print np.size(Areas),i
        if i != idxMaxArea:
            #deconlove
            A2,B2,C2 = ecliptic2quadratic(beams_array[i,0],beams_array[i,1],beams_array[i,2])
            Ak,Bk,Ck = deconvolve(A1,B1,C1,A2,B2,C2)
            print Ak,Bk,Ck
            try:
                b = quadratic2elliptic(Ak,Bk,Ck)
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
        Area_init = Areas[idxMaxArea]*1.05
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
                    A1,B1,C1 = elliptic2quadratic(cb[0],cb[1],cb[2])
                    i = 0
                    while i < np.size(Areas):
                        #deconlove
                        A2,B2,C2 = elliptic2quadratic(beams_array[i,0],beams_array[i,1],beams_array[i,2])
                        Ak,Bk,Ck = deconvolve(A1,B1,C1,A2,B2,C2)
                        print Ak,Bk,Ck
                        if Ak is None:
                            print "Failed convolve"
                            cb = None
                            break

                        try:
                            b = quadratic2elliptic(Ak,Bk,Ck)
                            if b is None:
                                cb = None
                                break
                                
                            else:
                                print "Transform possible:",b
                        except:
                            "Transform impossible:"
                            cb = None
                            break
                        i += 1
                    if cb is not None:
                        work = True
            Area *= inc
    return cb
    
def fftGaussian(A,B,C,X,Y):
    D = 4*A*C-B**2
    return 2*np.pi/np.sqrt(D)*np.exp(-4*np.pi/D*(-C*X**2 +B*X*Y -A*Y**2))

def gaussian(A,B,C,X,Y):
    return np.exp(-A*X**2 - B*X*Y - C*Y**2)

def psfTGSS1(dec):
    '''input declination in degrees
    return bmaj(arcsec), bmin(arcsec), bpa(degrees)'''
    if dec > 19.0836824:
        return 25.,25.,0.
    else:
        return 25.,25./np.cos(np.pi*(dec-19.0836824)/180.),0.


if __name__ == '__main__':
    #psf
    bmaj = 1.
    bmin = 0.5
    bpa = 90.
    print "Psf beam, elliptic:",bmaj,bmin,bpa
    Apsf,Bpsf,Cpsf = elliptic2quadratic(bmaj,bmin,bpa)
    print "test quadratic2elliptical"
    print "psf elliptical check:",quadratic2elliptic(Apsf,Bpsf,Cpsf)
    print "Quadratic:",Apsf,Bpsf,Cpsf
    #blob to deconvolve
    bmaj1 = 2.
    bmin1 = 1.5
    bpa1 = 0.
    print "Source ,elliptic:",bmaj1,bmin1,bpa1
    A1,B1,C1 = elliptic2quadratic(bmaj1,bmin1,bpa1)
    print "Quadratic:",A1,B1,C1
    A2,B2,C2,factor = convolve(A1,B1,C1,Apsf,Bpsf,Cpsf)
    bmaj,bmin,bpa = quadratic2elliptic(A2,B2,C2)
    print "Analytic Convolve, elliptic:",bmaj,bmin,bpa
    print "Quadratic:",A2,B2,C2
    Ak,Bk,Ck = deconvolve(A2,B2,C2,Apsf,Bpsf,Cpsf)
    bmaj,bmin,bpa = quadratic2elliptic(Ak,Bk,Ck)
    print "Deconvolve, elliptic:",bmaj,bmin,bpa
    print "Quadratic:",Ak,Bk,Ck
    print "Difference, elliptic:",bmaj-bmaj1,bmin-bmin1,bpa-bpa1
    print "Difference, Quadratic:",Ak-A1,Bk-B1,Ck-C1
    

