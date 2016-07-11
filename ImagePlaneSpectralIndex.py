import numpy as np
import pylab as plt
from scipy.stats import norm#is it poisson
import glob,os
from BeamDeconvolution import findCommonBeam

def specCalc(A,nu,nu_ref):
    Sout = A[0]*np.ones_like(nu)
    N = np.size(A)
    i = 1
    while i < N:
        Sout *= 10**(A[i]*(np.log10(nu/nu_ref)**i))
        i += 1
    return Sout

def doPixel(S_array,nu_array,errors = None, nu_ref=147e6,order=1):
    '''Assume log10 S = log10 A0 + A1 log10 nu + A2 (log10 nu) ^2'''
    #A0 = np.mean(S_array/(10**(-0.8*np.log10(nu_array/nu_ref))))
    #Linearized fit (non gaussian error)
    p = np.polyfit(np.log10(nu_array/nu_ref),np.log10(S_array),order)
    p[order] = 10**p[order]
    a_init = np.copy(p[::-1])
    if errors is None:
        errors = np.ones_like(S_array)
    res = minimize(lambda A: np.sqrt(np.mean((S_array - specCalc(A,nu_array,nu_ref))**2/errors**2)),a_init,method = 'Powell')
    return res.x

def nanmean(a,**args):
    mask = np.bitwise_not(np.isnan(a))
    return np.mean(a[mask],**args)

def nanstd(a,**args):
    mask = np.bitwise_not(np.isnan(a))
    return np.std(a[mask],**args)

def doPixelMC(S_array,error_array,nu_array,beam_array,cell_array,calError=0.2,nu_ref=150e6,order=None, M=1000,plot=False):
    #Mask, require snr >= 6
    #print "Masking, SNR >= 6"
    #Area = S_array/(Speak_array/beam_array)
    snr = S_array/error_array
    print "Uncert including systematics:",error_array*(1+calError)
    print "SN:",snr
    if order is None:
        if (np.min(snr) >= 10. ):
            order = 2
        else:
            order = 1
    mask = None
    if order == 1:
        mask = (snr >= 7.)
    else:
        if order is 2:
            mask = (snr >= 49.)
    if mask is None:
        mask = (snr==snr)
    if np.sum(mask) < 2:
        print "not enough good data"#,S_array
        print "Can only give constraints"
        #return
        #return np.array([np.nan]),np.array([np.nan])
    #S_array = S_array[mask]
    if np.sum(mask) == 0:
        #fit point
        mask = (snr == np.max(snr))
    print "MASK:",mask
    res = np.zeros([M,order+1])
    S = np.copy(S_array)

    m = 0
    while m < M:
        i = 0
        while i < np.size(S_array):
            if not mask[i]:
                S[i] = np.abs(np.random.normal(loc = np.random.uniform()*S_array[i], scale = (1.+calError)*error_array[i])) 
            else:
                S[i] = np.abs(np.random.normal(loc = S_array[i], scale = (1.+calError)*error_array[i]))#rms_array[i]))
            i += 1
        res[m,:] = doPixel(S,nu_array,errors=error_array,nu_ref=nu_ref,order=order)
        if (res[m,1] > 1.) or (res[m,1] < -4.):#remove unphysical
            res[m,:] = np.nan
        #plt.plot(nu,specCalc(res[m,:],nu,nu_ref),alpha=0.025,color='blue',lw=3.)
        m += 1
    a,s = np.nanmean(res,axis=0).reshape([order+1]),np.nanstd(res,axis=0).reshape([order+1])
    if plot:
        #print ax
        if ax is None:
            f,ax = plt.subplots(1)
        nu = np.linspace(np.min(nu_array),np.max(nu_array),20)
        S = np.zeros([res.shape[0],np.size(nu)])
        i = 0
        while i < res.shape[0]:
            S[i,:] = specCalc(res[i,:],nu,nu_ref)
            i += 1
        #mu = np.nanmean(S,axis=0)
        mu = specCalc(a,nu,nu_ref)
        std = np.nanstd(S,axis=0)
        #ax.plot(nu,(mu+std),alpha=1.,color='black',lw=3.,linestyle='--')
        #ax.plot(nu,(mu-std),alpha=1.,color='black',lw=3.,linestyle='--')
        #ax.plot(nu,mu/norm,alpha=1.,color='black',lw=3.,linestyle='-')
        ax.plot(nu,specCalc(a,nu,nu_ref),alpha=1.,color='blue',lw=2.,linestyle='--')
        i = 0
        while i < np.size(mask):
            if mask[i]:
                ax.errorbar(nu_array[i],S_array[i],yerr=(1.+calError)*error_array[i],fmt='o',color='red')
            else:
                ax.errorbar(nu_array[i],S_array[i],yerr=(1.+calError)*error_array[i],fmt='^',color='black')
            i += 1
        #ax.set_title(title)
        ax.text(0.85,0.8,title,transform=ax.transAxes,fontsize=12,weight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.set_ylabel('S[Jy] (normalized)')
        ax.set_xlabel('Frequency[Hz]')
        ax.set_ylim([max(0.,np.min(S_array-error_array)*0.5),np.max(S_array+error_array)*1.5])

        #plt.show()
    return a[1:],s[1:]#,specCalc(a,1400e6,nu_ref),specCalc(a-s,1400e6,nu_ref),specCalc(a+s,1400e6,nu_ref)


def mp_spectral(args):
    mask = args[0]
    S_array = args[1]
    error_array = args[2]
    nu_array = args[3]
    beam_array = None
    cell_array = None
    calError=0.2
    nu_ref=150e6
    order=1
    M=1000
    if mask:
        a,s = doPixelMC(S_array,error_array,nu_array,beam_array,cell_array,calError=calError,nu_ref=nu_ref,order=order, M=M000)
        print a,s
        return (a,s)
    else:
        return (np.nan,np.nan)
    
def CalculateSpectralIndex(regridSmoothedImages,spectralMap,spectralMapError,rmss,nu_array):
    os.system("cp -r %s %s"%(regridSmoothedImages[0],spectralMap))
    os.system("cp -r %s %s"%(regridSmoothedImages[0],spectralMapError))
    fluxes = []
    masks = []
    for image,rms in zip(regridSmoothedImages,rmss):
        tb.open(image)
        map = tb.getcol('map')
        shape = map.shape
        fluxes.append(np.flatten(map))
        masks.append(fluxes[-1]>3*rms)
        tb.close()
    mask = mask[0]
    for m in masks:
        mask = np.bitwise_and(mask,m)
    args = []
    i = 0 
    while i < np.size(mask):
        S_array = []
        j = 0 
        while j < len(fluxes):
            S_array.append(fluxes[j][i])
            j += 1
        S_array = np.array([S_array])
        error_array = np.array(rmss)
        nu_array = np.array(nu_array)
        args.append(mask[i],S_array,error_array,nu_array)
        i += 1
    res = p.map(mp_spectral,args)
    spectralIndexArray = np.zeros_like(fluxes[0])
    spectralIndexErrorArray = np.zeros_like(fluxes[0])
    i = 0
    while i < np.size(spectralIndexArray):
        spectralIndexArray[i] = res[0]
        spectralIndexErrorArray[i] = res[1]
        i += 1
    spectralIndexArray = np.reshape(spectralIndexArray,shape)
    spectralIndexErrorArray = np.reshape(spectralIndexErrorArray,shape)
    tb.open(spectralMap,nomodify=False)
    tb.putcol('map',spectralIndexArray)
    tb.close()
    tb.open(spectralMapError,nomodify=False)
    tb.putcol('map',spectralIndexArray)
    tb.close()
    print "Spectral map and error in:",spectralMap,spectralMapError
    
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

def getImageInfo(image):
    '''Get required information for calculation
    '''
    ia.open(image)
    summary = ia.summary()
    axes = summary['axisnames']
    freq = summary['refval'][axes=='Frequency']
    ra = summary['refval'][axes=='Right Ascension']*180./np.pi
    dec = summary['refval'][axes=='Declination']*180./np.pi
    beam = summary['restoringbeam']
    incr = min(summary['incr'][0],summary['incr'][1])
    ia.close()
    rms = getRms(image)
    return beam,incr,freq, ra, dec, rms

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
    nu_array = []
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
            rms.append(imageInfo[i][5])
            nu_array.append(imageInfo[i][2])
            if beams[i]['minor']['value'] >= beams[idxMaxBeam]['minor']['value']:#just minor axis
                idxMaxBeam = i
            if imageInfo[i][1] <= imageInfo[idxMaxSize][1]:#just x axis size
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
    spectralMap=output_dir+'/SpectralIndexMap.im'
    spectralMapError=output_dir+'/SpectralIndexMapError.im'
    CalculateSpectralIndex(regridSmoothedImages,spectralMap,spectralMapError,rms,nu_array)
    finalFits = spectralMap.replace('.im','.fits')
    exportfits(imagename=spectralMap,fitsimage=finalFits)
    print "Spectral Map:",finalFits
    finalFits = spectralMapError.replace('.im','.fits')
    exportfits(imagename=spectralMapError,fitsimage=finalFits)
    print "Spectral Map Error:",finalFits

    
if __name__=='__main__':
    run("*.fits","./SpectralIndexMap")
