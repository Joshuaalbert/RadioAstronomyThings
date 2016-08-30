import numpy as np
from scipy.optimize import minimize
import pylab as plt

def specCalc(A,nu,nu_ref):
    Sout = A[0]*np.ones_like(nu)
    N = np.size(A)
    i = 1
    while i < N:
        Sout *= 10**(A[i]*(np.log10(nu/nu_ref)**i))
        i += 1
    return Sout

def P(nu,z,alpha,nu_ref):
    '''nu in Hz, z is redshift
    output units: Jy*m^2 = W/Hz'''
    c = 3e8
    H0 = 0.7#units
    ch = 1.32151838e26#m
  #  ch = 4282.7494#Mpc c/H0
    q0 = 0.5#omega0/2
    D = ch*z*(1.+z*(1-q0)/(np.sqrt(1+2*q0*z) + 1 + q0*z))
    return 1e-26*4*np.pi*specCalc(alpha,nu,nu_ref)*D**2/(1+z)**(1+alpha[1])


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
    mask = (np.isnan(a)-1)*-1
    return np.mean(a[mask],**args)

def nanstd(a,**args):
    mask = (np.isnan(a)-1)*-1
    return np.std(a[mask],**args)

    

def doPixelMC(S_array,error_array,nu_array,calError=0.2,nu_ref=150e6,order=None,ax = None, M=1000,title=''):
    #Mask, require snr >= 6
    #print "Masking, SNR >= 6"
    #Area = S_array/(Speak_array/beam_array)
    error_array = np.sqrt(error_array**2 + (calError*S_array)**2)
    snr = S_array/error_array
    print "Uncert including systematics:",error_array
    print "SN:",snr
    if order is None:
        if (np.min(snr) >= 10. ):
            order = 2
        else:
            order = 1
    mask = None
    if order == 1:
        mask = (snr >= 4.)
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
                S[i] = np.abs(np.random.normal(loc = np.random.uniform()*S_array[i], scale = error_array[i])) 
            else:
                S[i] = np.abs(np.random.normal(loc = S_array[i], scale = error_array[i]))#rms_array[i]))
            i += 1
        weights = np.copy(error_array)
        weights[np.bitwise_not(mask)] *= 3
        #print weights
        res[m,:] = doPixel(S,nu_array,errors=weights,nu_ref=nu_ref,order=order)
        if (res[m,1] > 1.) or (res[m,1] < -4.):#remove unphysical
            res[m,:] = np.nan
        #plt.plot(nu,specCalc(res[m,:],nu,nu_ref),alpha=0.025,color='blue',lw=3.)
        m += 1
    a,s = np.nanmean(res,axis=0).reshape([order+1]),np.nanstd(res,axis=0).reshape([order+1])
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
    ax.plot(nu,specCalc(a,nu,nu_ref)*nu,alpha=1.,color='blue',lw=2.,linestyle='--')
    i = 0
    while i < np.size(mask):
        if mask[i]:
            ax.errorbar(nu_array[i],S_array[i]*nu_array[i],yerr=error_array[i]*nu_array[i],fmt='+',color='red')
        else:
            ax.errorbar(nu_array[i],S_array[i]*nu_array[i],yerr=error_array[i]*nu_array[i],fmt='^',color='black')
        i += 1
    #ax.set_title(title)
    ax.text(0.85,0.8,title,transform=ax.transAxes,fontsize=12,weight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_ylabel('S[Jy] (normalized)')
    ax.set_xlabel('Frequency[MHz]')
    ax.set_xticks(nu_array)
    ax.set_xticklabels([ "{0:.0f}".format(i/1e6) for i in nu_array])
    #ax.set_ylim([max(0.,np.min((S_array-error_array)/nu_array)*0.5),np.max((S_array+error_array)/nu_array)*1.5])

    #plt.show()
    return S_array*1e3,error_array*1e3,a[1:],s[1:],specCalc(a,1400e6,nu_ref),specCalc(a-s,1400e6,nu_ref),specCalc(a+s,1400e6,nu_ref),P(1400e6,0.516,a,nu_ref),P(1400e6,0.516,a-s,nu_ref),P(1400e6,0.516,a+s,nu_ref)

def doImages(images,rms,center=(289.271167, -33.522389),r500deg=1010./6.231/3600.):
    '''Images are gridded and smoothed properly
    0.13s/M M is mc-sample  s'''
    nu_array = []
    beamsize_array = []
    image_array = []
    rms_array = []
    for i in range(len(images)):
        print "Operating on:",images[i]
        ia.open(images[i])
        csys = ia.coordsys()
        c1 = csys.convert(coordin=[center[0]+r500deg,center[1]-r500deg,0,0],unitsin=['deg','deg','pix','pix'],unitsout=['pix','pix','pix','pix'])
        c2 = csys.convert(coordin=[center[0]-r500deg,center[1]+r500deg,0,0],unitsin=['deg','deg','pix','pix'],unitsout=['pix','pix','pix','pix'])
        r1 = rg.box(blc=[c1[0], c1[1],0,0],trc=[c2[0], c2[1],0,0])
        
        summary = ia.summary()
        nu_array.append(summary['refval'][3])
        beamsize_array.append(summary['restoringbeam']['major']['value'] * summary['restoringbeam']['major']['value'] * np.pi / np.log(2) / 4.)
        rms_array.append(rms[i])
        image_array.append(ia.getregion(r1)[:,:,0,0]*beamsize_array[i])
        ia.close()
        print image_array[i].shape
    beamsize_array = np.array(beamsize_array)
    rms_array = np.array(rms_array)
    nu_array = np.array(nu_array)

    rows = np.shape(image_array[0])[0]
    cols = np.shape(image_array[0])[1]
    alpha1 = np.zeros([rows,cols])
    alpha2 = np.zeros([rows,cols])
    error1 = np.zeros([rows,cols])
    error2 = np.zeros([rows,cols])
    i = 0
    while i < rows:
        j = 0
        while j < cols:
            a,s = doPixelMC(np.array([image_array[p][i,j] for p in range(len(image_array))]),rms_array,nu_array,nu_ref=150e6,order=None,M=100)
            print a,s
            if np.size(a) == 1:
                alpha1[i,j] = a[0]
                error1[i,j] = s[0]
                alpha2[i,j] = np.nan
                error2[i,j] = np.nan
            elif np.size(a) == 2:
                alpha1[i,j] = a[0]
                error1[i,j] = s[0]
                alpha2[i,j] = a[1]
                error2[i,j] = s[1]
            j += 1
        i += 1

    import pylab as plt
    plt.imshow(alpha1)
    plt.colorbar()
    plt.show()

def printRes(S,E,a,s,P14,low,up,p14,pl,pu):
    N = a.shape[0]
    i = 0
    st = ""
    while i < N:
        st += "$%.1f \pm %.1f$ & $%.1f \pm %.1f$ & $%.1f \pm %.1f$ & $%.2f \pm %.2f$ & $%.1f^{+%.1f}_{-%.1f}$ & $%.1f^{+%.1f}_{-%.1f}$"%(S[0],E[0],S[1],E[1],S[2],E[2],a[i],s[i],P14*1e3,(P14-low)*1e3,(up-P14)*1e3,p14*1e-24,(p14-pl)*1e-24,(pu-p14)*1e-24)
        if i < N-1:
            st += " & "
        i += 1
    print st

def beamArea(bmaj,bmin):
    return np.pi*bmaj*bmin/4./np.log(2)

if __name__ == '__main__':

    #nu_test = np.array([73,150,320,610])*1e6
    #S_test = specCalc(np.array([50,-0.7,-0.03]),nu_test,nu_ref=147e6)
   # print S_test
    #rms_test = np.array([2.,1.4,0.1,0.09])*1e-3
    #S_test += np.array([np.random.normal(scale=rms_test[i]) for i in range(np.size(rms_test)) ])
   # print S_test
    #plt.plot(nu_test,S_test)
    #plt.show()
    #print doPixelMC(S_test,rms_test,nu_test,nu_ref=147e6,order=2)
    #beam_array = np.array([443.6,90.6,18.6])#arcsec^2
    #rms_array = np.array([1.4e-3,419e-6,119e-6])*beam_array#Jy
    #doImages(['150_sm_reg.im','320_sm_reg.im','610_sm_reg.im'],rms_array)

    if True:
        nu_array = np.array([147.667e6,322.667e6,608.046e6])#Hz
        beam_array = (3600*3600)*np.array([beamArea(0.011321,0.0045504),beamArea(5.467392603556E-03,3.138420846727E-03),beamArea(1.990104383892E-03,1.352697875765E-03)])#as^2
        print "Beams:",beam_array
        #rms_array = np.array([4.894e-3,419e-6,119e-6])#Jy/beamm#Apx = np.array([5.25**2,2.5**2,1.])
        rms_array = np.array([1.4e-3,120e-6,90e-6])#Jy/beamm
        order = 1
        nu_ref = 150e6
        f,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex='col')
        #C1_S = np.array([0.0127236,0.0309509,0.0547769])*beam_array#Jy
        #C1_sb = np.array([6.59469e-5,0.000190467,0.000348898])*beam_array#Jy
        #A = C1_S/C1_sb
        #N = A/Apx
        #C1 = np.array([3*1.4e-3,0.001582096151567331,0.0027823728140710575])
        #error = np.array([3*1.4e-3,8.39241765249418E-4,1.944806254088021E-4])
        #print C1,C1p,error,nu_array
        #print "C1:",printRes(*doPixelMC(C1,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax1,order=order,title="C1"))

        #C2_S = np.array([0.108783,0.0871488,0.0649436])*beam_array#Jy
        #C2_sb = np.array([0.000438531,0.000366942,0.000267258])*beam_array#Jy
        #C2_S = np.array([0.,0.0871488,0.0649436])*beam_array#Jy
        #C2_sb = np.array([0.,0.000366942,0.000267258])*beam_array#Jy
        #A = C2_S/C2_sb
        #N = A/Apx
#        C2 = np.array([0.23937716489673583/2.,0.012790796071889128/2.,0.0020545694080541434])
#        error = np.array([0.0032304360847119165/np.sqrt(2),5.608656299598429E-4/np.sqrt(2),2.2810157907246877E-4])
#        print "C2:",printRes(*doPixelMC(C2,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax2,order=order,title="C2"))

        #C3_S = np.array([0.267871,0.151403,0.0819312])*beam_array#Jy
        #C3_sb = np.array([0.00069419,0.000410584,0.000225706])*beam_array#Jy
        #C3_S = np.array([0.,0.151403,0.0819312])*beam_array#Jy
        #C3_sb = np.array([0.,0.000410584,0.000225706])*beam_array#Jy
        #A = C3_S/C3_sb
        #N = A/Apx
#        C3 = np.array([0.23937716489673583/2,0.012790796071889128/2,0.002436982233881717])
#        error = np.array([0.0032304360847119165/np.sqrt(2),5.608656299598429E-4/np.sqrt(2),2.0611703235450427E-4])
#        print "C3:",printRes(*doPixelMC(C3,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax3,order=order,title="C3"))
        
        C23 = np.array([0.23937716489673583,0.012790796071889128,0.0020545694080541434+0.002436982233881717])
        error = np.array([0.0032304360847119165,5.608656299598429E-4,np.sqrt((2.0611703235450427E-4)**2+(2.2810157907246877E-4)**2)])
        print "C1+2:",printRes(*doPixelMC(C23,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax1,order=order,title="C1+2"))

        #C4_S = np.array([0.00466349,0.00677101,0.00836541])*beam_array#Jy
        #C4_sb = np.array([0.000169197,0.000120373,0.000160873])*beam_array#Jy
        #A = C4_S/C4_sb
        #N = A/Apx
        #print "C4:",doPixelMC(C4_S,np.sqrt(N)*rms_array,nu_array,nu_ref=nu_ref,order=order)


        #NW1_S = np.array([2.89578,1.998997,1.02739])*beam_array
        #NW1_sb = np.array([0.00338911,0.00216105,0.00109647])*beam_array#Jy
        #A = NW1_S/NW1_sb
        #N = A/Apx
        NW1 = np.array([0.16322469466877484,0.07567930968809516,0.041786190545708514])
        error = np.array([0.003341108607762229,8.142405496674987E-4,2.9522537791728815E-4])
        print "NW1:",printRes(*doPixelMC(NW1,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax2,order=order,title="NW1"))
        
        #NW2_S = np.array([3.13491,2.26586,1.21121])*beam_array#Jy
        #NW2_sb = np.array([0.00392201,0.00302114,0.00164996])*beam_array#Jy
        #A = NW2_S/NW2_sb
        #N = A/Apx
        NW2 = np.array([0.1693916039443248,0.08051560801375851,0.04662496890086012])
        error = np.array([0.0034195052942391654,8.487566730527668E-4,3.593382779606264E-4])
        print "NW2:",printRes(*doPixelMC(NW2,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax3,order=order,title="NW2"))

        #H_S = np.array([0.573249,0.11603,0.0751669])*beam_array#Jy
        #H_sb = np.array([0.00086659,0.000164291,0.000107381])*beam_array#Jy
        #A = H_S/H_sb
        #N = A/Apx
        H = np.array([0.175-(0.003*405./beam_array[0]),0.03601-(2.24e-4*400/beam_array[1]),0.05363-(4.93e-5*393.7/beam_array[2])])
        error = np.array([0.01993313031070147,rms_array[1]*np.sqrt(400./beam_array[1]),0.05363])
        print "H:",printRes(*doPixelMC(H,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax4,order=order,title="H"))

        #E_S = np.array([0.489795,0.609117,0.31653])*beam_array#Jy
        #E_sb = np.array([0.000683475,0.000818981,0.00041869])*beam_array#Jy
        #A =np.array( E_S/E_sb
        #N = A/Apx
        E = np.array([0.017545526812082806,0.02269147569517214,0.012827118830079233])
        error = np.array([0.003428040102224943,6.021116362376412E-4,2.894771085992118E-4])
        print "E:",printRes(*doPixelMC(E,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax5,order=order,title="E"))

        #X_S = np.array([0.427212,0.146576,0.0662238])*beam_array#Jy
        #X_sb = np.array([0.000704534,0.000257717,0.000117836])*beam_array#Jy
        #A = X_S/X_sb
        #N = A/Apx
        X1 = np.array([0.01202464178177413,0.004479764384444517,0.0024986433289830826])
        error = np.array([0.003485259201044762,rms_array[1],1.545623734472893E-4])
#6.785856666900907E-4
        print "X1:",printRes(*doPixelMC(X1,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax6,order=order,title="X1"))

        X2 = np.array([0.003950086255518878,0.004503567941614206,0.004485312896473871])
        error = np.array([0.0038607061772221213,rms_array[1],2.3338154276943916E-4])
        #7.59771827938236E-4
        print "X2:",printRes(*doPixelMC(X2,error,nu_array,calError=0.2,nu_ref=nu_ref,ax=ax7,order=order,title="X2"))

        #S_S = np.array([0.380486,0.410589,0.239914])*beam_array#Jy
        #S_sb = np.array([0.000690223,0.000791496,0.000484674])*beam_array#Jy
        #A = S_S/S_sb
        #N = A/Apx
        S = np.array([0.03519454180339246,0.015938588709435295,0.011570946169000226])
        error = np.array([0.0027222467224997777,6.218102056801986E-4,2.2068641023520924E-4])
        print "S:",printRes(*doPixelMC(S,error,nu_array,calError=0.1,nu_ref=nu_ref,ax=ax8,order=order,title="S"))

        f.subplots_adjust(hspace=0)
        f.subplots_adjust(wspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes], visible = False)
        plt.setp([a.get_yticklabels() for a in f.axes], visible = False)
        plt.setp(f.axes[6].get_xticklabels(), visible = True)
        plt.setp(f.axes[7].get_xticklabels(), visible = True)

        plt.show()









