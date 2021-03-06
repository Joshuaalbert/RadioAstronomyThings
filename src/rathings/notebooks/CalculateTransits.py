import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import sys
import pylab as plt
import numpy as np

def getTransit(lon,lat,time,ra,dec,tzOffset=0):
    objLoc = ac.SkyCoord(ra = ra*au.deg,dec = dec*au.deg,frame='icrs')
    eloc = ac.EarthLocation(lon=lon*au.deg,lat=lat*au.deg).geocentric
    loc = ac.SkyCoord(*eloc,frame='itrs')
    initTime = at.Time(time,format='isot',scale='utc').gps - tzOffset*3600.
    t0 = initTime - 12*3600.
    t1 = initTime + 12*3600.
    times = np.linspace(t0,t1,24*12)
    alts = []
    azs = []
    i = 0
    while i < np.size(times):
        obsFrame = ac.AltAz(obstime = at.Time(times[i],format='gps',scale='utc'),location=loc)
        altaz = objLoc.transform_to(obsFrame)
        alt = altaz.alt.deg#height above horizon
        az = altaz.az.deg#east of north
        alts.append(alt)
        azs.append(az)
        i += 1
    azs = np.array(azs)
    alts = np.array(alts)
    f,(ax1,ax2) = plt.subplots(2,sharex=True)
    ax1.plot_date(at.Time(times+tzOffset*3600.,format='gps',scale='utc').plot_date,alts)
    ax1.set_ylabel("Altitude above hor. (deg)")
    ax1.set_title("Alt/Az ({0},{1}) @ {2} of {3},{4}".format(lon,lat,time,ra,dec))
    ax2.plot_date(at.Time(times+tzOffset*3600.,format='gps',scale='utc').plot_date,azs)
    ax2.set_ylabel("Azimuth, E of N (deg)")
    f.autofmt_xdate()
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    plt.show()
    transitIdx = np.argmax(alts)
    riseIdx = np.argmin(alts[:transitIdx]**2)
    setIdx = np.argmin(alts[transitIdx:]**2)
    rise = at.Time(times[riseIdx],format='gps').isot
    transit = at.Time(times[transitIdx],format='gps').isot
    set = at.Time(times[setIdx],format='gps').isot
    return rise,transit,set

def getSolarTransit(obj,lon,lat,time,tzOffset=0,plotSun=True):
    #objLoc = ac.SkyCoord(ra = ra*au.deg,dec = dec*au.deg,frame='icrs')
    eloc = ac.EarthLocation(lon=lon*au.deg,lat=lat*au.deg)
    loc = ac.SkyCoord(*(eloc.geocentric),frame='itrs')
    initTime = at.Time(time,format='isot',scale='utc').gps - tzOffset*3600.
    t0 = initTime - 12*3600.
    t1 = initTime + 12*3600.
    times = np.linspace(t0,t1,24*12)
    alts = []
    azs = []
    if plotSun:
        alSun = []
        azSun = []
    i = 0
    while i < np.size(times):
        tnow = at.Time(times[i],format='gps',scale='utc')
        obsFrame = ac.AltAz(obstime = tnow,location=loc)
        with ac.solar_system_ephemeris.set('builtin'):
            objLoc = ac.get_body(obj,tnow,eloc)
            if plotSun:
                sunLoc = ac.get_body('Sun',tnow,eloc)
                altazSun = sunLoc.transform_to(obsFrame)
                alSun.append(altazSun.alt.deg)
                azSun.append(altazSun.az.deg)
            altaz = objLoc.transform_to(obsFrame)
            alt = altaz.alt.deg#height above horizon
            az = altaz.az.deg#east of north
            alts.append(alt)
            azs.append(az)
        i += 1
    azs = np.array(azs)
    alts = np.array(alts)
    f,(ax1,ax2) = plt.subplots(2,sharex=True)
    ax1.plot_date(at.Time(times+tzOffset*3600.,format='gps',scale='utc').plot_date,alts)
    if plotSun:
        alSun = np.array(alSun)
        azSun = np.array(azSun)
        ax1.plot_date(at.Time(times+tzOffset*3600.,format='gps',scale='utc').plot_date,alSun)
    ax1.set_ylabel("Altitude above hor. (deg)")
    ax1.set_title("Alt/Az ({0},{1}) @ {2} of {3}".format(lon,lat,time,obj))
    ax2.plot_date(at.Time(times+tzOffset*3600.,format='gps',scale='utc').plot_date,azs)
    if plotSun:
        ax2.plot_date(at.Time(times+tzOffset*3600.,format='gps',scale='utc').plot_date,azSun)
    ax2.set_ylabel("Azimuth, E of N (deg)")
    f.autofmt_xdate()
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    plt.show()
    transitIdx = np.argmax(alts)
    riseIdx = np.argmin(alts[:transitIdx]**2)
    setIdx = np.argmin(alts[transitIdx:]**2)
    rise = at.Time(times[riseIdx],format='gps').isot
    transit = at.Time(times[transitIdx],format='gps').isot
    set = at.Time(times[setIdx],format='gps').isot
    return rise,transit,set




if __name__=='__main__':
    try:
        try:
            lon = float(sys.argv[1])
        except:
            lon = -79.9663600
        try:
            lat = float(sys.argv[2])
        except:
            lat = 45.4334000
        time = str(sys.argv[3])
        if len(time.split("T")) != 2:
            time = "2016-08-16T23:00:00.000"
        print ("Lon: {0}, Lat: {1}, Time: {2}".format(lon,lat,time))
        obj = sys.argv[4:]
        print ("{0}".format(obj))
        if len(obj) == 2:
            ra = float(obj[0])
            dec = float(obj[1])
            getTransit(lon,lat,time,ra,dec,tzOffset=-6.)
        else:
            obj = str(obj[0])
            try:
                skyCoord = ac.SkyCoord.from_name(obj)
                ra = skyCoord.ra.deg
                dec = SkyCoord.dec.deg
                print( "{0} ra: {1}, dec: {2}".format(obj,ra,dec))
                getTransit(lon,lat,time,ra,dec,tzOffset=-6.)
            except:
                try:
                    getSolarTransit(obj,lon,lat,time,tzOffset=-6)
                except:
                    try:
                        ra = float(obj.split(',')[0])
                        dec = float(obj.split(',')[1]) 
                        getTransit(lon,lat,time,ra,dec,tzOffset=-6.)
                    except:
                        ra = 0.712305555
                        dec = 41.2694444
                        getTransit(lon,lat,time,ra,dec,tzOffset=-6.)
                    print( "ra: {0}, dec: {1}".format(ra,dec))
    except:
        print("Usage: {} longtude(deg) latitude(deg) time(yyyy-mm-ddThh:mm:ss.sss object(string or ra,dec no spaces)".format(sys.argv[0]))
        lon = -79.9663600
        lat = 45.4334000
        time = "2016-08-16T23:00:00.000"
        #obj = "M31"
        #skyCoord = ac.SkyCoord.from_name(obj)
        #ra = skyCoord.ra.deg
        #dec = SkyCoord.dec.deg    
        ra = 0.712305555
        dec = 41.2694444
    #getSolarTransit('saturn',lon,lat,time,tzOffset=-6)
        getTransit(lon,lat,time,ra,dec,tzOffset=-6.)

    #print rise, transit, set
