import pyfits
import numpy as np
from astropy.coordinates import SkyCoord, get_moon, EarthLocation, ICRS, GCRS,AltAz
from astropy.time import Time
from astropy.io.fits.hdu.hdulist import HDUList
from datetime import date
from astral import Astral, Location
from astropy.time import Time
import astropy.units as u

# don't worry about iers
#from astropy.utils import iers
#iers.conf.auto_download = False  
#iers.conf.auto_max_age = None  

# Location, never changes
apl = Location()
apl.name = 'Apache Point Observatory'
apl.region = 'NM'
apl.latitude = 32.780208
apl.longitude = -105.819749
apl.timezone = 'US/Mountain'
apl.elevation = 2790

aplEL=EarthLocation(lon=apl.longitude*u.deg, lat=apl.latitude*u.deg,height=apl.elevation*u.m)


def moon_angle_var(fobj,ext):
    #getting RA and DEC
    sampling=fobj[0].header

    #FINDING INTERPLATE SKY TIME
    fin_mean=[] #time for each moon sky observation interplate, used for moon angle
    h_beg=[]
    h_end=[]
    mean_per=[] #time for each moon sky observation intraplate
    h=ext.header
    h_beg.append(h['TAI-BEG']) #ONLY VARIES WITH PLATE NUMBER, different for k's
    h_end.append(h['TAI-END']) #ONLY VARIES WITH PLATE NUMBER, different for k's
    ttime=(h['TAI-BEG']+h['TAI-END'])/2
    #TAI TO MJD
    time_MJD=ttime/(86400.)

    t=Time(time_MJD, format='mjd')
    moon=get_moon(t,aplEL)
    moonaltaz=moon.transform_to(AltAz(location=aplEL, obstime=t))
    alt=h['ALT']
    az=h['AZ']
    myaltaz=AltAz(alt=alt*u.deg, az=az*u.deg, location=aplEL, obstime=t)
    #moon_coords=SkyCoord(ra=moon_coords.ra, dec=moon_coords.dec)
    #MOON-SKY ANGLE CALCULATIONS
    moon_sky_angle=myaltaz.separation(moonaltaz)
    #setting up MJD format
    date_of_ob=t.datetime
    #calculating moon phase
    moon_phase=apl.moon_phase(date=date_of_ob)

    #0=New moon
    #7=First quarter
    #14=Full moon
    #21=Last quarter
    
    #there are only four moon phases available
    #in astropy error possible here
    phase=0.5*(1-np.cos(moon_phase*np.pi*2/28.))
    var=np.cos(moon_sky_angle.radian)*phase
    return var
