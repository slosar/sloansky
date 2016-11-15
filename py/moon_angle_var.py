import fitsio
import numpy as np
from astropy.coordinates import SkyCoord, get_moon, EarthLocation
from astropy.time import Time
from astropy.io.fits.hdu.hdulist import HDUList
from datetime import date
from astral import Astral, Location
from jdutil import mjd_to_jd, jd_to_date

def moon_angle_var(filename):
    #getting RA and DEC
    sampling=fitsio.read_header(filename)
    RA=sampling['RA']
    DEC=sampling['DEC']

    #FINDING INTERPLATE SKY TIME
    fin_mean=[] #time for each moon sky observation interplate, used for moon angle
    sampling1=fitsio.read_header(filename,0) #finding number of exposures
    h_beg=[]
    h_end=[]
    mean_per=[] #time for each moon sky observation intraplate
    for k in range(4, sampling1['NEXP']+4): #4 is constant for everything
        h=fitsio.read_header(filename,k)
        h_beg.append(h['TAI-BEG']) #ONLY VARIES WITH PLATE NUMBER, different for k's
        h_end.append(h['TAI-END']) #ONLY VARIES WITH PLATE NUMBER, different for k's
        mean_times=(h['TAI-BEG']+h['TAI-END'])/2
        mean_per.append(mean_times)
    tot=np.mean(mean_per)
    fin_mean.append(tot)

    #TAI TO MJD
    new_time=np.array(fin_mean)
    time_MJD=new_time/(86400)
    t=Time(time_MJD, format='mjd')
    moon_coords1=get_moon(t)
    moon_coords=(moon_coords1.spherical._values)
    sky_coords=np.vstack((RA,DEC)).T

    #SPLITTING ARRAY
    tmp = np.array([moon_coords])
    moon_RA = np.array([tmp[0][0]])*(np.pi/180)
    moon_DEC = np.array([tmp[0][1]])*(np.pi/180)
    new_sky_RA=RA*(np.pi/180)
    new_sky_DEC=DEC*(np.pi/180)

    #MOON-SKY ANGLE CALCULATIONS
    moon_sky_angle=(np.arccos((np.sin(new_sky_RA)*np.sin(moon_RA))+np.cos(new_sky_RA)
                    *np.cos(moon_RA)*(np.cos(new_sky_DEC-moon_DEC)))*(180/np.pi))

    #setting up MJD format
    date_of_ob=jd_to_date((mjd_to_jd(time_MJD)))

    #Actual dates
    year = (date_of_ob[0])
    month = (date_of_ob[1])
    day = (date_of_ob[2])
    d= date(int(year), int(month), int(day))

    #giving location
    l = Location()
    l.name = 'Apache Point Observatory'
    l.region = 'NM'
    l.latitude = 32.780208
    l.longitude = -105.819749
    l.timezone = 'US/Mountain'
    l.elevation = 2790

    #calculating moon phase
    moon_phase=l.moon_phase(date=d)

    #0=New moon
    #7=First quarter
    #14=Full moon
    #21=Last quarter
    
    #there are only four moon phases available
    #in astropy error possible here
    if moon_phase == 0:
        phase=0
    if moon_phase == 7:
        phase=0.5
    if moon_phase == 14:
        phase=1
    if moon_phase == 21:
        phase=0.5

    var=moon_sky_angle*phase
    return(var)
