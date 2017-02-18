#!/usr/bin/env python
from glob import glob
import numpy as np
import pyfits
import sys, cPickle
import matplotlib.pyplot as plt
from optparse import OptionParser
from mpi4py import MPI
import moonsun_angle_var as mav
import time
from astropy.coordinates import SkyCoord,  ICRS, BarycentricTrueEcliptic
import astropy.units as u
import healpy as hp
from SolarActivity import SolarActivity
# don't worry about iers
#from astropy.utils import iers
#iers.conf.auto_download = False  
#iers.conf.auto_max_age = None  

cnlist=["moon", "sun", "ecliptic","galactic", "SFD", "halpha", "haslam"]


def options():

    dirname="sky_data/"
    outcat="catalog.cp"
    skipfact=10000
    parser = OptionParser()

    parser.add_option("-D","--dirname", dest="dirname", default=dirname,
                      help="Root directory of data files", type="string")
    parser.add_option("-o","--output filename", dest="outcat", default=outcat,
                      help="Output filename", type="string")
    parser.add_option("--skip", dest="skipfact", default=skipfact,
                      help="skip factor", type="int")
    parser.add_option("--Naz", dest="Naz", default=8,
                      help="# azmimuths", type="int")
    parser.add_option("--weather", dest="weather", default=0,
                      help="weather: 0=ignore, 1=use", type="int")
    parser.add_option("--sunspots", dest="sunspots", default=0,
                      help="sunspots: 0=ignore, 1=use", type="int")
    parser.add_option("--chips", dest="chips", default=0,
                      help="chips: 0=ignore, 1=use", type="int")
    parser.add_option("--platepos", dest="platepos", default=0,
                      help="chips: 0=ignore, 1=use", type="int")

    for n in cnlist:
        parser.add_option("--"+n, dest=n, default=0,
                      help=n+": 0=ignore, 1=use, 2=cache", type="int")
    return parser.parse_args()
    

def main():
    #init
    o,args=options()
    setMPI(o)
    ## file list, Number of files, Nf after skipping, mystart-end indices
    filelist,Nf,Nfp, mystart, myend =getFlist(o)
    # variable names
    vnames,caches = getVnames(o)
    # actually get variables
    cat,totvars = getVars(o,filelist,caches,mystart,myend,Nfp)
    # save cached values of things like moon
    saveCaches(o,caches)
    #save actual catalog
    saveCatalog(o,vnames,cat,totvars)

def setMPI(o):
    global comm,rank,size
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    size=comm.Get_size()
    if (o.moon==2):
        time.sleep(rank)

def getMaps(o):
    SFD,halpha,haslam=None,None,None
    if o.SFD==2:
        if rank==0:
            print "Reading SFD map"
            SFD=pyfits.open("data/lambda_sfd_ebv.fits")[1].data["TEMPERATURE"]
        SFD=comm.bcast(SFD,root=0)
    if o.halpha==2:
        if rank==0:
            print "Reading halpha map"
            halpha=pyfits.open("data/lambda_halpha_fwhm06_0512.fits")[1].data["TEMPERATURE"]
        halpha=comm.bcast(halpha,root=0)
    if o.haslam==2:
        if rank==0:
            print "Reading haslam map"
            haslam=pyfits.open("data/haslam408_dsds_Remazeilles2014.fits")[1].data["TEMPERATURE"]
            haslam=haslam.reshape((12*512**2,))
        haslam=comm.bcast(haslam,root=0)

    return SFD,halpha,haslam

def getVars(o,filelist,caches,mystart,myend,Nfp):
    cat=[]
    totvars=[]
    SFD,halpha,haslam=getMaps(o)
    if o.sunspots:
        sa=SolarActivity(comm)

    for ifile,filename in enumerate(filelist[mystart*o.skipfact:myend*o.skipfact:o.skipfact]):
        if rank==0:
            print "Doing: %i/%i"%(ifile*size, Nfp)
        da=pyfits.open(filename)
        fiber=da[0].header['FIBERID']
        vars=[]
        for i,ext in enumerate(da[4:]):
            var=[]
            ## values of variables
            alt,az,ra,dec=[ext.header[x]/180.*np.pi for x in ["ALT","AZ","RA","DEC"]]
            ## add azimuth vars
            azstep=2*np.pi/o.Naz
            for ai in range(o.Naz):
                taz=azstep*(ai+0.5)
                if (az<0):
                    az+=2*np.pi
                daz=abs(az-taz)
                #print daz,taz,az
                assert (daz<2*np.pi)
                if (daz>np.pi):
                    daz=2*np.pi-daz
                if (daz<azstep):
                    v=(1.-daz/(azstep))*ext.header["AIRMASS"]
                else:
                    v=0
                var.append(v)
            if o.weather:
                try:
                    #var.append(ext.header['DUSTA']/50000)
                    #var.append(ext.header['DUSTB']/50000)
                    var.append(ext.header['SEEING50'])
                except:
                    print "bad",filename, i
                    sys.exit(1)

            if o.platepos:
                var.append(100*np.sqrt(ext.header["ARCOFFX"]**2+ext.header["ARCOFFY"]**2))
                    
                    
            if o.sunspots:
                var.append(sa.activity(ext.header['MJD']))

            if o.chips:
                ##camera
                var.append(int(fiber>500))
                ## chip no
                if (fiber>500):
                    cfiber=fiber-500
                else:
                    cfiber=fiber
                var.append(np.sin((cfiber-1)*np.pi/499)) ## zero at edges

            if o.moon:
                if (o.moon==2):
                    caches["moon"][(filename,i)]=mav.moon_angle_var(da,ext)
                var.append(caches["moon"][(filename,i)])

            if o.sun:
                if (o.sun==2):
                    caches["sun"][(filename,i)]=mav.sun_angle_var(da,ext)
                var.append(caches["sun"][(filename,i)])

            scord=None
            if o.ecliptic:
                if (o.ecliptic==2):
                    scord=SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
                    ecl=scord.geocentrictrueecliptic
                    caches["ecliptic"][(filename,i)]=ecl.lat
                ecllat=caches["ecliptic"][(filename,i)]
                for d in [1.,5.,10.,20.]:
                    var.append(np.exp(-(ecllat**2/(2*(d*u.deg)**2))))

            if o.galactic:
                if (o.galactic==2):
                    if scord is None:
                        scord=SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
                    gal=scord.galactic
                    caches["galactic"][(filename,i)]=gal.b
                    
                galb=caches["galactic"][(filename,i)]
                for d in [1.,5.,10.,20.]:
                    var.append(np.exp(-(galb**2/(2*(d*u.deg)**2))))
            
            if o.SFD:
                if (o.SFD==2):
                    if scord is None:
                        scord=SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
                    gal=scord.galactic
                    theta,phi=np.pi/2-gal.b/u.rad, gal.l/u.rad
                    px=hp.ang2pix(512,theta.value,phi.value,nest=True)
                    caches["SFD"][(filename,i)]=SFD[px]
                var.append(caches["SFD"][(filename,i)])

            if o.halpha:
                if (o.halpha==2):
                    if scord is None:
                        scord=SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
                    gal=scord.galactic
                    theta,phi=np.pi/2-gal.b/u.rad, gal.l/u.rad
                    px=hp.ang2pix(512,theta.value,phi.value,nest=True)
                    caches["halpha"][(filename,i)]=halpha[px]
                var.append(caches["halpha"][(filename,i)])

            if o.haslam:
                if (o.haslam==2):
                    if scord is None:
                        scord=SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
                    gal=scord.galactic
                    theta,phi=np.pi/2-gal.b/u.rad, gal.l/u.rad
                    #haslam not nest
                    px=hp.ang2pix(512,theta.value,phi.value,nest=False)
                    caches["haslam"][(filename,i)]=haslam[px]/10.
                var.append(caches["haslam"][(filename,i)])


            vars.append(var)
            totvars.append(var)
        cat.append((filename, np.array(vars)))
    return cat, totvars
        

def getFlist(o):
    if (rank==0):
        globst=o.dirname+"/*/*.fits"
        print "Preparing file list from:",globst
        filelist=glob(globst)
    else:
        filelist=[]
    filelist=comm.bcast(filelist,root=0)

    Nf=len(filelist)
    Nfp=Nf/o.skipfact
    if (rank==0):
        print "Done, we have ",Nf,"files, will use", Nfp,"."
    mystart=int(Nfp*rank*1.0/size)
    myend=int(Nfp*(rank+1)*1.0/size)

    if (Nf==0):
        if rank==0:
            print "No files, quitting."
        sys.exit(1)
    return filelist,Nf,Nfp, mystart, myend


def getVnames(o):
    vnames=[]
    vnames=['az'+str(i) for i in range(o.Naz)]
    if (o.weather):
        #for n in ['dusta','dustb','seeing']:
        #dusta,dustb seemingly not everywhere
        for n in ['seeing']:
            vnames.append(n)
    if (o.platepos):
        vnames.append("platepos")
        
    if (o.sunspots):
        vnames.append("sunspots")
    if (o.chips):
        vnames.append("camera")
        vnames.append("specedge")


    caches={}
    for name in cnlist:
    ## cached moon
        if (getattr(o,name)):
            if name=="galactic":
                vnames.append("gal1")
                vnames.append("gal5")
                vnames.append("gal10")
                vnames.append("gal20")
            elif name=="ecliptic":
                vnames.append("ecl1")
                vnames.append("ecl5")
                vnames.append("ecl10")
                vnames.append("ecl20")
            else:
                vnames.append(name)
            if (getattr(o,name)==1):
                caches[name]=cPickle.load(open(name+".cache"))
            else:
                caches[name]={}
    return vnames, caches

def saveCaches(o, caches):
    for name in cnlist:
        if (getattr(o,name)==2):
            ## moons below can by anything
            allmoons=comm.gather(caches[name],root=0)
            if (rank==0):
                for imoon in allmoons[1:]:
                    caches[name].update(imoon)
                print "Writing ",name+".cache"
                cPickle.dump(caches[name],open(name+".cache",'w'),-1)

def saveCatalog(o,vnames,cat,totvars):
    allcats=comm.gather(cat,root=0)
    alltotvars=comm.gather(totvars,root=0)
    if rank==0:
    # adding cats from nodes 1 onward to root cat
        for subcat in allcats[1:]:
            cat+=subcat
        for subtotvars in alltotvars[1:]:
            totvars+=subtotvars
        totvars=np.array(totvars)
        minv=totvars.min(axis=0)
        maxv=totvars.max(axis=0)
        meanv=totvars.mean(axis=0)
        medv=np.median(totvars,axis=0)

        for i,mn,mx,mean,med in zip(range(len(minv)),minv,maxv,meanv,medv):
            print "Variable %i %s : min, max, mean, median: %f %f %f %f"%(i,
              vnames[i],mn,mx,mean,med)

        cPickle.dump((cat,vnames, (minv,maxv,meanv,medv)),open(o.outcat,'w'))

                
if __name__ == "__main__":
    main()


