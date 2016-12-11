#!/usr/bin/env python
from glob import glob
import numpy as np
import pyfits
import sys, cPickle
import matplotlib.pyplot as plt
from optparse import OptionParser
from mpi4py import MPI
import moon_angle_var as mav
import time
from astropy.coordinates import SkyCoord,  ICRS, BarycentricTrueEcliptic
import astropy.units as u
import healpy as hp
# don't worry about iers
#from astropy.utils import iers
#iers.conf.auto_download = False  
#iers.conf.auto_max_age = None  

cnlist=["moon", "ecliptic","galactic", "SFD", "halpha", "haslam"]


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

    if o.SFD==2:
        if rank==0:
            print "Reading SFD map"
            SFD=pyfits.open("data/lambda_sfd_ebv.fits")[1].data["TEMPERATURE"]
        else:
            SFD=None
        SFD=comm.bcast(SFD,root=0)
    if o.halpha==2:
        if rank==0:
            print "Reading halpha map"
            halpha=pyfits.open("data/lambda_halpha_fwhm06_0512.fits")[1].data["TEMPERATURE"]
        else:
            halpha=None
        halpha=comm.bcast(halpha,root=0)
    if o.haslam==2:
        if rank==0:
            print "Reading haslam map"
            haslam=pyfits.open("data/haslam408_dsds_Remazeilles2014.fits")[1].data["TEMPERATURE"]
            haslam=haslam.reshape((12*512**2,))
        else:
            haslam=None
        haslam=comm.bcast(haslam,root=0)

    return SFD,halpha,haslam

def getVars(o,filelist,caches,mystart,myend,Nfp):
    cat=[]
    totvars=[]
    SFD,halpha,haslam=getMaps(o)
    for ifile,filename in enumerate(filelist[mystart*o.skipfact:myend*o.skipfact:o.skipfact]):
        if rank==0:
            print "Doing: %i/%i"%(ifile*size, Nfp)
        da=pyfits.open(filename)
        vars=[]
        for i,ext in enumerate(da[4:]):
            var=[]
            ## values of variables
            alt,az,ra,dec=[ext.header[x]/180.*np.pi for x in ["ALT","AZ","RA","DEC"]]
            ## add azimuth vars
            azstep=2*np.pi/12.
            for ai in range(12):
                taz=azstep*(ai+0.5)
                daz=abs(az-taz)
                if (daz>np.pi):
                    daz-=np.pi
                if (daz<azstep/2):
                    v=(1.-daz/(azstep/2))*np.cos(alt)
                else:
                    v=0
                var.append(v)

            if o.moon:
                if (o.moon==2):
                    caches["moon"][(filename,i)]=mav.moon_angle_var(da,ext)
                var.append(caches["moon"][(filename,i)])

            scord=None
            if o.ecliptic:
                if (o.ecliptic==2):
                    scord=SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
                    ecl=scord.geocentrictrueecliptic
                    caches["ecliptic"][(filename,i)]=np.exp(-(ecl.lat**2/(2*(10.*u.deg)**2)))
                var.append(caches["ecliptic"][(filename,i)])

            if o.galactic:
                if (o.galactic==2):
                    if scord is None:
                        scord=SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
                    gal=scord.galactic
                    caches["galactic"][(filename,i)]=np.exp(-(gal.b**2/(2*(10.*u.deg)**2)))
                var.append(caches["galactic"][(filename,i)])
            
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
    vnames=['az'+str(i) for i in range(12)]
    caches={}
    for name in cnlist:
    ## cached moon
        if (getattr(o,name)):
            vnames.append(name)
            if (getattr(o,name)==1):
                caches[name]=cPickle.load(open(name+".cache"))
            else:
                caches[name]={}
    return vnames, caches

def saveCaches(o, caches):
    for name in cnlist:
        print name, getattr(o,name)
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


