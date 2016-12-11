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

# don't worry about iers
#from astropy.utils import iers
#iers.conf.auto_download = False  
#iers.conf.auto_max_age = None  

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
parser.add_option("--moon", dest="moon", default=0,
                  help="Moon: 0=ignore, 1=use, 2=cache", type="int")



def main():
    #init
    (o, args) = parser.parse_args()
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


def getVars(o,filelist,caches,mystart,myend,Nfp):
    cat=[]
    totvars=[]
    for ifile,filename in enumerate(filelist[mystart*o.skipfact:myend*o.skipfact:o.skipfact]):
        if rank==0:
            print "Doing: %i/%i"%(ifile*size, Nfp)
        da=pyfits.open(filename)
        vars=[]
        for i,ext in enumerate(da[4:]):
            var=[]
            ## values of variables
            alt,az=[ext.header[x]/180.*np.pi for x in ["ALT","AZ"]]
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
    ## cached moon
    if (o.moon):
        vnames.append("moon")
        if (o.moon==1):
            caches["moon"]=cPickle.load(open("moon.cache"))
        else:
            caches["moon"]={}
    return vnames, caches

def saveCaches(o, caches):

    if (o.moon==2):
        allmoons=comm.gather(caches["moon"],root=0)
        if (rank==0):
            for imoon in allmoons[1:]:
                caches["moon"].update(imoon)
            cPickle.dump(caches["moon"],open('moon.cache','w'),-1)

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
            print "Variable %i: min, max, mean, median: %f %f %f %f"%(i,
                 mn,mx,mean,med)

        cPickle.dump((cat,vnames, (minv,maxv,meanv,medv)),open(o.outcat,'w'))

                
if __name__ == "__main__":
    main()


