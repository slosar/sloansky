#!/usr/bin/env python
from glob import glob
import numpy as np
import pyfits
import sys, cPickle
import matplotlib.pyplot as plt
from optparse import OptionParser
from mpi4py import MPI
import moon_angle_var as mav

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
## add loglam options here

(o, args) = parser.parse_args()
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
if rank==0:
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
    
cat=[]
totvars=[]
for ifile,filename in enumerate(filelist[mystart*o.skipfact:myend*o.skipfact:o.skipfact]):
    sys.stdout.flush()
    da=pyfits.open(filename)
    vars=[]
    for i,ext in enumerate(da[4:]):
        ## values of variables
        var=[np.cos(ext.header["ALT"]/180.*np.pi),
             np.sin(ext.header["AZ"]/180.*np.pi),
             np.cos(ext.header["AZ"]/180.*np.pi),
             mav.moon_angle_var(da,ext)]
        vars.append(var)
        totvars.append(var)
    cat.append((filename, np.array(vars)))
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

    cPickle.dump((cat,(minv,maxv,meanv,medv)),open(o.outcat,'w'))



