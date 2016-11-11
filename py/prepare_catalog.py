#!/usr/bin/env python
from glob import glob
import numpy as np
import pyfits
import sys
import matplotlib.pyplot as plt
from optparse import OptionParser

dirname="data/spectra_sky"
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

print "Preparing file list..."
filelist=glob(o.dirname+"/*/*.fits")
Nf=len(filelist)
print "Done, we have ",Nf,"files, will use", Nf/o.skipfact,"."

if (Nf==0):
    print "No files, quitting."
    sys.exit(1)
    
cat=[]
totvars=[]
for ifile,filename in enumerate(filelist[::o.skipfact]):
    print filename
    da=pyfits.open(filename)
    vars=[]
    for ext in da[4:]:
        ## values of variables
        var=[np.cos(ext.header["ALT"]/180.*np.pi),
             np.sin(ext.header["AZ"]/180.*np.pi),
             np.cos(ext.header["AZ"]/180.*np.pi)]
        vars.append(var)
        totvars.append(var)
    cat.append((filename, np.arrays(vars)))

totvars=np.array(totvars)
minv=totvars.min(axis=0)
maxv=totvars.max(axis=0)
meanv=totvars.mean(axis,0)
med=np.median(totvars,axis=0)

for i,mn,mx,mean,med in zip(np.range(minv),minv,maxv,meanv,med):
    print "Variable i: min, max, mean, median: %f %f %f %f"%(
         mn,mx,mean,med)

cPickle.dump((cat,(mn,mx,mean,med)),open(o.outcat,'w'))



