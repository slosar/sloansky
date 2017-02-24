#!/usr/bin/env python
from glob import glob
import numpy as np
import pyfits
from optparse import OptionParser
from sklearn.decomposition import FastICA, PCA
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.linalg as la

try:
    from mpi4py import MPI
except:
    MPI=None
# need a better solution
import warnings
import sys

warnings.filterwarnings("ignore")
#lines=[(5875,5890),(6131,6153)]

def main():
    o = options() 
    setMPI(o)
    dlist=load(o)
    analyze(o,dlist)
    
def options():
    parser = OptionParser()
    ll_start=3.5
    ll_end=4.1
    ll_step=3e-4# 1e-4
    dirname="sky_data/"
    skipfact=10
    Nc=5
    parser.add_option("-D","--dirname", dest="dirname", default=dirname,
                      help="Root directory of data files", type="string")
    parser.add_option("--skip", dest="skipfact", default=skipfact,
                      help="skip factor", type="int")
    parser.add_option("--ll_start", dest="ll_start", default=ll_start,
                      help="loglam start", type="float")
    parser.add_option("--ll_end", dest="ll_end", default=ll_end,
                      help="loglam end", type="float")
    parser.add_option("--ll_step", dest="ll_step", default=ll_step,
                      help="loglam step", type="float")
    parser.add_option("--Nc", dest="Nc", default=Nc,
                      help="Numer of components", type="int")

    parser.add_option("--outroot", dest="outroot", default="ica",
                      help="Output root", type="string")

    #
    ## add loglam options here
    o, args=parser.parse_args()
    return o


def setMPI(o):
    global comm, rank,size
    if MPI is not None:
        comm=MPI.COMM_WORLD
        rank=comm.Get_rank()
        size=comm.Get_size()
        if (rank==0):
            print "We have ",size,"nodes."
    else:
        comm=None
        rank=0
        size=1
        
def load(o):
    if (rank==0):
        globst=o.dirname+"/*/*.fits"
        print "Preparing file list from:",globst
        filelist=glob(globst)[::o.skipfact]
    else:
        filelist=[]

    if MPI:
        filelist=comm.bcast(filelist,root=0)
    cc=0
    Np=(o.ll_end-o.ll_start)/o.ll_step+1
    ndxl=np.arange(Np-1)
    nskip=0
    dlist=[]
    for i,filename in enumerate(filelist):
        if (i%size!=rank):
            continue
        cc+=1
        if rank==0:
            print "Done: %i/%i"%(cc,len(filelist))

        da=pyfits.open(filename)
        No=(len(da)-4)/2
        for j in range (No):
            ext1=da[4+j]
            ext2=da[4+No+j]
            if ((ext1.header['TAI-BEG']!=ext2.header['TAI-BEG'])):
                print "skipping"
                nskip+=1
                break
            vec=np.zeros(Np)
            ivar=np.zeros(Np)
            ndx1=1e5
            ndx2=0
            for ext in [ext1,ext2]:
                cloglam=ext.data["loglam"]
                csky=ext.data["sky"]+ext.data["flux"]
                civar=ext.data["ivar"]
                wh=np.where(cloglam<o.ll_end)
                cloglam=cloglam[wh]
                csky=csky[wh]
                civar=civar[wh]

                clam=10**cloglam
                #for l1,l2 in lines:
                #    wh=np.where((clam>l1)&(clam<l2))
                #    csky[wh]=0.0
                #    civar[wh]=0.0
                ndx=np.array([int(v) for v in ((cloglam-o.ll_start)/o.ll_step+0.5)])
                if len(ndx)==0:
                    continue
                vec[ndx]+=csky
                ivar[ndx]+=civar
                ndx1=min(ndx1,ndx[0])
                ndx2=max(ndx2,ndx[-1])
            wh=np.where((ivar==0) & (ndxl>ndx1) & (ndxl<ndx2))
            whp=np.array(wh)+1
            whm=np.array(wh)-1
            vec[wh]=0.5*(ivar[whp]*vec[whp]+ivar[whm]*vec[whm])
            ivar[wh]=0.5*(ivar[whp]+ivar[whm])
            vec[wh]/=ivar[wh]
            vec[np.where(np.isnan(vec))]=0.0
            vec[np.where(ivar<0.5)]=0.0
            dlist.append(vec)

    print "Skipped",nskip
    return np.array(dlist)

def gnorm(v):
    if -v.min()>v.max():
        return v.min()
    else:
        return v.max()

def mplot(lam, v):
    tp=v/gnorm(v)
    plt.plot(lam,v,'b-')
    #plt.plot(lam,-v,'r')
def analyze(o,dlist):
    print "taking off best fit mean"
    lam=10**(np.arange(o.ll_start,o.ll_end,o.ll_step))
    
    Ns,Np=dlist.shape
    mspec=dlist.mean(axis=0)
    plt.plot(lam,mspec)
    plt.show()
    plt.plot(lam,dlist[8,:])

    mspec[np.where(mspec<1)]=0.
    coef=(dlist/np.outer(np.ones(Ns),mspec))
    coef[np.where(np.isnan(coef))]=0.0
    coef[np.where(np.isinf(coef))]=0.0
    coefx=coef.sum(axis=1)
    coef[np.where(coef!=0.0)]=1.0
    coefx/=coef.sum(axis=1)
    bfit=np.outer(coefx,mspec)
    dlist-=bfit
    plt.plot(lam,bfit[8,:],'r-')
    print bfit
    plt.show()
    plt.plot(coefx)
    plt.show()
    
    dlist[np.where(np.isnan(dlist))]=0.0
    #rnorm=np.sqrt(dlist.var(axis=0))
    rnorm=np.ones(len(dlist[0]))
    print len(rnorm)
    rnorm[np.where(rnorm==0)]=1.0
    for i in range(len(dlist)):
        dlist[i,:]/=rnorm
    # compute ICA
    print "Doing ICA..."
    ica = FastICA(n_components=o.Nc)
    S = ica.fit_transform(dlist.T)  # Get the estimated sources
    # compute PCA
    print "Doing PCA..."
    pca = PCA(n_components=o.Nc)
    H = pca.fit_transform(dlist.T)  # estimate PCA sources
    np.savez(o.outroot+"_ica",S)
    np.savez(o.outroot+"_pca",H)

    for ofi in range(o.Nc/5+1):
        plt.figure(figsize=(10,10))
        for i in range(5):
            plt.subplot(5,2,2*i+1)
            #plt.plot(np.log(rnorm[:plx]),'r-')
            mplot(lam,S[:,5*ofi+i])
            #plt.semilogy()
            plt.subplot(5,2,2*i+2)
            mplot(lam,H[:,5*ofi+i])
            #plt.semilogy()
        plt.show()


    
if __name__ == "__main__":
    main()
