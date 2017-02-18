#!/usr/bin/env python
from glob import glob
import numpy as np
import pyfits
from optparse import OptionParser
from sklearn.decomposition import FastICA, PCA
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


def main():
    o = options() 
    setMPI(o)
    cov=load(o)
    if rank==0:
        analyze(o,cov)
    
def options():
    parser = OptionParser()
    ll_start=3.5
    ll_end=4.1
    ll_step=3e-4
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

    parser.add_option("--outroot", dest="outroot", default="pca",
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
    Np=int((o.ll_end-o.ll_start)/o.ll_step+1)
    nskip=0
    cov=np.zeros((Np,Np))
    covw=np.zeros((Np,Np))
    Nsp=0
    for i,filename in enumerate(filelist):
        if (i%size!=rank):
            continue
        cc+=1
        if rank==0:
            print "Done: %i/%i"%(cc,len(filelist)/size)

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
            for ext in [ext1,ext2]:
                cloglam=ext.data["loglam"]
                csky=ext.data["sky"]+ext.data["flux"]
                civar=ext.data["ivar"]
                ndx=np.array([int(v) for v in ((cloglam-o.ll_start)/o.ll_step+0.5)])
                vec[ndx]+=csky
                ivar[ndx]+=civar
            #w=np.ones(len(ivar))
            #w[np.where(ivar<1)]=0.0
            #veci=vec*w
            #veci[np.where(w==0)]=0.0
            #veci=vec
            vec[np.where(ivar<0.5)]=0.0
            cov+=np.outer(vec,vec)
            Nsp+=1
            #covw+=np.outer(w,w)
    print "Skipped",nskip
    if MPI:
        cov=comm.allreduce(cov,op=MPI.SUM)
        Nsp=comm.allreduce(Nsp,op=MPI.SUM)

    cov/=Nsp
    #cov[np.where(np.isnan(cov))]=0.0
    #for i in range(Np):
    #    if cov[i,i]==0:
    #        cov[i,i]=1e-3
    return cov

def analyze(o,cov):
    print "renorm"
    print "Doing PCA my way"
    evl,evc=la.eig(cov)
    np.savez(o.outroot+"_evl",evl)
    np.savez(o.outroot+"_evc",evc)

    plt.figure(figsize=(10,10))
    cd=np.sqrt(cov.diagonal())
    print np.where(cd<0)
    ncov=cov/np.outer(cd,cd)
    plt.imshow(ncov)
    plt.colorbar()
    plt.savefig('2cov.pdf')
    #plt.show()


    plt.figure(figsize=(10,10))
    plt.plot(np.cumsum(evl))
    plt.semilogx()
    plt.savefig("2cumsum.pdf")
    #plt.show()
    plt.figure(figsize=(10,10))
    plx=6000
    for i in range(o.Nc):
        plt.subplot(o.Nc,1,i+1)
        plt.plot(evc[:,i][:plx])
    plt.savefig("2comp.pdf")
    #plt.show()
    
if __name__ == "__main__":
    main()
