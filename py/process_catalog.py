#!/usr/bin/env python
from glob import glob
import numpy as np
import pyfits,sys, cPickle
from optparse import OptionParser
from mpi4py import MPI
# need a better solution
import warnings
warnings.filterwarnings("ignore")


def main():
    o, catname = options() 
    setMPI(o)
    cat,vnames, (mn,mx,mean,med) = getNumbers(catname)
    analyze(o, cat,vnames,med)

def options():

    parser = OptionParser()
    ll_start=3.45
    ll_end=4.1
    ll_step=1e-4
    Nit=5
    parser.add_option("--ll_start", dest="ll_start", default=ll_start,
                      help="loglam start", type="float")
    parser.add_option("--ll_end", dest="ll_end", default=ll_end,
                      help="loglam end", type="float")
    parser.add_option("--ll_step", dest="ll_step", default=ll_step,
                      help="loglam step", type="float")
    parser.add_option("--Nit", dest="Nit", default=Nit,
                      help="Number of iterations", type="int")
    parser.add_option("--outroot", dest="outroot", default="output/",
                      help="Output root", type="string")

    #
    ## add loglam options here
    o, args=parser.parse_args()
    if (len(args)!=1):
        print "Specify catalog name as option."
        sys.exit(1)
    return o, args[0]


def setMPI(o):
    global comm, rank,size
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    size=comm.Get_size()

def getNumbers(catname):

    if (rank==0):
        cat,vnames,stat=cPickle.load(open(catname))
    else:
        stat=None
        cat=None
        vnames=None
    if (rank==0):
        print "Broadcasting..."
    cat=comm.bcast(cat,root=0)
    vnames=comm.bcast(vnames,root=0)
    stat=comm.bcast(stat,root=0)
    comm.Barrier()
    if (rank==0):
        print "Done..."
    return cat,vnames, stat
        

def analyze(o,cat,vnames,ofs):

    Nf=len(cat)
    mystart=int(Nf*rank*1.0/size)
    myend=int(Nf*(rank+1)*1.0/size)
    if (rank==size-1):
        myend=Nf
    Np=int((o.ll_end-o.ll_start)/o.ll_step+1)
    meansky=np.zeros(Np)
    Nc=len(ofs)
    contr=[np.zeros(Np) for i in range(Nc)]
    loglam=o.ll_start+o.ll_step*np.arange(Np)
    if (rank==0):
        print "Starting..."
    for iter in range(o.Nit):
        for varcount in range(-1,Nc):
            if (rank==0):
                print "Iteration, variable: ",iter, varcount, 
                if (varcount<0):
                    print " mean sky"
                else:
                    print vnames[varcount]
            nsp=0
            current=np.zeros(Np)
            currentw=np.zeros(Np)
            cc=0
            for filename,vars in cat[mystart:myend]:
                cc+=1
                if rank==0:
                    print "Done: %i/%i"%(cc,myend-mystart)
                da=pyfits.open(filename)
                Ne=len(da)-4
                assert(Ne==len(vars))
                for j,ext in enumerate(da[4:]):
                    cloglam=ext.data["loglam"]
                    csky=ext.data["sky"]+ext.data["flux"]
                    civar=ext.data["ivar"]
                    ndx=np.array([int(v) for v in ((cloglam-o.ll_start)/o.ll_step+0.5)])
                    ## values of variables
                    var=vars[j,:]-ofs
                    if (varcount==-1):
                        unwanted=np.zeros(Np)
                        for i in range(Nc):
                            unwanted+=contr[i]*var[i]
                    else:
                        unwanted=meansky*1.0
                        for i in range(Nc):
                            if (i!=varcount):
                                unwanted+=contr[i]*var[i]
                    w = 1 if varcount == -1 else var[varcount]
                    current[ndx]+=civar*w*(csky-unwanted[ndx])
                    currentw[ndx]+=civar*w**2
            current=comm.allreduce(current,op=MPI.SUM)
            currentw=comm.allreduce(currentw,op=MPI.SUM)

            current/=currentw
            current[np.where(current>20)]=20.0
            current[np.where(current<-20)]=-20.0
            if (varcount==-1):
                meansky=current*1.0
                if (rank==0):
                    f=open(o.outroot+"meansky%i.txt"%(iter),"w")
                    for l,w,v in zip(loglam,currentw,meansky):
                        if w>0:
                            f.write("%g %g %g\n"%(l,w,v))
                    f.close()

            else:
                contr[varcount]=current*1.0
                if (rank==0):
                    f=open(o.outroot+"component_%s_%i.txt"%(vnames[varcount],iter),"w")
                    for l,w,v in zip(loglam,currentw,current):
                        if w>0:
                            f.write("%g %g %g\n"%(l,w,v))
                    f.close()
        if (rank==0):
            print "Finished iteration/varcount",iter,varcount
        
if __name__ == "__main__":
    main()
