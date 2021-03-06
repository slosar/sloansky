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
    if (rank==0):
        print "We have ",size,"nodes."
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
        fchi2=open(o.outroot+"chi2.txt",'w')
    for iter in range(o.Nit):
        if rank==0:
            todolist=[-1,-2]+list(np.random.permutation(Nc))
        else:
            todolist=None
        todolist=comm.bcast(todolist,root=0)
        for varcount in todolist:
            if (rank==0):
                print "Iteration, variable: ",iter, varcount, 
                if (varcount==-1):
                    print " mean sky"
                elif (varcount==-2):
                    print " chi2 calc"
                else:
                    print vnames[varcount]
            nsp=0
            current=np.zeros(Np)
            currentw=np.zeros(Np)
            cc=0
            chi2=0.
            dof=0
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
                    if (varcount<0):
                        unwanted=np.zeros(Np) if varcount==-1 else meansky*1.0
                        for i in range(Nc):
                            unwanted+=contr[i]*var[i]
                    else:
                        unwanted=meansky*1.0
                        for i in range(Nc):
                            if (i!=varcount):
                                unwanted+=contr[i]*var[i]
                    toadd=(csky-unwanted[ndx])
                    whnan=np.where(np.isnan(toadd))
                    if len(whnan[0])>0:
                        print whnan,np.any(np.isnan(csky)), np.any(np.isnan(unwanted[ndx]))
                    toadd[whnan]=0.0
                    civar[whnan]=0.0
                    if (varcount==-2):
                        chi2+=((toadd-meansky[ndx])**2*civar).sum()
                        dof+=(civar>0).sum()
                    else:
                        w = 1 if varcount == -1 else var[varcount]
                        current[ndx]+=civar*w*toadd
                        currentw[ndx]+=civar*w**2
            if (varcount==-2):
                chi2=comm.allreduce(chi2,op=MPI.SUM)
                dof=comm.allreduce(dof,op=MPI.SUM)
                if (rank==0):
                    fchi2.write("%g %i \n"%(chi2,dof))
                    fchi2.flush()
            else:    
                current=comm.allreduce(current,op=MPI.SUM)
                currentw=comm.allreduce(currentw,op=MPI.SUM)
                wh=np.where(currentw>0)
                current[wh]/=currentw[wh]
                current[np.where(current>20)]=20.0
                current[np.where(current<-20)]=-20.0
                if (varcount==-1):
                    meansky=current*1.0
                    if (rank==0):
                        f=open(o.outroot+"meansky_%i.txt"%(iter),"w")
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
