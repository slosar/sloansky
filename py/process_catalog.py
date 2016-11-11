#!/usr/bin/env python
from glob import glob
import numpy as np
import pyfits,sys
from optparse import OptionParser

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
#
## add loglam options here
(o, args) = parser.parse_args()
if (len(args)!=1):
    print "Specify catalog name as option."
    sys.exit(1)
    

cat,(mn,mx,mean,med)=cPickle.load(open(args[0]))

Np=(ll_end-ll_start)/ll_step
meansky=np.zeros(Np)
Nc=len(mn)
contr=[np.zeros(Np) for i in range(Nc)]
loglam=loglam0+loglamstep*np.arange(Np)

for iter in range(Nit):
    for varcount in range(-1,Nc):
        nsp=0
        current=np.zeros(Np)
        currentw=np.zeros(Np)
        for filename,vars in cat:
            da=pyfits.open(filename)
            Ne=len(da)-4
            assert(Ne==len(vars))
            for j,ext in da[4:]:
                cloglam=ext.data["loglam"]
                csky=ext.data["sky"]+ext.data["flux"]
                civar=ext.data["ivar"]
                ndx=np.array([int(v) for v in ((cloglam-loglam0)/loglamstep+0.5)])
                ## values of variables
                var=vars[j,:]-med
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
        current/=currentw
        current[np.where(current>10)]=10.0
        current[np.where(current<0)]=0.0
        if (varcount==-1):
            meansky=current*1.0
            f=open("meansky%i.txt"%(iter),"w")
            for l,w,v in zip(loglam,currentw,meansky):
                print l,w,v
                if w>0:
                    f.write("%g %g %g \n"%(l,w,v))
            f.close()

        else:
            contr[varcount]=current*1.0
            f=open("componet%i_%i.txt"%(varcount,iter),"w")
            for l,w,v in zip(loglam,currentw,current):
                if w>0:
                    f.write("%g %g %g %g  \n"%(l,w,v))
            f.close()
    print "Finished iteration/varcount",iter,varcount
        
    

