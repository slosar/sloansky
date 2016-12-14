from datetime import datetime
from astropy.time import Time
import glob
from mpi4py import MPI


## Enter solar files (the range of years)
class SolarActivity:
        def __init__ (self, comm):
            master={}
            if comm.Get_rank()==0:
                for fn in glob.glob('data/g*'):
                    print "Reading",fn
                    ## We only need year, month, day of month, and mill. of sol. disk
                    ## Cols         1-4    5-6      7-8               31-34
                    for y,m,d,v in [(int(x[:4]), int(x[4:6]), int(x[6:8]), float(x[31:34])) for x in open(fn).readlines()]:
                        mjd=int(Time(datetime(y,m,d,0)).mjd)
                        if master.has_key(mjd):
                            master[mjd]+=v
                        else:
                            master[mjd]=v
            comm.bcast(master,root=0)
            self.dic=master

        def activity(self,mjd):
            imjd=int(mjd)
            if self.dic.has_key(imjd):
                return self.dic[imjd]
            else:
                return 0.0

