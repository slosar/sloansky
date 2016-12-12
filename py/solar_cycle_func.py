import pyfits
import pandas as pd
import numpy as np
from itertools import chain
from collections import Counter
from scipy.interpolate import interp1d
from datetime import datetime
from astropy.time import Time


#a) first load all files and convert all dates into mjds.
def get_mjd(file):
    da=pyfits.open(file)
    return (da[4].header['MJD'])


## Master List of Solar File MJDs & Coverage
def merge_subs(lst_of_lsts):
    res = []
    for row in lst_of_lsts:
        for i, resrow in enumerate(res):
            if row[:3]==resrow[:3]:
                res[i] += row[1:]
                break
        else:
            res.append(row)
    return (res)


## removes duplicates while retaining one
def f2(seq): 
    # order preserving
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked


## Enter solar files (the range of years)
def mjds_sol(txt_file, fits_files):
    master_list=[]
    for i in txt_file:
        ## We only need year, month, day of month, and mill. of sol. disk
        ## Cols         1-4    5-6      7-8               31-34
        demo=list(chain.from_iterable((x[:4], x[4:6], x[6:10], x[31:34]) for x in open(i).readlines()))
        ## Organizing cols
        str_list=[demo[i*4:i*4+4] for i in range(int(len(demo)/4))]
        ## Converting to float
        float_dates=[]
        for k in range(len(str_list)):
            float_dates.append([float(i) for i in str_list[k]])
        b=([x[:4] for x in float_dates])
        ve=merge_subs(b)
        new=[]
        summed_mills=[]
        no_mills=[]
        merged_lists=[]
        merged_utc=[]
        for i in range(len(ve)):
            new.append(f2(ve[i]))
            summed_mills.append(sum(new[i][3:]))
            no_mills.append(new[i][:3])
            merged_lists.append(np.append(no_mills[i],summed_mills[i]).tolist())
            ## Conversion to datetime ##
            merged_lists[i][3:3]=[12]
            merged_utc.append([int(n) for n in merged_lists[i]])
        utc_df=pd.DataFrame(merged_utc)
        utc_df.columns=['year','month','day','hour','mill']
        no_mills_mjds=[]
        vt=[]
        merged_mjds=[]
        ## Conversion to MJDs ##
        for i in range(len(utc_df)):
            vt.append(Time(datetime(utc_df['year'][i],utc_df['month'][i],utc_df['day'][i],utc_df['hour'][i]), scale='utc'))
            no_mills_mjds.append(vt[i].mjd)
            merged_mjds.append(np.append(no_mills_mjds[i],summed_mills[i]).tolist())
            master_list.append(merged_mjds[i]) ## lists all info from solar files
        ## preparing to interpolate
    master_df=pd.DataFrame(master_list)
    master_df.columns=['mjd','sun_cov']
    mjd=master_df['mjd']
    sun_cov=master_df['sun_cov']
    ifunc=interp1d(mjd,sun_cov)
    seq_sol=[ifunc(i).tolist() for i in [get_mjd(i) for i in fits_files]]
    return(seq_sol)

#rand_files=('spec-3586-55181-0496.fits','spec-3586-55181-0788.fits','spec-3586-55181-0996.fits',
#            'spec-3761-55272-0008.fits','spec-10000-57346-0334.fits','spec-3761-55272-0475.fits',
#            'spec-10000-57346-0659.fits','spec-5478-56014-0654.fits','spec-10000-57346-0955.fits',
#            'spec-5478-56014-0716.fits','spec-10000-57346-0994.fits')

#solar_files=('g2009.txt','g2010.txt','g2011.txt','g2012.txt',
#'g2013.txt','g2014.txt','g2015.txt','g2016.txt')

#print(mjds_sol(solar_files, rand_files))
