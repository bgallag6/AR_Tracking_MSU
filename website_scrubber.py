# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 13:51:23 2017

@author: Brendan
"""
import numpy as np
import urllib

# gets all files, regardless of whether they are files.  Maybe ok - go through and delete filesizes under 1 MB?
# or see if there is a reject if not found option

# create array needed includes all dates, and all variations of times: 000615/001615 + 121615/120615 + all months with 31 days 
# 20100513 - 20160514

hms1 = '000615'
hms2 = '001615'
hms3 = '120615'
hms4 = '121615'

arr_need = []

#for year in [2010,2011,2012,2013,2014,2015,2016]:
    #if year == 2010:
    #    month_list = [i for i in range(5,13)]
    #else:
    #    month_list = [i for i in range(1,13)]
for year in [2011]:
    #for month in month_list:
    for month in range(2,3):
        #for day in range(1,32):
        for day in range(9,16):
            t1 = '%i%0.2i%0.2i_%s_%i' % (year,month,day,hms1, 304)
            arr_need = np.append(arr_need, t1)
            t2 = '%i%0.2i%0.2i_%s_%i' % (year,month,day,hms2, 304)
            arr_need = np.append(arr_need, t2)
            t3 = '%i%0.2i%0.2i_%s_%i' % (year,month,day,hms3, 304)
            arr_need = np.append(arr_need, t3)
            t4 = '%i%0.2i%0.2i_%s_%i' % (year,month,day,hms4, 304)
            arr_need = np.append(arr_need, t4)
            
for fn in arr_need:    
    urllib.urlretrieve("https://solarmuse.jpl.nasa.gov/data/euvisdo_maps_carrington_12hr/304fits/%s.fts" % fn, "%s.fts" % fn)