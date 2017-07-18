# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:10:55 2017

@author: Brendan
"""

"""
###############################
### Comparison NOAA AR Data ###
### and our data  #############
###############################
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
import jdcal
import csv
import urllib2
import urllib
from astropy.time import Time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
import jdcal

#"""
plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

fmt = '%Y%m%d'

s = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)
f_names = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/ar_filenames.npy')
datesARs = []
for i in range(len(f_names)):
    date = f_names[i][0:8]
    dt = datetime.datetime.strptime(date[0:8],fmt)
    jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
    datesARs = np.append(datesARs, jul_date)
    
datesARs -= datesARs[0]
datesARs *= 2

date_num_check = 0
add_half = 0

for i in range(1,len(datesARs)):
    if datesARs[i] == datesARs[i-1]:
        datesARs[i] = datesARs[i-1]+1
    

#trim = 2922  # image before jump 20140818-20151103
trim = 2872  # last index for end of Carrington rotation

fStart = [11,417,1115,1825,2511]
fEnd = [416,1114,1824,2510,2710]

coord = s.STRS.coordinates  # rectangular box in pixels
cen_coord = s.STRS.centroid_cord  # centroid in degrees
n_regions = s.STRS.n_region
med_inten = s.STRS.median_intensity
tot_int1 = s.STRS.tot_int1
tot_area1 = s.STRS.tot_area1

all_cen_coords = cen_coord.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()

int_thresh = 0

