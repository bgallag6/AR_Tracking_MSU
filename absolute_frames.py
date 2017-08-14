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

fmt = '%Y%m%d%H'

s = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)
f_names = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/ar_filenames.npy')
datesARs = []
greg_date = []
for i in range(len(f_names)):
#for i in range(19):
    date = f_names[i][0:8]+'%s' % f_names[i][9:11]
    greg_date = np.append(greg_date, date)
    dt = datetime.datetime.strptime(date[0:10],fmt)
    jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day)) + dt.hour/24.
    datesARs = np.append(datesARs, jul_date)


#trim = 2922  # image before jump 20140818-20151103
trim_start = 11
trim_end = 2796  # last index for end of Carrington rotation

datesARs0 = np.copy(datesARs)    
datesARs0 -= datesARs0[trim_start]
#datesARs0 *= 2

dates_final = np.zeros((3,trim_end-trim_start))
dates_final[0] = datesARs0[trim_start:trim_end]
dates_final[1] = datesARs[trim_start:trim_end]
dates_final[2] = greg_date[trim_start:trim_end]

fStart = [11,417,1115,1825,2511]
fEnd = [416,1114,1824,2510,2710]

coord = s.STRS.coordinates  # rectangular box in pixels
cen_coord = s.STRS.centroid_cord  # centroid in degrees
n_regions = s.STRS.n_region
med_inten = s.STRS.median_intensity
tot_int1 = s.STRS.tot_int1
tot_area1 = s.STRS.tot_area1

all_cen_coords = cen_coord.tolist()
n_regions = n_regions.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()

arr_total = np.zeros((trim_end-trim_start,4,50))

int_thresh = 30

for k in range(trim_start, trim_end):
    numAR = n_regions[k]
    xtemp = all_cen_coords[k][:numAR,0]
    ytemp = all_cen_coords[k][:numAR,1]
    inttemp = all_tot_int1[k][:numAR]
    xtemp = xtemp[inttemp > int_thresh]
    ytemp = ytemp[inttemp > int_thresh]
    inttemp = inttemp[inttemp > int_thresh]
    arr_total[k-trim_start,0,:len(xtemp)] = xtemp
    arr_total[k-trim_start,1,:len(xtemp)] = ytemp
    arr_total[k-trim_start,2,:len(xtemp)] = inttemp
    arr_total[k-trim_start,3,:len(xtemp)] = [datesARs0[k] for j in range(len(inttemp))]

np.save('C:/Users/Brendan/Desktop/EUV_Absolute_ARs.npy', arr_total)
np.save('C:/Users/Brendan/Desktop/EUV_Absolute_Dates.npy', dates_final)

