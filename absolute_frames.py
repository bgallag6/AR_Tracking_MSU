# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:10:55 2017

@author: Brendan
"""

"""
##################################
### Convert IDL program output ###
###        to numpy array      ###
##################################
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
import jdcal
import csv
from astropy.time import Time
import datetime
import jdcal

#"""
plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

fmt = '%Y%m%d%H'

s = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)
f_names = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/Active_Longitude/ar_filenames.npy')
datesARs = []
greg_date = []
for i in range(len(f_names)):
#for i in range(19):
    date = f_names[i][0:8]+'%s' % f_names[i][9:11]
    greg_date = np.append(greg_date, date)
    dt = datetime.datetime.strptime(date[0:10],fmt)
    jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day)) + dt.hour/24.
    datesARs = np.append(datesARs, jul_date)


trim_start = 11  # start of 1st Carrington rotation
trim_end = 2796  # last index for end of Carrington rotation

datesARs0 = np.copy(datesARs)    
datesARs0 -= datesARs0[trim_start]
#datesARs0 *= 2

# make array of correct dates for EUV frames
dates_final = np.zeros((3,trim_end-trim_start))
dates_final[0] = datesARs0[trim_start:trim_end] # frames elapsed since 2010/05/19
dates_final[1] = datesARs[trim_start:trim_end] # julian date of frame
dates_final[2] = greg_date[trim_start:trim_end] # gregorian date of frame

# frames starting each year: 2010,2011,2012,2013,2014
fStart = [11,417,1115,1825,2511]
fEnd = [416,1114,1824,2510,2710]

# extract values from IDL file
coord = s.STRS.coordinates  # rectangular box in pixels
cen_coord = s.STRS.centroid_cord  # centroid in degrees
n_regions = s.STRS.n_region
med_inten = s.STRS.median_intensity
tot_int1 = s.STRS.tot_int1
tot_area1 = s.STRS.tot_area1

# convert IDL structures to python lists
all_coords = coord.tolist()
all_cen_coords = cen_coord.tolist()
n_regions = n_regions.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()

arr_total = np.zeros((trim_end-trim_start,4,50))
#arr_total = np.zeros((trim_end-trim_start,5,50))  #long, lat, int, date, width

int_thresh = 0

num_reg_rev = 0

# for each frame, extract the values for each detected plage and add to final array
for k in range(trim_start, trim_end):
    numAR = n_regions[k]
    xtemp = all_cen_coords[k][:numAR,0]
    ytemp = all_cen_coords[k][:numAR,1]
    inttemp = all_tot_int1[k][:numAR]
    xtemp = xtemp[inttemp > int_thresh]
    ytemp = ytemp[inttemp > int_thresh]
    
    #"""
    x1coords0 = np.array(all_coords[k])[:numAR,0] / 10
    x2coords0 = np.array(all_coords[k])[:numAR,1] / 10
    #x1_temp = x1coords0[ytemp != 0]
    #x2_temp = x2coords0[ytemp != 0]
    x1_temp = x1coords0[inttemp > int_thresh]
    x2_temp = x2coords0[inttemp > int_thresh]
    
    wid_temp = x2_temp - x1_temp
    #"""
    
    inttemp = inttemp[inttemp > int_thresh]
    
    #"""
    # screen out the bad detections
    rep_arr = np.array([t for t in range(len(inttemp))])

    inttemp = np.array([0 if xtemp[r] > 357 and wid_temp[r] < 4 else inttemp[r] for r in rep_arr])
    ytemp = np.array([0 if xtemp[r] > 357 and wid_temp[r] < 4 else ytemp[r] for r in rep_arr]) 
    xtemp = np.array([0 if xtemp[r] > 357 and wid_temp[r] < 4 else xtemp[r] for r in rep_arr]) 
    
    inttemp = np.array([0 if xtemp[r] < 2 and wid_temp[r] < 4 else inttemp[r] for r in rep_arr])
    ytemp = np.array([0 if xtemp[r] < 2 and wid_temp[r] < 4 else ytemp[r] for r in rep_arr]) 
    xtemp = np.array([0 if xtemp[r] < 2 and wid_temp[r] < 4 else xtemp[r] for r in rep_arr]) 
    
    inttemp = inttemp[inttemp !=0]
    ytemp = ytemp[ytemp !=0]
    xtemp = xtemp[xtemp !=0]
    #"""
    
    num_reg_rev += len(inttemp)
    
    arr_total[k-trim_start,0,:len(xtemp)] = xtemp
    arr_total[k-trim_start,1,:len(xtemp)] = ytemp
    arr_total[k-trim_start,2,:len(xtemp)] = inttemp
    arr_total[k-trim_start,3,:len(xtemp)] = [datesARs0[k] for j in range(len(inttemp))]
 
tot_AR = np.sum(n_regions[trim_start:trim_end])
print tot_AR



#np.save('C:/Users/Brendan/Desktop/Inbox/EUV_Absolute_ARs_0thresh_revised.npy', arr_total)
#np.save('C:/Users/Brendan/Desktop/Inbox/EUV_Absolute_Dates_0thresh_revised.npy', dates_final)