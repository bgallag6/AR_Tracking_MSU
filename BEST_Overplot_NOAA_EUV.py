# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 12:24:42 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
import jdcal
import csv
from astropy.time import Time
import datetime
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#"""
plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

#frm_bins = [30*i for i in range(13)]   
#long_bins = [60*i for i in range(7)]

int_thresh = 24

## EUV
#abs_dates = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/EUV_Absolute_Dates_%ithresh.npy' % int_thresh)
#abs_ARs = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/EUV_Absolute_ARs_%ithresh.npy' % int_thresh)
abs_dates = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/EUV_Absolute_Dates_%ithresh_revised.npy' % int_thresh)
abs_ARs = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/EUV_Absolute_ARs_%ithresh_revised.npy' % int_thresh)

latN_tot = []
latS_tot = []
lonN_tot = []
lonS_tot = []
intN_tot = []
intS_tot = []
frmN_tot = []
frmS_tot = []

for i in range(abs_ARs.shape[0]):   
    longitude = abs_ARs[i,0,:]
    latitude = abs_ARs[i,1,:]
    intensity = abs_ARs[i,2,:]
    frames = abs_ARs[i,3,:]
    longitude = longitude[intensity != 0]
    latitude = latitude[intensity != 0]
    frames = frames[intensity != 0]
    intensity = intensity[intensity != 0]
    lonN = longitude[latitude > 0]
    latN = latitude[latitude > 0]
    intN = intensity[latitude > 0]
    frmN = frames[latitude > 0]
    lonS = longitude[latitude < 0]
    latS = latitude[latitude < 0]
    intS = intensity[latitude < 0]
    frmS = frames[latitude < 0]
    
    latN_tot = np.append(latN_tot, latN)
    latS_tot = np.append(latS_tot, latS)
    lonN_tot = np.append(lonN_tot, lonN)
    lonS_tot = np.append(lonS_tot, lonS)
    intN_tot = np.append(intN_tot, intN)
    intS_tot = np.append(intS_tot, intS)
    frmN_tot = np.append(frmN_tot, frmN)
    frmS_tot = np.append(frmS_tot, frmS)
#"""
    
    
#"""    
## NOAA
abs_dates = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/NOAA_Absolute_Dates.npy')
abs_ARs = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/NOAA_Absolute_ARs.npy')

longitude = abs_ARs[0]
latitude = abs_ARs[1]
intensity = abs_ARs[2]
frames = abs_ARs[3]
    
lonN_NOAA = longitude[latitude > 0]
frmN_NOAA = frames[latitude > 0]
latN_NOAA = latitude[latitude > 0]

lonS_NOAA = longitude[latitude < 0]
frmS_NOAA = frames[latitude < 0]
latS_NOAA = latitude[latitude < 0]
#"""


#"""
## HMI
abs_ARs = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/HMI_Absolute_ARs.npy')

longitude = abs_ARs[0]
latitude = abs_ARs[1]
intensity = abs_ARs[2]
frames = abs_ARs[3]
    
lonN_HMI = longitude[latitude > 0]
frmN_HMI = frames[latitude > 0]
latN_HMI = latitude[latitude > 0]

lonS_HMI = longitude[latitude < 0]
frmS_HMI = frames[latitude < 0]
latS_HMI = latitude[latitude < 0]
#"""


### Plotting
#hemi = "North"
hemi = "South"
frm_indices = ['2010/05/19', '2011/01/01', '2012/01/01', '2013/01/01', '2014/01/01', '2014/05/31']
frm_ticks = [0, 227, 592, 958, 1323, 1473]

plt.figure(figsize=(40,5))
#plt.figure(figsize=(20,10))
#fig = plt.figure(figsize=(10,20))
ax1 = plt.gca()

#plt.xticks(frm_ticks, frm_indices)
ax1.set_title('EUV Plage Regions: Centroid Longitude vs Time \n %sern Hemisphere | %i Threshold' % (hemi, int_thresh), y=1.01, fontsize=font_size)
#ax1.set_title('EUV Plage Regions: Uncorrected Longitudes \n %sern Hemisphere | %i Threshold' % (hemi, int_thresh), y=1.01, fontsize=font_size)
#ax1.set_ylabel('Date', fontsize=font_size)
ax1.set_ylabel('Time [Days]', fontsize=font_size)
ax1.set_xlabel('Longitude [Deg]', fontsize=font_size)

"""
if hemi == "North":
    #plt.scatter(frmN_tot, lonN_tot)
    plt.scatter(frmN_tot, lonN_tot, intN_tot)
    plt.scatter(frmN_NOAA, lonN_NOAA, color='orange')
    plt.scatter(frmN_HMI, lonN_HMI, color='red')
elif hemi == "South":
    #plt.scatter(frmS_tot, lonS_tot)
    plt.scatter(frmS_tot, lonS_tot, intS_tot)
    plt.scatter(frmS_NOAA, lonS_NOAA, color='orange')
    plt.scatter(frmS_HMI, lonS_HMI, color='red')
    
plt.ylim(0,360)
plt.xlim(592,958)
"""


cmap = cm.get_cmap('jet_r', 10) 

if hemi == "North":
    #plt.scatter(frmN_tot, lonN_tot, label='EUV')
    #plt.scatter(frmN_tot, lonN_tot, intN_tot)
    #im = ax1.scatter(lonN_tot, frmN_tot, intN_tot, c=latN_tot, cmap=cmap, vmin=5, vmax=30)
    im = ax1.scatter(frmN_tot, lonN_tot, intN_tot, c=latN_tot, cmap=cmap, vmin=5, vmax=30)
    #plt.scatter(frmN_NOAA, lonN_NOAA, color='red', label='NOAA')
    #plt.scatter(lonN_HMI, frmN_HMI, color='red')
elif hemi == "South":
    #plt.scatter(frmS_tot, lonS_tot, label='EUV')
    #plt.scatter(frmS_tot, lonS_tot, intS_tot)
    #im = ax1.scatter(lonS_tot, frmS_tot, intS_tot, c=latS_tot, cmap=cmap, vmin=-30, vmax=-5)
    im = ax1.scatter(frmS_tot, lonS_tot, intS_tot, c=latS_tot, cmap=cmap, vmin=-30, vmax=-5)
    #plt.scatter(frmS_NOAA, lonS_NOAA, color='red', label='NOAA')
    #plt.scatter(lonS_HMI, frmS_HMI, color='red')
#plt.xticks(frm_ticks, frm_indices, fontsize=font_size)
#plt.yticks([60*i for i in range(7)], fontsize=font_size)
#plt.legend(fontsize=font_size)
 
plt.colorbar(im, pad=0.02, aspect=30)

#ax1.set_xlim(0,360)
#ax1.set_ylim(1473,0)
ax1.set_xlim(0,1473)
#ax1.set_xlim(958,1323)
ax1.set_ylim(0,360)

#plt.xlim(592,958)
#plt.ylim(0,360)
#"""

"""
ax1.set_title(r'NOAA Southern Hemisphere' + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)     
ax1.set_ylabel('Longitude', fontsize=font_size)
ax1.set_xlabel('Day', fontsize=font_size)  
ax1.scatter(frmS_tot, xS_tot,color='blue', label='Our Data') 
#ax1.scatter(frmS_tot, xS_tot, intS_tot, color='blue', label='Our Data')
ax1.scatter(datesS, xS,color='orange', label='NOAA Data') 
plt.xticks(frm_bins, fontsize=tick_size)
plt.yticks(long_bins, fontsize=tick_size)
"""

#plt.savefig('C:/Users/Brendan/Desktop/EUV_%s_Full_Vertical_%ithresh_D.pdf' % (hemi, int_thresh), bbox_inches='tight')
#plt.savefig('C:/Users/Brendan/Desktop/EUV_%s_Full_Horizontal_%ithresh.pdf' % (hemi, int_thresh), bbox_inches='tight')