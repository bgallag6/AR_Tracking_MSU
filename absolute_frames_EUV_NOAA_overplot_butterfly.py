# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 21:56:47 2017

@author: Brendan
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

## EUV
#abs_dates = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/EUV_Absolute_Dates.npy')
#abs_ARs = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/EUV_Absolute_ARs.npy')
#abs_dates = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/EUV_Absolute_Dates_0thresh.npy')
#abs_ARs = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/EUV_Absolute_ARs_0thresh.npy')
abs_dates = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/EUV_Absolute_Dates_0thresh_revised.npy')
abs_ARs = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/EUV_Absolute_ARs_0thresh_revised.npy')

plt.figure(figsize=(20,10))
ax1 = plt.gca()

latN_tot = []
latS_tot = []
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
    frmN = frames[latitude > 0]
    lonS = longitude[latitude < 0]
    latS = latitude[latitude < 0]
    frmS = frames[latitude < 0]
    
    latN_tot = np.append(latN_tot, latN)
    latS_tot = np.append(latS_tot, latS)
    frmN_tot = np.append(frmN_tot, frmN)
    frmS_tot = np.append(frmS_tot, frmS)
    #plt.scatter(frames, longitude)
    #plt.scatter(frmN, lonN)
    #plt.scatter(frmS, lonS)
    #plt.scatter(frames, latitude)
    
frm_indices = ['2010/05/19', '2011/01/01', '2012/01/01', '2013/01/01', '2014/01/01', '2014/05/31']
frm_ticks = [0, 227, 592, 958, 1323, 1473]
    
plt.scatter(frmN_tot, latN_tot)
plt.scatter(frmS_tot, latS_tot)

mN, bN = np.polyfit(frmN_tot, latN_tot, 1)
mS, bS = np.polyfit(frmS_tot, latS_tot, 1)

ax1.plot(frmN_tot, mN*frmN_tot + bN, color='red', linestyle='dashed', linewidth=3.)
ax1.plot(frmS_tot, mS*frmS_tot + bS, color='red', linestyle='dashed', linewidth=3.)

ax1.set_xlim(0,1500)
ax1.set_ylim(-40,40)

plt.xticks(frm_ticks, frm_indices, fontsize=19)
plt.yticks(fontsize=19)

ax1.set_title('EUV Plage Regions: Centroid Latitude vs Time | Threshold = 24', y=1.01, fontsize=font_size)
ax1.set_xlabel('Date', fontsize=font_size)
ax1.set_ylabel('Latitude [Deg]', fontsize=font_size)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/Latitude_vs_Time_Scatter_24thresh_rev.pdf', bbox_inches='tight')


    
"""    
frm_max = np.max(frames)   

## NOAA
abs_dates = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/8_1/NOAA_Absolute_Dates.npy')
abs_ARs = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/8_1/NOAA_Absolute_ARs.npy')

#plt.figure(figsize=(20,10))

longitude = abs_ARs[0]
latitude = abs_ARs[1]
intensity = abs_ARs[2]
frames = abs_ARs[3]
    
lonN = longitude[latitude > 0]
frmN = frames[latitude > 0]
lonS = longitude[latitude < 0]
frmS = frames[latitude < 0]
#plt.scatter(frames, longitude)
#plt.scatter(frmN, lonN, color='red')
plt.scatter(frmS, lonS, color='red')
plt.ylim(0,360)
plt.xlim(0,np.max([frm_max,np.max(frames)]))
"""