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
abs_dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/8_1/EUV_Absolute_Dates.npy')
abs_ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/8_1/EUV_Absolute_ARs.npy')

plt.figure(figsize=(30,10))

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
    frmN = frames[latitude > 0]
    lonS = longitude[latitude < 0]
    frmS = frames[latitude < 0]
    #plt.scatter(frames, longitude)
    #plt.scatter(frmN, lonN)
    plt.scatter(frmS, lonS)
    
frm_max = np.max(frames)   

## NOAA
abs_dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/8_1/NOAA_Absolute_Dates.npy')
abs_ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/8_1/NOAA_Absolute_ARs.npy')

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