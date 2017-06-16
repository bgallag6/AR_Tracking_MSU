# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:43:07 2017

@author: Brendan
"""

"""
#########################################
### based on number of frames ###########
### - shows full animated scatter  ######
###   and emergence histogram ###########
#########################################
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
import jdcal

import csv
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import urllib2
import urllib
import matplotlib.pyplot as plt
from astropy.time import Time
import jdcal
import datetime

fmt = '%Y%m%d'

dates = []
AR_num = []
Latitude = []
Longitude = []

count = 0

with open('C:/Users/Brendan/Desktop/NOAA AR/2011_NOAA_AR.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     
     for row in spamreader:
        if count > 0:
             date = '%s%s%s' % (row[0],row[1],row[2])
             dt = datetime.datetime.strptime(date[0:8],fmt)
             jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
             dates = np.append(dates, jul_date)
             Latitude = np.append(Latitude, int(row[5]))
             Longitude = np.append(Longitude, int(row[4]))
             if count == 1:
                 date_start = '%s/%s/%s' % (row[0],row[1],row[2])
        count += 1

date_end = '%s/%s/%s' % (row[0],row[1],row[2])
             
dates = np.array(dates)
Latitude = np.array(Latitude)
Longitude = np.array(Longitude)
             
frms = dates - dates[[0]]
frms = frms*2

frmN = frms[Latitude > 0]
frmS = frms[Latitude < 0]
xN = Longitude[Latitude > 0]
xS = Longitude[Latitude < 0]


plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

#"""  ### plot North / South Hemispheres scatter
fig = plt.figure(figsize=(22,11))
ax = plt.gca()
ax.set_title(r'NOAA Data [Northern Hemisphere] | Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Longitude', fontsize=font_size)
ax.set_xlabel('Frame', fontsize=font_size)
ax.set_xlim(0,730)
ax.set_ylim(0,360)  
im = ax.scatter(frmN, xN)  

fig = plt.figure(figsize=(22,11))
ax = plt.gca()
ax.set_title(r'NOAA Data [Southern Hemisphere] | Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Longitude', fontsize=font_size)
ax.set_xlabel('Frame', fontsize=font_size)
ax.set_xlim(0,730)
ax.set_ylim(0,360)  
im = ax.scatter(frmS, xS)  

fig = plt.figure(figsize=(22,11))
ax = plt.gca()
ax.set_title(r'NOAA Data [Full Disc] | Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Longitude', fontsize=font_size)
ax.set_xlabel('Frame', fontsize=font_size)
ax.set_xlim(0,730)
ax.set_ylim(0,360)  
im = ax.scatter(frms, Longitude)  
#"""