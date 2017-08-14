# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:13:15 2017

@author: Brendan
"""
import numpy as np
import jdcal
import csv
import urllib2
import urllib
from astropy.time import Time
import datetime
import matplotlib.pyplot as plt

"""
arr_len = 0
cumulative = 0
fmt = '%Y%m%d'

for k in range(140):
    text_file = open('C:/Users/Brendan/Downloads/RGO_NOAA1874_2013/g%i.txt' % (1874+k), 'r')
    lines = text_file.read().split('\n')
    arr_len += (len(lines)-1)
    text_file.close()

arr = np.zeros((arr_len,8))

for k in range(140):
    text_file = open('C:/Users/Brendan/Downloads/RGO_NOAA1874_2013/g%i.txt' % (1874+k), 'r')
    lines = text_file.read().split('\n')
    temp_len = (len(lines)-1)
    text_file.close()


    for i in range(temp_len):
    #for i in range(1):
        year = int(lines[i][0:4])
        month = int(lines[i][4:6])
        day = np.floor(float(lines[i][6:12]))
        r1 = lines[i][12:22] # sunpsot group number
        r2 = lines[i][22:24] # NOAA group type
        r3 = float(lines[i][24:29]) # depends on year (observed umbral area)
        r4 = float(lines[i][29:34]) # observed whole spot area in millionths of solar disk
        r5 = float(lines[i][34:39]) # depends on year (corrected umbral area)
        r6 = float(lines[i][39:44]) # corrected whole spot area in millions of solar hemisphere
        r7 = float(lines[i][44:50]) # distance from center of solar disk (in disk radii)
        r8 = float(lines[i][50:56]) # position angle from heliographic north
        r9 = float(lines[i][56:62]) # longitude
        r10 = float(lines[i][62:68]) # latitude
        r11 = float(lines[i][68:74]) # central meridian distance
        date = '%i%0.2i%0.2i' % (int(year),int(month),int(day))
        dt = datetime.datetime.strptime(date[0:8],fmt)
        jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
        temp = year, month, day, jul_date, r4, r6, r9, r10
        arr[cumulative+i] = temp
    
    cumulative += temp_len
"""
lat = arr[:,7]
latN = lat[lat > 0]
latS = lat[lat < 0]
day = arr[:,3]
day -= day[0]
dayN = day[lat > 0]
dayS = day[lat < 0]

plt.figure()
plt.hist(latN, bins=50, range=(0,50))
plt.figure()
plt.hist(latS, bins=50, range=(-50,0))

plt.figure()
plt.plot(dayN, latN)

    
plt.figure(figsize=(20,10))
plt.scatter(dayN,latN)
plt.scatter(dayS,latS)
plt.ylim(-55,55)
plt.xlim(-1000,52000)
#plt.savefig('C:/Users/Brendan/Desktop/Butterfly_Diagram_1874_2013.jpeg', bbox_inches='tight')
#plt.figure()

dayN_temp = dayN[700:5450]
latN_temp = latN[700:5450]
day_full = np.array([c for c in range(1700,5600)])
plt.figure(figsize=(20,10))
plt.scatter(dayN[700:5450],latN[700:5450])
m, b = np.polyfit(dayN_temp, latN_temp, 1)
plt.plot(day_full, b + m*day_full, color='red', linestyle='dashed', linewidth=5.)
