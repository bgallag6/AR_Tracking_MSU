# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:59:45 2017

@author: Brendan
"""

"""
#########################################
### based on number of frames ###########
### - shows full animated scatter  ######
### (longitude vs frame)  ###############
#########################################
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

fmt = '%Y%m%d'

dates = []
datesD = []
AR_num = []
Latitude = []
Longitude = []

count = 0

with open('C:/Users/Brendan/Desktop/Week3/NOAA AR/Full_NOAA_AR.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     
     for row in spamreader:
        if count > 0:
             date = '%i%0.2i%0.2i' % (int(row[0]),int(row[1]),int(row[2]))
             datesD = np.append(datesD, date)
             dt = datetime.datetime.strptime(date[0:8],fmt)
             jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
             dates = np.append(dates, jul_date)
             Latitude = np.append(Latitude, int(row[5]))
             Longitude = np.append(Longitude, int(row[4]))
        count += 1
             
dates0 = np.array(dates)
dates = dates0*2
dates -= dates[0]
Latitude = np.array(Latitude)
Longitude = np.array(Longitude)
             
#frms = dates - dates[[0]]
#frms = frms*2

#frmN = frms[Latitude > 0]
#frmS = frms[Latitude < 0]
xN = Longitude[Latitude > 0]
datesN = dates[Latitude > 0]
xS = Longitude[Latitude < 0]
datesS = dates[Latitude < 0]

font_size = 17

x_bins = [20*l for l in range(19)]
x_ticks = [40*l for l in range(10)]

rotations = 3

car_days = [(rotations*27.25*2)*i for i in range(18)]

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

for c in range(17):
    
    startIndN = np.searchsorted(datesN,car_days[c])  # dont' think this is exactly correct, but close?
    endIndN = np.searchsorted(datesN,car_days[c+1])
    
    startIndS = np.searchsorted(datesS,car_days[c])  # dont' think this is exactly correct, but close?
    endIndS = np.searchsorted(datesS,car_days[c+1])
    
    startIndD = np.searchsorted(dates,car_days[c])  # dont' think this is exactly correct, but close?
    endIndD = np.searchsorted(dates,car_days[c+1])
    
    date_start = datesD[startIndD]
    date_end = datesD[endIndD]
    
    date_start = '%s/%s/%s' % (date_start[0:4],date_start[4:6],date_start[6:8])
    date_end = '%s/%s/%s' % (date_end[0:4],date_end[4:6],date_end[6:8])

    ### plot North / South Hemispheres scatter
    fig = plt.figure(figsize=(22,10))
    plt.suptitle(r'NOAA Nothern Hemisphere - Carrington Rotation Periods: %i - %i' % ((c*rotations)+1, ((c+1)*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=0.97, fontweight='bold', fontsize=font_size)
    ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
    ax1 = plt.gca()    
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Frame', fontsize=font_size)
    ax1.set_xlim(car_days[c],car_days[c+1])
    ax1.set_ylim(0,360)   
    ax1.scatter(datesN[startIndN:endIndN], xN[startIndN:endIndN])  
    
    ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
    ax2 = plt.gca()
    ax2.set_ylabel('Number of ARs', fontsize=font_size)
    ax2.set_xlabel('Longitude', fontsize=font_size)
    ax2.set_xlim(0,360)
    ax2.hist(xN[startIndN:endIndN], bins=x_bins) 
    plt.xticks(x_ticks)
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Car_Rot_%i_%i_North.pdf' % ((c*rotations)+1, ((c+1)*rotations)), bbox_inches = 'tight')
    plt.close()
    
    #"""
    fig = plt.figure(figsize=(22,10))
    plt.suptitle(r'NOAA Southern Hemisphere - Carrington Rotation Periods: %i - %i' % ((c*rotations)+1, ((c+1)*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=0.97, fontweight='bold', fontsize=font_size)
    ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
    ax1 = plt.gca()
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Frame', fontsize=font_size)
    ax1.set_xlim(car_days[c],car_days[c+1])
    ax1.set_ylim(0,360)  
    ax1.scatter(datesS[startIndS:endIndS], xS[startIndS:endIndS])  
    
    ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
    ax2 = plt.gca()
    ax2.set_ylabel('Number of ARs', fontsize=font_size)
    ax2.set_xlabel('Longitude', fontsize=font_size)
    ax2.set_xlim(0,360)
    ax2.hist(xS[startIndS:endIndS], bins=x_bins)  
    plt.xticks(x_ticks)
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Car_Rot_%i_%i_South.pdf' % ((c*rotations)+1, ((c+1)*rotations)), bbox_inches = 'tight')
    plt.close()    
    #"""

