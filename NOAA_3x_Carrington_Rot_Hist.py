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

year0 = 2010

with open('C:/Users/Brendan/Desktop/Week3/NOAA AR/%i_NOAA_AR.csv' % year0, 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     
     for row in spamreader:
        if count > 0:
             date = '%i%0.2i%0.2i' % (int(row[0]),int(row[1]),int(row[2]))
             dt = datetime.datetime.strptime(date[0:8],fmt)
             jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
             dates = np.append(dates, jul_date)
             Latitude = np.append(Latitude, int(row[5]))
             Longitude = np.append(Longitude, int(row[4]))
             if count == 1:
                 date_start = '%i/%0.2i/%0.2i' % (int(row[0]),int(row[1]),int(row[2]))
        count += 1

date_end = '%i/%0.2i/%0.2i' % (int(row[0]),int(row[1]),int(row[2]))
             
dates = np.array(dates)
dates *= 2
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

x_bins = [20*i for i in range(19)]

car_days = [163*i for i in range(4)]

for i in range(3):
    
    startInd = np.searchsorted(datesN,car_days[i])  # dont' think this is exactly correct, but close?
    endInd = np.searchsorted(datesN,car_days[i+1])
    
    #startD = datesN[startInd]
    #endD = datesN[endInd]

    ### plot North / South Hemispheres scatter
    fig = plt.figure(figsize=(22,10))
    #plt.suptitle(r'Nothern Hemisphere - Carrington Rotation Periods: %i - %i' % ((c*rotations)+1, ((c+1)*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=0.97, fontweight='bold', fontsize=font_size)
    ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
    ax1 = plt.gca()    
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Frame', fontsize=font_size)
    #ax1.set_xlim(ind_start[c],ind_end[c])
    ax1.set_ylim(0,360)   
    ax1.scatter(datesN[startInd:endInd], xN[startInd:endInd])  
    
    ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
    ax2 = plt.gca()
    ax2.set_ylabel('Number of ARs', fontsize=font_size)
    ax2.set_xlabel('Longitude', fontsize=font_size)
    #ax2.set_ylim(0,bin_max)  
    ax2.set_xlim(0,360)
    ax2.hist(xN[startInd:endInd], bins=x_bins) 
    #plt.xticks(x_ticks)
    #plt.savefig('C:/Users/Brendan/Desktop/Car_Rot_%i_%i_North.jpg' % ((c*rotations)+1, ((c+1)*rotations)), bbox_inches = 'tight')
    #plt.close()
    
    """
    fig = plt.figure(figsize=(22,10))
    #plt.suptitle(r'Southern Hemisphere - Carrington Rotation Periods: %i - %i' % ((c*rotations)+1, ((c+1)*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=0.97, fontweight='bold', fontsize=font_size)
    ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
    ax1 = plt.gca()
    
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Frame', fontsize=font_size)
    #ax1.set_xlim(ind_start[c],ind_end[c])
    ax1.set_ylim(0,360)  
    ax1.scatter(datesS, xS)  
    
    ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
    ax2 = plt.gca()
    ax2.set_ylabel('Number of ARs', fontsize=font_size)
    ax2.set_xlabel('Longitude', fontsize=font_size)
    #ax2.set_ylim(0,bin_max)  
    ax2.set_xlim(0,360)
    ax2.hist(xS, bins=x_bins)  
    #plt.xticks(x_ticks)
    #plt.savefig('C:/Users/Brendan/Desktop/Car_Rot_%i_%i_South.jpg' % ((c*rotations)+1, ((c+1)*rotations)), bbox_inches = 'tight')
    #plt.close()    
    """

