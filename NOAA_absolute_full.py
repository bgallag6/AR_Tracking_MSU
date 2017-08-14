# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 18:40:44 2017

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

datesNOAA = []
datesD = []
AR_num = []
latitude = []
longitude = []
area = []

count = 0

fmt = '%Y%m%d'

datesARs = []
greg_date = []

with open('C:/Users/Brendan/Desktop/NOAA_2010_2014.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     
     for row in spamreader:
        if count > 0:
             date = '%i%0.2i%0.2i' % (int(row[0]),int(row[1]),int(row[2]))
             datesD = np.append(datesD, date)
             greg_date = np.append(greg_date, date)
             dt = datetime.datetime.strptime(date[0:8],fmt)
             jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
             datesNOAA = np.append(datesNOAA, jul_date)
             latitude = np.append(latitude, int(row[6]))
             longitude = np.append(longitude, int(row[5]))
             area = np.append(area, float(row[4]))
        count += 1

datesNOAA0 = np.copy(datesNOAA) 
datesNOAA0 -= datesNOAA[0]
datesNOAA0 += 2

dates_final = np.zeros((3,len(greg_date)+2))
dates_final[0,2:len(greg_date)+2] = datesNOAA0
dates_final[1,2:len(greg_date)+2] = datesNOAA
dates_final[2,2:len(greg_date)+2] = greg_date     
dates_final[0,1] = 1

arr_total = np.zeros((4,len(greg_date)+2)) 

xtemp = longitude
ytemp = latitude
areatemp = area
arr_total[0,2:len(greg_date)+2] = xtemp
arr_total[1,2:len(greg_date)+2] = ytemp
arr_total[2,2:len(greg_date)+2] = areatemp
arr_total[3,2:len(greg_date)+2] = datesNOAA0
arr_total[3,1] = 1

   
#np.save('C:/Users/Brendan/Desktop/NOAA_Absolute_ARs.npy', arr_total)
#np.save('C:/Users/Brendan/Desktop/NOAA_Absolute_Dates.npy', dates_final)