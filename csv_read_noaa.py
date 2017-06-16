# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:17:28 2017

@author: Brendan
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

count = 0

dates = []
AR_num = []
Latitude = []
Longitude = []

with open('C:/Users/Brendan/Desktop/NOAA AR/2011_NOAA_AR.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     
     for row in spamreader:
        if count > 0:
             date = '%s%s%s' % (row[0],row[1],row[2])
             dates = np.append(dates, date)
             AR_num = np.append(AR_num, int(row[3]))
             Latitude = np.append(Latitude, int(row[5]))
             Longitude = np.append(Longitude, int(row[4]))
        count += 1
             
dates = dates.tolist()
AR_num = AR_num.tolist()
Latitude = Latitude.tolist()
Longitude = Longitude.tolist()

plt.hist(Longitude)
