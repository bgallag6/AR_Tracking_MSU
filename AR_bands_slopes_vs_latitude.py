# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:58:32 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt

fit_params = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_slopes.npy')
ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_lat.npy')

count = 0
for i in range(500):
    if ARs[i,0,0] == 0.:
        count = i
        break
    
slopes = fit_params[:,0]
slopes = slopes[slopes != 0]

med_lat = np.zeros((count))
avg_lat = np.zeros((count))
    
for c in range(count):
    lat_temp = ARs[c,3,:]
    lat_temp = lat_temp[lat_temp != 0]
    med_lat[c] = np.median(lat_temp)
    avg_lat[c] = np.average(lat_temp)

m, b = np.polyfit(med_lat, slopes, 1)

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
plt.figure(figsize=(15,10))
plt.title('Band Slopes: Southern Hemisphere', y=1.01, fontsize=font_size)
plt.scatter(med_lat,slopes)
plt.plot(med_lat, m*med_lat + b, 'r-')   
plt.xlabel('Latitude [Deg]',fontsize=font_size)
plt.ylabel('Slope',fontsize=font_size)
plt.xticks(fontsize=font_size-3)
plt.yticks(fontsize=font_size-3)
#plt.savefig('C:/Users/Brendan/Desktop/Band_Slopes_South.jpeg', bbox_inches = 'tight')