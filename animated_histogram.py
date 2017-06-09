# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 17:06:09 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt  
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from scipy.io.idl import readsav

#s = readsav('fits_sample_strs_20161219v7.sav')
s = readsav('fits_strs_20161219v7.sav')

coord = s.STRS.coordinates  # rectangular box in pixels
cen_coord = s.STRS.centroid_cord  # centroid in degrees
n_regions = s.STRS.n_region

all_xcoords = []
all_ycoords = []

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

xticks_long = [60*i for i in range(7)]
xticks_lat = [-90+(30*i) for i in range(7)]



fig = plt.figure(figsize=(20,10))

ax = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
ax = plt.gca()
ax.set_title(r'Longitude', y = 1.01, fontsize=25)
ax.set_xlim(0,360)
ax.set_ylabel('Count')
ax.set_xlabel('Degrees')
ax.set_xticks(xticks_long)

ax1 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
ax1 = plt.gca()
ax1.set_title(r'Latitude', y = 1.01, fontsize=25)
ax1.set_xlim(-90,90)
ax1.set_ylabel('Count')
ax1.set_xlabel('Degrees')
ax1.set_xticks(xticks_lat)

plt.show(False)
plt.draw()
fig.canvas.draw()

year = 2010
month = 5
day = 13

for i in range(n_regions.size):
    num_reg = n_regions[i]
    date = '%i/%0.2i/%0.2i' % (year, month, day)
    for j in range(num_reg):
        all_xcoords = np.append(all_xcoords, cen_coord[i][j][0])
        all_ycoords = np.append(all_ycoords, cen_coord[i][j][1])
        
        ax.clear()
        ax.set_title(r'Longitude', y = 1.01, fontsize=25)
        ax.set_xlim(0,360)
        ax.set_ylabel('Count')
        ax.set_xlabel('Degrees')
        ax.set_xticks(xticks_long)
        ax.hist(all_xcoords)
        
        
        ax1.clear()
        ax1.set_title(r'Latitude', y = 1.01, fontsize=25)
        ax1.set_xlim(-90,90)
        ax1.set_ylabel('Count')
        ax1.set_xlabel('Degrees')
        ax1.set_xticks(xticks_lat)
        ax1.hist(all_ycoords)

        plt.suptitle('Date: %s' % date)
        
        plt.pause(0.0001)
        
    day += 0.5
        
    