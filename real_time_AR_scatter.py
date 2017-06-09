# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 16:42:47 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.idl import readsav
import jdcal

#"""
def linear(f, m, b):
    return m*f + b
  
#s = readsav('fits_sample_strs_20161219v7.sav')
s = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)

trim = 2922  # image before jump 20140818-20151103

coord = s.STRS.coordinates  # rectangular box in pixels
cen_coord = s.STRS.centroid_cord  # centroid in degrees
n_regions = s.STRS.n_region
med_inten = s.STRS.median_intensity
tot_int1 = s.STRS.tot_int1
tot_area1 = s.STRS.tot_area1
tot_int3 = s.STRS.tot_int3
tot_area3 = s.STRS.tot_area3

all_xcoords = []
all_ycoords = []
all_med_inten = []
all_tot_int1 = []
all_tot_area1 = []
all_tot_int3 = []
all_tot_area3 = []
total_intensity = []
all_scaled_intensity = []

long_scaled_intensity = np.zeros((18))

#for i in range(n_regions.size):
for i in range(trim):
    num_reg = n_regions[i]
    all_med_inten = np.append(all_med_inten, med_inten[i])
    temp_int = 0
    for j in range(num_reg):
        all_xcoords = np.append(all_xcoords, cen_coord[i][j][0])
        all_ycoords = np.append(all_ycoords, cen_coord[i][j][1])       
        all_tot_int1 = np.append(all_tot_int1, tot_int1[i][j])
        all_tot_area1 = np.append(all_tot_area1, tot_area1[i][j])
        all_tot_int3 = np.append(all_tot_int3, tot_int3[i][j])
        all_tot_area3 = np.append(all_tot_area3, tot_area3[i][j])
        tempx = cen_coord[i][j][0]
        tempy = cen_coord[i][j][1]
        temp_int += tot_int1[i][j]
        all_scaled_intensity = np.append(all_scaled_intensity, tot_int1[i][j]/med_inten[i])
        for k in range(18):
            if tempx/20. >= k and tempx/20. < k+1:
                long_scaled_intensity[k] += (tot_int1[i][j]/med_inten[i])
    total_intensity = np.append(total_intensity, temp_int)
      

xticks_long = [60*i for i in range(7)]
xticks_lat = [-90+(30*i) for i in range(7)]
    

seg = (dates[trim]-dates[0])/27
#n = int(all_xcoords.size/seg)

lat_max = 50
long_max = 100

lat_bin_size = 1
long_bin_size = 2

lat_bins = np.arange(-90, 90 + lat_bin_size, lat_bin_size)
long_bins = np.arange(0, 360 + long_bin_size, long_bin_size)

lat_num = lat_bins.size
long_num = long_bins.size
#"""

all_tot_int1 *= 2

#lat_mode = np.zeros((seg,2))
#long_mode = np.zeros((seg))

xticks_long = [60*i for i in range(7)]
xticks_lat = [-90+(30*i) for i in range(7)]

#"""
c = np.zeros((1000,3))
for t in range(1000):
    c[t][0] = t*0.001
    c[t][2] = 1. - t*0.001
#"""

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

fig = plt.figure(figsize=(22,11))
ax = plt.gca()
#canvas = ax.figure.canvas
ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date Range: 2010/05/13 - 2016/05/14', y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Latitude', fontsize=font_size)
ax.set_xlabel('Longitude', fontsize=font_size)
#plt.tick_params(axis='both', labelsize=font_size, pad=7)
im = ax.scatter(all_xcoords[0:5],all_ycoords[0:5],all_tot_int1[0:5],c=c[0],cmap='jet')
ax.set_xlim(0,360)
ax.set_ylim(-45,45)
plt.ion()
"""
plt.show(False)
plt.draw()
# cache the background
background = fig.canvas.copy_from_bbox(ax.bbox)
fig.canvas.draw()
"""
cum_ARs = 0

#"""
#for i in range(1,n_regions.size):
for i in range(1,100):
#for i in range(1,1000):
#for i in range(1001,2000):
    
    start = dates[0] + i
    
    dt_begin = 2400000.5
    dt_dif1 = start-dt_begin    
    dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)
    
    img_ARs = n_regions[i]
    
    """
    fig = plt.figure(figsize=(22,11))
    
    ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + 'Date: %i/%i/%i' % (dt_greg1[0], dt_greg1[1], dt_greg1[2]), y=1.01, fontweight='bold', fontsize=font_size)
    ax.set_ylabel('Latitude', fontsize=font_size)
    ax.set_xlabel('Longitude', fontsize=font_size)
    #plt.tick_params(axis='both', labelsize=font_size, pad=7)
    ax.scatter(all_xcoords[cum_ARs:(cum_ARs+img_ARs)],all_ycoords[cum_ARs:(cum_ARs+img_ARs)],all_tot_int1[cum_ARs:(cum_ARs+img_ARs)])
    ax.set_xlim(0,360)
    ax.set_ylim(-45,45)
    """
    #canvas.restore_region(background)
    # redraw just the points
    #im.set_data(all_xcoords[cum_ARs:(cum_ARs+img_ARs)],all_ycoords[cum_ARs:(cum_ARs+img_ARs)],all_tot_int1[cum_ARs:(cum_ARs+img_ARs)])
    # fill in the figure
    #canvas.blit(ax.bbox)
    ax.scatter(all_xcoords[cum_ARs:(cum_ARs+img_ARs)],all_ycoords[cum_ARs:(cum_ARs+img_ARs)],all_tot_int1[cum_ARs:(cum_ARs+img_ARs)],c=c[i],cmap='jet')
    #plt.pause(0.01)
    plt.pause(0.1) # used for 1000 points, reasonable
    
    cum_ARs += img_ARs

        
"""
### if want to make 2nd 1000 points, have to find cum_AR of first 1000, and start with that - duh
when changing points - subtract from 'c'
"""