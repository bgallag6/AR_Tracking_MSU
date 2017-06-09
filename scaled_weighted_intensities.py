# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 18:06:00 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
"""
#s = readsav('fits_sample_strs_20161219v7.sav')
s = readsav('fits_strs_20161219v7.sav')

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

long_scaled_intensity = np.zeros((36))

for i in range(n_regions.size):
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
        for k in range(36):
            if tempx/10. >= k and tempx/10. < k+1:
                long_scaled_intensity[k] += (tot_int1[i][j]/med_inten[i])
    total_intensity = np.append(total_intensity, temp_int)
      

xticks_long = [60*i for i in range(7)]
xticks_lat = [-45+(15*i) for i in range(7)]

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

fig = plt.figure(figsize=(22,11))

plt.title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date Range: 2010/05/13 - 2016/05/14', y=1.01, fontweight='bold', fontsize=font_size)
plt.plot(n_regions)
plt.ylabel('Number of Active Regions Detected', fontsize=font_size)
plt.xlabel('Number of 12-Hour Periods Passed', fontsize=font_size)
plt.tick_params(axis='both', labelsize=font_size, pad=7)

#plt.savefig('C:/Users/Brendan/Desktop/Number_Active_Regions.pdf', format='pdf')


fig = plt.figure(figsize=(22,11))

plt.suptitle(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date Range: 2010/05/13 - 2016/05/14', y=0.96, fontweight='bold', fontsize=font_size)

ax = plt.subplot2grid((11,11),(1, 0), colspan=5, rowspan=10)
ax = plt.gca()
ax.set_title(r'Longitude', y = 1.01, fontsize=25)
ax.set_xlim(0,360)
ax.set_ylabel('Count', fontsize=font_size)
ax.set_xlabel('Degrees', fontsize=font_size)
ax.set_xticks(xticks_long)
ax.tick_params(axis='both', labelsize=font_size, pad=7)
ax.hist(all_xcoords,bins=36)

ax1 = plt.subplot2grid((11,11),(1, 6), colspan=5, rowspan=10)
ax1 = plt.gca()
ax1.set_title(r'Latitude', y = 1.01, fontsize=25)
ax1.set_xlim(-45,45)
ax1.set_ylabel('Count', fontsize=font_size)
ax1.set_xlabel('Degrees', fontsize=font_size)
ax1.set_xticks(xticks_lat)
ax1.tick_params(axis='both', labelsize=font_size, pad=7)
ax1.hist(all_ycoords,bins=36)

#plt.savefig('C:/Users/Brendan/Desktop/Long_Lat_Unweighted.pdf', format='pdf')


fig = plt.figure(figsize=(22,11))

plt.suptitle(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date Range: 2010/05/13 - 2016/05/14', y=0.96, fontweight='bold', fontsize=font_size)

ax = plt.subplot2grid((11,11),(1, 0), colspan=5, rowspan=10)
ax = plt.gca()
ax.set_title(r'Longitude -- Raw', y = 1.01, fontsize=25)
ax.set_xlim(0,360)
ax.set_ylabel('Count', fontsize=font_size)
ax.set_xlabel('Degrees', fontsize=font_size)
ax.set_xticks(xticks_long)
ax.tick_params(axis='both', labelsize=font_size, pad=7)
ax.hist(all_xcoords,bins=36)

ax1 = plt.subplot2grid((11,11),(1, 6), colspan=5, rowspan=10)
ax1 = plt.gca()
ax1.set_title(r'Longitude -- Scaled', y = 1.01, fontsize=25)
ax1.set_xlim(0,360)
ax1.set_ylabel('Total Scaled Intensity', fontsize=font_size)
ax1.set_xlabel('Degrees', fontsize=font_size)
ax1.set_xticks(xticks_long)
ax1.tick_params(axis='both', labelsize=font_size, pad=7)
ax1.bar([10*i for i in range(36)], long_scaled_intensity)

#plt.savefig('C:/Users/Brendan/Desktop/Long_Raw_Weighted.pdf', format='pdf')
"""

#r = all_tot_int1.sort()

plt.plot(all_tot_int1)
