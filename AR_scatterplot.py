# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 16:31:41 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.idl import readsav
import jdcal

def linear(f, m, b):
    return m*f + b
#"""   
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
    
#"""
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

#lat_mode = np.zeros((seg,2))
#long_mode = np.zeros((seg))

xticks_long = [60*i for i in range(7)]
xticks_lat = [-90+(30*i) for i in range(7)]

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

cum_ARs = 0
#"""
for i in range(int(seg)):
#for i in range(3):
    
    start = dates[0] + (27*i)
    end = start + 27
    
    dt_begin = 2400000.5
    dt_dif1 = start-dt_begin   
    dt_dif2 = (start+27)-dt_begin  
    dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)
    dt_greg2 = jdcal.jd2gcal(dt_begin,dt_dif2)
    
    ind_start = np.searchsorted(dates,start)  # dont' think this is exactly correct, but close?
    ind_end = np.searchsorted(dates,end)

    rot_ARs = np.sum(n_regions[ind_start:ind_end])
    
    """
    fig = plt.figure(figsize=(22,11))

    plt.suptitle(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date Range: %i/%0.2i/%0.2i - %i/%0.2i/%0.2i' % (dt_greg1[0], dt_greg1[1], dt_greg1[2], dt_greg2[0], dt_greg2[1], dt_greg2[2]) + '\n Rotation Cycle %i of %i' % (i+1, seg), y=0.96, fontweight='bold', fontsize=font_size)
    
    ax = plt.subplot2grid((11,11),(1, 0), colspan=5, rowspan=10)
    ax = plt.gca()
    ax.set_title(r'Longitude', y = 1.01, fontsize=25)
    ax.set_xlim(0,360)
    ax.set_ylim(0,long_max)
    ax.set_ylabel('Count', fontsize=font_size)
    ax.set_xlabel('Degrees', fontsize=font_size)
    ax.set_xticks(xticks_long)
    ax.tick_params(axis='both', labelsize=font_size, pad=7)
    y, x, _ = ax.hist(all_xcoords[cum_ARs:(cum_ARs+rot_ARs)],bins=36)
    
    ax1 = plt.subplot2grid((11,11),(1, 6), colspan=5, rowspan=10)
    ax1 = plt.gca()
    ax1.set_title(r'Latitude', y = 1.01, fontsize=25)
    ax1.set_xlim(-90,90)
    ax1.set_ylim(0,lat_max)
    ax1.set_ylabel('Count', fontsize=font_size)
    ax1.set_xlabel('Degrees', fontsize=font_size)
    ax1.set_xticks(xticks_lat)
    ax1.tick_params(axis='both', labelsize=font_size, pad=7)
    y, x, _ = ax1.hist(all_ycoords[cum_ARs:(cum_ARs+rot_ARs)],lat_bins)
    """
    
    fig = plt.figure(figsize=(22,11))
    
    plt.title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date Range: 2010/05/13 - 2016/05/14', y=1.01, fontweight='bold', fontsize=font_size)
    plt.ylabel('Latitude', fontsize=font_size)
    plt.xlabel('Longitude', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size, pad=7)
    plt.scatter(all_xcoords[cum_ARs:(cum_ARs+rot_ARs)],all_ycoords[cum_ARs:(cum_ARs+rot_ARs)],all_tot_int1[cum_ARs:(cum_ARs+rot_ARs)])
    plt.xlim(0,360)
    plt.ylim(-45,45)
    
    cum_ARs += rot_ARs
            
    #plt.savefig('C:/Users/Brendan/Desktop/%i_of_%i.pdf' % ((i+1),seg), format='pdf')
    #plt.savefig('C:/Users/Brendan/Desktop/rotations/scatter_%i_of_%i.jpeg' % ((i+1),seg))
    #plt.close()
#"""    
