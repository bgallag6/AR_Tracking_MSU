# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 16:25:21 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.idl import readsav
import jdcal

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

all_cen_coords = cen_coord.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()


intensities = np.array(all_tot_int1[0])
intensities = [0 if x < 35 else x for x in intensities]
xcoords = np.array(all_cen_coords[0])[:,0]
ycoords = np.array(all_cen_coords[0])[:,1]

images = 100
num_ar = 100

ARs = np.zeros((num_ar,3,images))
count = 0

for k in range(100):
    if intensities[k] > 0:
        ARs[count,0,0] = intensities[k]
        ARs[count,1,0] = xcoords[k]
        ARs[count,2,0] = ycoords[k]
        count += 1

start = dates[0]
dt_begin = 2400000.5
dt_dif1 = start-dt_begin    
dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)

color = np.zeros((images,3))
for t in range(images):
    color[t][0] = t*(1./images)
    color[t][2] = 1. - t*(1./images)
  
plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

"""    
fig = plt.figure(figsize=(22,11))
ax = plt.gca()
canvas = ax.figure.canvas
ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date: %i/%i/%i' % (dt_greg1[0], dt_greg1[1], dt_greg1[2]), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Latitude', fontsize=font_size)
ax.set_xlabel('Longitude', fontsize=font_size)
#plt.tick_params(axis='both', labelsize=font_size, pad=7)
#im = ax.scatter(np.array(all_cen_coords[0])[:,0], np.array(all_cen_coords[0])[:,1],np.array(all_tot_int1[0]))
#im = ax.scatter(np.array(all_cen_coords[0])[:,0], np.array(all_cen_coords[0])[:,1],intensities)
im = ax.scatter(xcoords, ycoords,intensities, c=color[0])
ax.set_xlim(0,360)
ax.set_ylim(-45,45)
background = fig.canvas.copy_from_bbox(ax.bbox)
plt.ion()
"""
num_reg_above_thresh = np.zeros(())

### need to add time component to AR array? - or just put in which image it is

for i in range(1,images):
    
    start = dates[0] + i
    
    dt_begin = 2400000.5
    dt_dif1 = start-dt_begin    
    dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)
    
    
    
    
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

xticks_long = [60*i for i in range(7)]
xticks_lat = [-90+(30*i) for i in range(7)]

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

cum_ARs = 0

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
    
    #all_tot_int1 = [0 if x < 35 else x for x in all_tot_int1]
    intensities = np.array(all_tot_int1[cum_ARs:rot_ARs])
    intensities = [0 if x < 35 else x for x in intensities]  # maybe also eliminate zeros
    xcoords = np.array(all_cen_coords[i])[:,0]
    ycoords = np.array(all_cen_coords[i])[:,1]

    
    num_reg = n_regions[i]
    for k in range(num_reg):  # if got rid zeros, just range(intensities.size)
        if intensities[k] > 0:
            found = 0
            #print intensities[k]
            #for c in range(100):  # somehow just current AR -- count duh
            for c in range(count):
                dr = np.sqrt((ARs[c,1,(i-1)]-xcoords[k])**2 + (ARs[c,2,(i-1)]-ycoords[k])**2)
                #print "dr = %0.2f" % dr
                if dr < 5:
                    ARs[c,0,i] = intensities[k]
                    ARs[c,1,i] = xcoords[k]
                    ARs[c,2,i] = ycoords[k]
                    found = 1  # maybe need to say closest one if more than one found?  don't think should be issue
                    #print "found"
            if found == 0:
                ARs[count,0,i] = intensities[k]
                ARs[count,1,i] = xcoords[k]
                ARs[count,2,i] = ycoords[k]
                count += 1
                #print "count = ", count
      
    frames = np.zeros((count))
    first = np.zeros((count))
    x_form = np.zeros((count))
    y_form = np.zeros((count))
    x_end = np.zeros((count))
    y_end = np.zeros((count))
    x_avg = np.zeros((count))
    y_avg = np.zeros((count))
    avg_int = np.zeros((count))
    med_int = np.zeros((count))
    distance = np.zeros((count))
    
    for c in range(count): 
    
        ar0 = ARs[c,:,:]
        
        frames[c] = np.count_nonzero(ar0,axis=1)[0] # - how many frames lasts
        for d in range(images):
            if ar0[0,d] != 0:
                first[c] = d
                break
        x_form[c] = ar0[1][ar0[1] != 0][0]
        y_form[c] = ar0[2][ar0[2] != 0][0]
        x_end[c] = ar0[1][ar0[1] != 0][-1]
        y_end[c] = ar0[2][ar0[2] != 0][-1]
        
        x_coords = ar0[1][ar0[1] != 0]
        y_coords = ar0[2][ar0[2] != 0]
        x_avg[c] = np.average(x_coords)
        y_avg[c] = np.average(y_coords)
        
        avg_int[c] = np.average(ar0[0][ar0[0] != 0])
        med_int[c] = np.median(ar0[0][ar0[0] != 0])
    
   
    fig = plt.figure(figsize=(22,11))
    
    plt.title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date Range: 2010/05/13 - 2016/05/14', y=1.01, fontweight='bold', fontsize=font_size)
    plt.ylabel('Latitude', fontsize=font_size)
    plt.xlabel('Longitude', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size, pad=7)
    plt.scatter(ARs[cum_ARs:(cum_ARs+rot_ARs),1,xform],y_form[cum_ARs:(cum_ARs+rot_ARs)],avg_int[cum_ARs:(cum_ARs+rot_ARs)])
    plt.xlim(0,360)
    plt.ylim(-45,45)
    
    cum_ARs += rot_ARs
            
    #plt.savefig('C:/Users/Brendan/Desktop/%i_of_%i.pdf' % ((i+1),seg), format='pdf')
    #plt.savefig('C:/Users/Brendan/Desktop/rotations/scatter_%i_of_%i.jpeg' % ((i+1),seg))
    #plt.close()
#"""    
