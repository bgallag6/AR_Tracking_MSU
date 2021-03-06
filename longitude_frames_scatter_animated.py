# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:38:15 2017

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

#"""
s = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)
f_names = np.load('C:/Users/Brendan/Desktop/Files/MSU_Project/Active_Longitude/ar_filenames.npy')

trim = 2922  # image before jump 20140818-20151103

coord = s.STRS.coordinates  # rectangular box in pixels
cen_coord = s.STRS.centroid_cord  # centroid in degrees
n_regions = s.STRS.n_region
med_inten = s.STRS.median_intensity
tot_int1 = s.STRS.tot_int1
tot_area1 = s.STRS.tot_area1

all_cen_coords = cen_coord.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()

images = 300
num_ar = 300
start_frame = 2500
int_thresh = 30

ARs = np.zeros((num_ar,3,images))
count = 0

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

fig = plt.figure(figsize=(22,11))
ax = plt.gca()
canvas = ax.figure.canvas
#ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date: %i/%i/%i' % (dt_greg1[0], dt_greg1[1], dt_greg1[2]), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Longitude', fontsize=font_size)
ax.set_xlabel('Frame', fontsize=font_size)
ax.set_xlim(0,300)
ax.set_ylim(0,360)
background = fig.canvas.copy_from_bbox(ax.bbox)
plt.ion()

for i in range(start_frame,start_frame+images):
    
    start = f_names[i][0:8]

    date_title = '%s/%s/%s' % (start[0:4],start[4:6],start[6:8])
    
    intensities0 = np.array(all_tot_int1[i])
    #intensities = [0 if x < int_thresh else x for x in intensities]  # maybe also eliminate zeros
    intensities = intensities0[intensities0 > int_thresh]    
    xcoords0 = np.array(all_cen_coords[i])[:,0]
    ycoords0 = np.array(all_cen_coords[i])[:,1]
    #for q in range(len(intensities)):
    #    if intensities[q] < int_thresh:
    #        xcoords[q] = 0
    #        ycoords[q] = 0
    xcoords = xcoords0[intensities0 > int_thresh]
    ycoords = ycoords0[intensities0 > int_thresh]
    
    """
    num_reg = n_regions[i]
    for k in range(num_reg):  # if got rid zeros, just range(intensities.size)
        if intensities[k] > 0:
            found = 0
            for c in range(count):
                dr = np.sqrt((ARs[c,1,(i-start_frame-1)]-xcoords[k])**2 + (ARs[c,2,(i-start_frame-1)]-ycoords[k])**2)
                if dr < 5:
                    ARs[c,0,i-start_frame] = intensities[k]
                    ARs[c,1,i-start_frame] = xcoords[k]
                    ARs[c,2,i-start_frame] = ycoords[k]
                    found = 1  # maybe need to say closest one if more than one found?  don't think should be issue
            if found == 0:
                ARs[count,0,i-start_frame] = intensities[k]
                ARs[count,1,i-start_frame] = xcoords[k]
                ARs[count,2,i-start_frame] = ycoords[k]
                count += 1
    """
    

    ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date: %s' % (date_title), y=1.01, fontweight='bold', fontsize=font_size)
    canvas.restore_region(background)
    
    x_ar0 = xcoords[xcoords != 0]
    y_ar0 = ycoords[ycoords != 0]
    int_ar0 = intensities[intensities != 0]
    
    #x_ar02 = xcoords[xcoords > 0]
    x_ar02 = xcoords[ycoords < 0]
    #int_ar02 = np.zeros((len(x_ar02)))
    #for v in range(len(x_ar02)):
    #    int_ar02[v] = int_ar0[]
    
    frms = np.array([i-start_frame for y in range(len(x_ar0))]) 
    #frms = np.array([i-start_frame for y in range(len(x_ar02))]) 
    
    im = ax.scatter(frms, x_ar0, c=color[i-start_frame])
    #im = ax.scatter(frms, x_ar0)
    canvas.blit(ax.bbox)
    plt.pause(0.001) # used for 1000 points, reasonable
    #plt.pause(0.1) # used for 1000 points, reasonable
    #plt.pause(0.5) # used for 1000 points, reasonable
#"""