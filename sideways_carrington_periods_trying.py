# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:54:37 2017

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

#trim = 2922  # image before jump 20140818-20151103
trim = 2872  # last index for end of Carrington rotation

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


rotations = 4
seg = ((dates[trim]-dates[11])/27)/rotations
  
ind_start = np.zeros((int(seg)))
ind_end = np.zeros((int(seg)))


for i in range(int(seg)):
#for i in range(3):
    
    start = dates[11] + ((27.25*i)*rotations)
    end = start + (27*rotations)
    
    dt_begin = 2400000.5
    dt_dif1 = start-dt_begin   
    dt_dif2 = (start+27)-dt_begin  
    dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)
    dt_greg2 = jdcal.jd2gcal(dt_begin,dt_dif2)
    
    ind_start[i] = np.searchsorted(dates,start)  # dont' think this is exactly correct, but close?
    ind_end[i] = np.searchsorted(dates,end)


#for c in range(int(seg)):
for c in range(1):
    
    date_start = f_names[int(ind_start[c])][0:8]
    date_end = f_names[int(ind_end[c])][0:8]
    
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 23
    
    fig = plt.figure(figsize=(22,11))
    ax = plt.gca()
    ax.set_title(r'Carrington Rotation Periods: %i - %i' % ((c*rotations)+1, (c*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)
    canvas = ax.figure.canvas
    ax.set_ylabel('Longitude', fontsize=font_size)
    ax.set_xlabel('Frame', fontsize=font_size)
    ax.set_xlim(ind_start[c],ind_end[c])
    ax.set_ylim(0,360)
    background = fig.canvas.copy_from_bbox(ax.bbox)
    plt.ion()
    
    for i in range(int(ind_start[c]),int(ind_end[c])):
        
        start = f_names[i][0:8]
        start_frame = int(ind_start[0])
    
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
                
        canvas.restore_region(background)
        
        x_ar0 = xcoords[xcoords != 0]
        y_ar0 = ycoords[ycoords != 0]
        int_ar0 = intensities[intensities != 0]
        
        #x_ar02 = xcoords[xcoords > 0]
        x_ar02 = xcoords[ycoords > 0]
        #int_ar02 = np.zeros((len(x_ar02)))
        #for v in range(len(x_ar02)):
        #    int_ar02[v] = int_ar0[]
        
        #frms = np.array([i-start_frame for y in range(len(x_ar0))]) 
        frms = np.array([i-start_frame for y in range(len(x_ar02))]) 
        
        #im = ax.scatter(frms, x_ar0)
        im = ax.scatter(frms, x_ar02)
        canvas.blit(ax.bbox)
        plt.pause(0.001) # used for 1000 points, reasonable
        #plt.pause(0.1) # used for 1000 points, reasonable
        #plt.pause(0.5) # used for 1000 points, reasonable
    #"""
        
    ######################################### 
    

    