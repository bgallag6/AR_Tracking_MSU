# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 20:55:38 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.idl import readsav
import jdcal

#"""
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

#all_xcoords = [0 for k in range(trim)]
#all_ycoords = [0 for k in range(trim)]
#all_med_inten = [0 for k in range(trim)]
#all_tot_int1 = [0 for k in range(trim)]
#all_tot_area1 = [0 for k in range(trim)]
#total_intensity = [0 for k in range(trim)]
#all_scaled_intensity = [0 for k in range(trim)]

#long_scaled_intensity = np.zeros((18))

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

### need to add time component to AR array? - or just put in which image it is

for i in range(1,images):
    
    start = dates[0] + i
    
    dt_begin = 2400000.5
    dt_dif1 = start-dt_begin    
    dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)
    
    #all_tot_int1 = [0 if x < 35 else x for x in all_tot_int1]
    intensities = np.array(all_tot_int1[i])
    intensities = [0 if x < 35 else x for x in intensities]  # maybe also eliminate zeros
    xcoords = np.array(all_cen_coords[i])[:,0]
    ycoords = np.array(all_cen_coords[i])[:,1]
    
    #print "int = ", intensities[:n_regions[i]]
    #print "xcoords = ", xcoords[:n_regions[i]]
    #print "ycoords = ", ycoords[:n_regions[i]]
    
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
    
    
    """
    #canvas = ax.figure.canvas
    ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date: %i/%i/%i' % (dt_greg1[0], dt_greg1[1], dt_greg1[2]), y=1.01, fontweight='bold', fontsize=font_size)
    #plt.tick_params(axis='both', labelsize=font_size, pad=7)
    #im = ax.scatter(np.array(all_cen_coords[0])[:,0], np.array(all_cen_coords[0])[:,1],np.array(all_tot_int1[0]))
    #im = ax.scatter(np.array(all_cen_coords[0])[:,0], np.array(all_cen_coords[0])[:,1],intensities)
    canvas.restore_region(background)
        
    im = ax.scatter(xcoords, ycoords,intensities,c=color[i])
    canvas.blit(ax.bbox)
    plt.pause(0.001) # used for 1000 points, reasonable
    #plt.pause(0.1) # used for 1000 points, reasonable
    #plt.pause(0.5) # used for 1000 points, reasonable
    """
#"""

"""

###############################################################################
###############################################################################

#########################
# select AR for summary #
#########################

ar = 119

ar0 = ARs[ar,:,:]
frames = np.count_nonzero(ar0,axis=1)[0] # - how many frames lasts
ar0sort = np.flipud(np.sort(ar0[1]))
int_ar0 = ar0[0][ar0[0] != 0]
x_ar0 = ar0[1][ar0[1] != 0]
y_ar0 = ar0[2][ar0[2] != 0]
x_range = np.max(x_ar0) - np.min(x_ar0)

for d in range(500):
    if ar0[0,d] != 0:
        first = d

color = np.zeros((frames,3))
for t in range(frames):
    color[t][0] = t*(1./frames)
    color[t][2] = 1. - t*(1./frames)


plt.rcParams["font.family"] = "Times New Roman"
font_size = 17

fig = plt.figure(figsize=(22,11))
plt.suptitle('Active Region #%i \n Frames %i through %i' % (ar,first,first+frames), fontsize=23, y=0.97)

ax1 = plt.subplot2grid((11,11),(0, 0), colspan=7, rowspan=5)
ax1.set_ylabel('Latitude', fontsize=font_size)
ax1.set_xlabel('Longitude', fontsize=font_size)
im = ax1.scatter(ARs[ar,1,:], ARs[ar,2,:], ARs[ar,0,:],c=color)
ax1.set_xlim(0,360)
ax1.set_ylim(-45,45)

ax2 = plt.subplot2grid((11,11),(6, 0), colspan=5, rowspan=5)
ax2.plot(x_ar0, y_ar0)
ax2.set_ylabel('Latitude', fontsize=font_size)
ax2.set_xlabel('Longitude', fontsize=font_size)

ax3 = plt.subplot2grid((11,11),(6, 6), colspan=5, rowspan=5)
ax3.plot(int_ar0)
ax3.set_ylabel('Total Intensity', fontsize=font_size)
ax3.set_xlabel('Frame', fontsize=font_size)

ax4 = plt.subplot2grid((11,11),(0, 8), colspan=3, rowspan=5)
ax4.set_ylabel('Latitude', fontsize=font_size)
ax4.set_xlabel('Longitude', fontsize=font_size)
im = ax4.scatter(ARs[ar,1,:], ARs[ar,2,:], ARs[ar,0,:],c=color)
ax4.plot(x_ar0, y_ar0)
ax4.set_xlim(np.min(x_ar0)-3,np.max(x_ar0)+3)
ax4.set_ylim(np.min(y_ar0)-3,np.max(y_ar0)+3)

#plt.savefig('C:/Users/Brendan/Desktop/ar%i.jpeg' % ar)

###############################################################################
###############################################################################
"""

xbins = [3*i for i in range(120)]

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
    
    dist_steps = np.zeros((int(frames[c]-1)))    
    
    for r in range(int(frames[c])-1):
        dist_steps[r] = np.sqrt((x_coords[r+1]-x_coords[r])**2 + (y_coords[r+1]-y_coords[r])**2)
    distance[c] = np.sum(dist_steps)
        
    
plt.rcParams["font.family"] = "Times New Roman"
font_size = 17

fig = plt.figure(figsize=(22,11))
plt.suptitle('Active Region Statistics Summary \n Frames: 0 through 100', fontsize=23, y=0.97)

ax1 = plt.subplot2grid((11,11),(0, 0), colspan=5, rowspan=5)
ax1.set_ylabel('Duration', fontsize=font_size)
ax1.set_xlabel('Average Intensity', fontsize=font_size)
ax1.scatter(avg_int,frames)

ax2 = plt.subplot2grid((11,11),(6, 0), colspan=5, rowspan=5)
ax2.set_ylabel('Duration', fontsize=font_size)
ax2.set_xlabel('Median Intensity', fontsize=font_size)
ax2.scatter(med_int,frames)

ax3 = plt.subplot2grid((11,11),(6, 6), colspan=5, rowspan=5)
ax3.set_ylabel('Formation Longitude', fontsize=font_size)
ax3.set_xlabel('Frame Formed', fontsize=font_size)
ax3.scatter(first, x_form)

ax4 = plt.subplot2grid((11,11),(0, 6), colspan=5, rowspan=5)
ax4.set_ylabel('Distance Traveled', fontsize=font_size)
ax4.set_xlabel('Average Intensity', fontsize=font_size)
ax4.scatter(avg_int,distance)

#plt.savefig('C:/Users/Brendan/Desktop/AR_Statistics_Summary_100.jpeg')