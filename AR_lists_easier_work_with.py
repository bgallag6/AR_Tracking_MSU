# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 20:19:16 2017

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

all_cen_coords = cen_coord.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()


intensities = np.array(all_tot_int1[0])
intensities = [0 if x < 35 else x for x in intensities]
xcoords = np.array(all_cen_coords[0])[:,0]
ycoords = np.array(all_cen_coords[0])[:,1]

images = 500
num_ar = 500

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
#"""

#"""
plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
ar = 127

#fig = plt.figure(figsize=(22,11))
ax = plt.gca()
#canvas = ax.figure.canvas
ax.set_ylabel('Latitude', fontsize=font_size)
ax.set_xlabel('Longitude', fontsize=font_size)
#plt.tick_params(axis='both', labelsize=font_size, pad=7)
im = ax.scatter(ARs[ar,1,:], ARs[ar,2,:], ARs[ar,0,:])
#im = ax.scatter(ARs[ar,1,0], ARs[ar,2,0], ARs[ar,0,0])
ax.set_xlim(0,360)
ax.set_ylim(-45,45)
plt.ion()
#"""

"""
fig = plt.figure(figsize=(22,11))
ax = plt.gca()
ax.set_xlim(0,360)
ax.set_ylim(-45,45)
for i in range(50,60):
    #canvas = ax.figure.canvas
    ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date: %i/%i/%i + %i' % (dt_greg1[0], dt_greg1[1], dt_greg1[2], i), y=1.01, fontweight='bold', fontsize=font_size)
    #plt.tick_params(axis='both', labelsize=font_size, pad=7)
    #im = ax.scatter(np.array(all_cen_coords[0])[:,0], np.array(all_cen_coords[0])[:,1],np.array(all_tot_int1[0]))
    #im = ax.scatter(np.array(all_cen_coords[0])[:,0], np.array(all_cen_coords[0])[:,1],intensities)
    im = ax.scatter(ARs[ar,1,i], ARs[ar,2,i], ARs[ar,0,i])
    
    plt.pause(0.1) # used for 1000 points, reasonable
"""

xbins = [3*i for i in range(120)]

"""
histx = np.zeros((count))

for c in range(count):
    for i in range(images):
        if ARs[c,0,i] > 0:
            cc = ARs[c,:,i]
            histx[c] = cc[1]
            #print "frame = %i | int = %0.2f | x = %0.2f | y = %0.2f" % (i,cc[0],cc[1],cc[2])
            #im = ax.scatter(cc[1],cc[2],cc[0])
            #im = ax.scatter(cc[1],cc[2],i*3)
            break

plt.hist(histx, bins=xbins)
"""

"""
histx = np.zeros((sum(n_regions[:1000])))
histx = np.zeros((sum(n_regions[:1000])))
nAR = 0

for c in range(1000):
    numberAR = n_regions[c]
     
    for i in range(numberAR):
        if all_tot_int1[c][i] > 35:
            histx[nAR+i] = all_cen_coords[c][]
        
    nAR += numberAR
        
plt.hist(histx, bins=xbins)
"""



