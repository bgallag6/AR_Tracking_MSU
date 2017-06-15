# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 08:47:15 2017

@author: Brendan
"""

"""
#################################################
### interactive tool so you can click ###########
### on a scatter point of duration/intensity  ###
### and displays AR life statistics #############
#################################################
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.idl import readsav
import jdcal

def onclick(event):
    global ix, iy, c, ind
    ixx, iyy = event.xdata, event.ydata
    ax2.clear()
    ax3.clear()
    ax4.clear()
    plt.draw()
    print ('x = %d, y = %d' % ( ixx, iyy))  # print location of pixel
    ix = int(ixx)
    iy = int(iyy)
    ind = 0
    
    ARs_copy = np.zeros_like((ARs))
    avg_int_copy = np.zeros_like((avg_int))
    frames_copy = np.zeros_like((frames))

    ARs_copy[:,0,:] = ARs[:,0,:]
    ARs_copy[:,1,:] = ARs[:,1,:]
    ARs_copy[:,2,:] = ARs[:,2,:]
    ARs_copy[:,3,:] = ARs[:,3,:]
    avg_int_copy[:] = avg_int[:]
    frames_copy[:] = frames[:]
    
    for q in range(count):
        if avg_int_copy[q] >= ix-2 and avg_int_copy[q] <= ix+2 and frames_copy[q] >= iy-2 and frames_copy[q] <= iy+2:         
            ind = q
            break
   
    ar0 = ARs_copy[ind,:,:]
    
    x_coords = ar0[1][ar0[1] != 0]
    y_coords = ar0[2][ar0[2] != 0]
    int_ar0 = ar0[0][ar0[0] != 0]
    area_ar0 = ar0[3][ar0[3] != 0]
    
    num_frm = int(frames_copy[q])
    color_copy = np.zeros((num_frm,3))
    for t in range(num_frm):
        color_copy[t][0] = t*(1./num_frm)
        color_copy[t][2] = 1. - t*(1./num_frm)
        
    ax2.plot(area_ar0)
    ax2.set_ylabel('Total Area', fontsize=font_size)
    ax2.set_xlabel('Frame', fontsize=font_size)
    
    ax3.plot(int_ar0)
    ax3.set_ylabel('Total Intensity', fontsize=font_size)
    ax3.set_xlabel('Frame', fontsize=font_size)
    
    ax4.set_ylabel('Latitude', fontsize=font_size)
    ax4.set_xlabel('Longitude', fontsize=font_size)
    ax4.scatter(x_coords, y_coords, int_ar0, c=color_copy)
    ax4.plot(x_coords, y_coords)
    
    plt.draw()
    return ix,iy


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
areas = np.array(all_tot_area1[0])

images = 1000
num_ar = 1000

global ARs, count

ARs = np.zeros((num_ar,4,images))
count = 0

for k in range(100):
    if intensities[k] > 0:
        ARs[count,0,0] = intensities[k]
        ARs[count,1,0] = xcoords[k]
        ARs[count,2,0] = ycoords[k]
        ARs[count,3,0] = areas[k]
        count += 1

start = dates[0]
dt_begin = 2400000.5
dt_dif1 = start-dt_begin    
dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)

color = np.zeros((images,3))
for t in range(images):
    color[t][0] = t*(1./images)
    color[t][2] = 1. - t*(1./images)
  
for i in range(1,images):
    
    start = dates[0] + i
    
    dt_begin = 2400000.5
    dt_dif1 = start-dt_begin    
    dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)
    
    intensities = np.array(all_tot_int1[i])
    intensities = [0 if x < 35 else x for x in intensities]  # maybe also eliminate zeros
    xcoords = np.array(all_cen_coords[i])[:,0]
    ycoords = np.array(all_cen_coords[i])[:,1]
    areas = np.array(all_tot_area1[i])
    
    num_reg = n_regions[i]
    for k in range(num_reg):  # if got rid zeros, just range(intensities.size)
        if intensities[k] > 0:
            found = 0
            for c in range(count):
                dr = np.sqrt((ARs[c,1,(i-1)]-xcoords[k])**2 + (ARs[c,2,(i-1)]-ycoords[k])**2)
                if dr < 5:
                    ARs[c,0,i] = intensities[k]
                    ARs[c,1,i] = xcoords[k]
                    ARs[c,2,i] = ycoords[k]
                    ARs[c,3,i] = areas[k]
                    found = 1  # maybe need to say closest one if more than one found?  don't think should be issue
            if found == 0:
                ARs[count,0,i] = intensities[k]
                ARs[count,1,i] = xcoords[k]
                ARs[count,2,i] = ycoords[k]
                ARs[count,3,i] = areas[k]
                count += 1
                
global frames, avg_int, area
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
area = np.zeros((count))

for c in range(count): 

    ar0 = ARs[c,:,:]
    
    frames[c] = np.count_nonzero(ar0,axis=1)[0] # - how many frames AR lasts
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
    
    area = ar0[3][ar0[3] != 0]
    
    dist_steps = np.zeros((int(frames[c]-1)))    
    
    for r in range(int(frames[c])-1):
        dist_steps[r] = np.sqrt((x_coords[r+1]-x_coords[r])**2 + (y_coords[r+1]-y_coords[r])**2)
    distance[c] = np.sum(dist_steps)



if 1:
    
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 17
    
    fig = plt.figure(figsize=(22,11))
    plt.suptitle('Active Region Statistics Summary \n Frames: 0 through %i' % images, fontsize=23, y=0.97)
    ax1 = plt.gca()
    ax1 = plt.subplot2grid((11,11),(0, 0), colspan=5, rowspan=5)
    ax1.set_ylabel('Duration', fontsize=font_size)
    ax1.set_xlabel('Average Intensity', fontsize=font_size)
    coll = ax1.scatter(avg_int, frames, picker = 5)
    
    ax2 = plt.subplot2grid((11,11),(6, 0), colspan=5, rowspan=5)
    ax2 = plt.gca()
    ax2.plot(0, 0)
    ax2.set_ylabel('Total Area', fontsize=font_size)
    ax2.set_xlabel('Frame', fontsize=font_size)
    
    ax3 = plt.subplot2grid((11,11),(6, 6), colspan=5, rowspan=5)
    ax3 = plt.gca()
    ax3.plot(0, 0)
    ax3.set_ylabel('Total Intensity', fontsize=font_size)
    ax3.set_xlabel('Frame', fontsize=font_size)
    
    ax4 = plt.subplot2grid((11,11),(0, 6), colspan=5, rowspan=5)
    ax4.plot(0)
    ax4.set_ylabel('Latitude', fontsize=font_size)
    ax4.set_xlabel('Longitude', fontsize=font_size)
    
    fig.canvas.mpl_connect('button_press_event', onclick)

plt.draw()

