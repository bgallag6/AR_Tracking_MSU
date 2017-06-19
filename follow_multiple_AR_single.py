# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:01:18 2017

@author: Brendan
"""

"""
#################################
### specify start / end frame ###
### - shows animated scatter  ###
### - latitude / longitude    ###
### specify ARs within range..###
### get their combined life   ###
#################################
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
import jdcal

#"""
s = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)
f_names = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/ar_filenames.npy')

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
start_frame = 500
int_thresh = 30

intensities = np.array(all_tot_int1[start_frame])
#intensities = [0 if x < 35 else x for x in intensities]
intensities = [0 if x < int_thresh else x for x in intensities]
xcoords = np.array(all_cen_coords[start_frame])[:,0]
ycoords = np.array(all_cen_coords[start_frame])[:,1]

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
im = ax.scatter(xcoords, ycoords,intensities, c=color[0])
ax.set_xlim(0,360)
ax.set_ylim(-45,45)
background = fig.canvas.copy_from_bbox(ax.bbox)
plt.ion()


for i in range(start_frame+1,start_frame+images):
    
    start = f_names[i][0:8]
    
    #dt_begin = 2400000.5
    #dt_dif1 = start-dt_begin    
    #dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)
    date_title = '%s/%s/%s' % (start[0:4],start[4:6],start[6:8])
    
    intensities = np.array(all_tot_int1[i])
    #intensities = [0 if x < 35 else x for x in intensities]  # maybe also eliminate zeros
    intensities = [0 if x < int_thresh else x for x in intensities]  # maybe also eliminate zeros
    xcoords = np.array(all_cen_coords[i])[:,0]
    ycoords = np.array(all_cen_coords[i])[:,1]
    
    num_reg = n_regions[i]
    for k in range(num_reg):  # if got rid zeros, just range(intensities.size)
        if intensities[k] > 0:
            found = 0
            for c in range(count):
                dr = np.sqrt((ARs[c,1,(i-start_frame-1)]-xcoords[k])**2 + (ARs[c,2,(i-start_frame-1)]-ycoords[k])**2)
                #print "dr = %0.2f" % dr
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
    
    
    #ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date: %i/%i/%i' % (dt_greg1[0], dt_greg1[1], dt_greg1[2]), y=1.01, fontweight='bold', fontsize=font_size)
    ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date: %s' % (date_title), y=1.01, fontweight='bold', fontsize=font_size)
    canvas.restore_region(background)
        
    im = ax.scatter(xcoords, ycoords,intensities,c=color[i-start_frame])
    canvas.blit(ax.bbox)
    #plt.pause(0.001) # used for 1000 points, reasonable
    plt.pause(0.1) # used for 1000 points, reasonable
    #plt.pause(0.5) # used for 1000 points, reasonable
"""    
    
frames = np.zeros((count))
first = np.zeros((count))
xy_form = np.zeros((count,2))
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
    xy_form[c] = [ar0[1][ar0[1] != 0][0], ar0[2][ar0[2] != 0][0]]
    x_end[c] = ar0[1][ar0[1] != 0][-1]
    y_end[c] = ar0[2][ar0[2] != 0][-1]
    
    x_coords = ar0[1][ar0[1] != 0]
    y_coords = ar0[2][ar0[2] != 0]
    x_avg[c] = np.average(x_coords)
    y_avg[c] = np.average(y_coords)
    
    avg_int[c] = np.average(ar0[0][ar0[0] != 0])
    med_int[c] = np.median(ar0[0][ar0[0] != 0])            
"""

"""  
#########################
# select AR for summary #
#########################

ar = 3

ar0 = ARs[ar,:,:]
frames = np.count_nonzero(ar0,axis=1)[0] # - how many frames lasts
ar0sort = np.flipud(np.sort(ar0[1]))
int_ar0 = ar0[0][ar0[0] != 0]
x_ar0 = ar0[1][ar0[1] != 0]
y_ar0 = ar0[2][ar0[2] != 0]
x_range = np.max(x_ar0) - np.min(x_ar0)

for d in range(images):
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
"""

"""
##################################
# select mulitple AR for summary #
##################################
ar = [41,167,215,300, 328, 332, 369, 380, 387, 393, 403, 408, 409]

int_ar_tot = []
x_ar_tot = []
y_ar_tot = []
first = []
frames = []

for n in range(len(ar)):

    ar0 = ARs[ar[n],:,:]
    for d in range(images):
            if ar0[0,d] != 0:
                first = np.append(first, d)
                break
    frames = np.append(frames, np.count_nonzero(ar0,axis=1)[0]) # - how many frames lasts
    
    int_ar0 = ar0[0][ar0[0] != 0]
    int_ar_tot = np.append(int_ar_tot, int_ar0)
    
    x_ar0 = ar0[1][ar0[1] != 0]
    y_ar0 = ar0[2][ar0[2] != 0]
    x_ar_tot = np.append(x_ar_tot, x_ar0)
    y_ar_tot = np.append(y_ar_tot, y_ar0)
    
tot_frames = int((first[len(ar)-1]+frames[len(ar)-1]) - first[0])

int_frames = np.zeros((tot_frames))


for n in range(len(ar)):

    ar0 = ARs[ar[n],:,:]                
    
    #ar0sort = np.flipud(np.sort(ar0[1]))
    int_ar0 = ar0[0][ar0[0] != 0]
    
    for k in range(len(int_ar0)):
    #x_range = np.max(x_ar0) - np.min(x_ar0)
        int_frames[int(first[n])+k-int(first[0])] += int_ar0[k]

color = np.zeros((tot_frames,3))
for t in range(tot_frames):
    color[t][0] = t*(1./tot_frames)
    color[t][2] = 1. - t*(1./tot_frames)

days = np.linspace(0,tot_frames/2,tot_frames)

plt.rcParams["font.family"] = "Times New Roman"
font_size = 17

fig = plt.figure(figsize=(22,11))
plt.suptitle('Active Region #%i \n Frames %i through %i' % (11726,start_frame+first[0],start_frame+first[0]+tot_frames), fontsize=23, y=0.97)
#plt.suptitle('Random Region \n Frames %i through %i' % (first,first+frames), fontsize=23, y=0.97)

ax1 = plt.subplot2grid((11,11),(0, 0), colspan=7, rowspan=5)
ax1.set_ylabel('Latitude', fontsize=font_size)
ax1.set_xlabel('Longitude', fontsize=font_size)
im = ax1.scatter(x_ar_tot, y_ar_tot, int_ar_tot,c=color)
ax1.set_xlim(0,360)
ax1.set_ylim(-45,45)

ax2 = plt.subplot2grid((11,11),(6, 0), colspan=5, rowspan=5)
ax2.plot(x_ar_tot, y_ar_tot)
ax2.set_ylabel('Latitude', fontsize=font_size)
ax2.set_xlabel('Longitude', fontsize=font_size)
ax2.set_xlim(np.min(x_ar_tot)-3,np.max(x_ar_tot)+3)
ax2.set_ylim(np.min(y_ar_tot)-3,np.max(y_ar_tot)+3)

ax3 = plt.subplot2grid((11,11),(6, 6), colspan=5, rowspan=5)
#ax3.plot(int_ar_tot)
ax3.plot(days,int_frames)
#ax3.plot(days[0:53],int_frames[0:53])
ax3.set_ylabel('Total Intensity', fontsize=font_size)
ax3.set_xlabel('Days', fontsize=font_size)

ax4 = plt.subplot2grid((11,11),(0, 8), colspan=3, rowspan=5)
ax4.set_ylabel('Latitude', fontsize=font_size)
ax4.set_xlabel('Longitude', fontsize=font_size)
im = ax4.scatter(x_ar_tot, y_ar_tot, int_ar_tot,c=color)
ax4.plot(x_ar_tot, y_ar_tot)
ax4.set_xlim(np.min(x_ar_tot)-3,np.max(x_ar_tot)+3)
ax4.set_ylim(np.min(y_ar_tot)-3,np.max(y_ar_tot)+3)
"""

#AR_transpose = np.zeros((images,num_ar,3))
#for u in range(images):
#    AR_transpose[:,u,:] = np.transpose(ARs[u,:,:])

""" dont need
nAR = []
frameAR = []
 
for q in range(images):
    ar0 = AR_transpose[q,:,:]
    for d in range(num_ar):
        if ar0[d,1] >250 and ar0[d,1] < 300 and ar0[d,2] > 5 and ar0[d,2] < 23:
            nAR = np.append(nAR, d)
            frameAR = np.append(frameAR, q)
"""

