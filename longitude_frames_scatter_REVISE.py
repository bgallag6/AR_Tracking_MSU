# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:29:51 2017

@author: Brendan
"""

"""
###############################################
### based on number of frames #################
### - shows full animated scatter  ############
### - with north / south hemisphere ###########
### - plus manually set AR fit + intensity ####
###############################################
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
import jdcal

"""
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

all_coords = coord.tolist()  # x1, x2, y1, y2
all_cen_coords = cen_coord.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()

images = 700
num_ar = 700
start_frame = 100
int_thresh = 30

#ARs = np.zeros((num_ar,3,images))
count = 0

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
#ax.set_xlim(0,300)
ax.set_xlim(0,images)
ax.set_ylim(0,360)
background = fig.canvas.copy_from_bbox(ax.bbox)
plt.ion()

int_tot = []
intN_tot = []
intS_tot = []
x_tot = []
xN_tot = []
xS_tot = []
y_tot = []
yN_tot = []
yS_tot = []
frm_tot = []
frmN_tot = []
frmS_tot = []
xerr1_tot = []
xerr2_tot = []
xerr1N_tot = []
xerr1S_tot = []
xerr2N_tot = []
xerr2S_tot = []

for i in range(start_frame,start_frame+images):
    
    start = f_names[i][0:8]

    date_title = '%s/%s/%s' % (start[0:4],start[4:6],start[6:8])
    
    intensities0 = np.array(all_tot_int1[i])
    #intensities = [0 if x < int_thresh else x for x in intensities]  # maybe also eliminate zeros
    intensities = intensities0[intensities0 > int_thresh]    
    xcoords0 = np.array(all_cen_coords[i])[:,0]
    ycoords0 = np.array(all_cen_coords[i])[:,1]
    xerr1_0 = np.array(all_coords[i])[:,0]
    xerr2_0 = np.array(all_coords[i])[:,1]
    #for q in range(len(intensities)):
    #    if intensities[q] < int_thresh:
    #        xcoords[q] = 0
    #        ycoords[q] = 0
    xcoords = xcoords0[intensities0 > int_thresh]
    ycoords = ycoords0[intensities0 > int_thresh]
    
    xerr1 = xerr1_0[intensities0 > int_thresh]
    xerr2 = xerr2_0[intensities0 > int_thresh]
    
    
    
    #num_reg = n_regions[i]
    #for k in range(num_reg):  # if got rid zeros, just range(intensities.size)
    #    if intensities[k] > 0:
    #        found = 0
    #        for c in range(count):
    #            dr = np.sqrt((ARs[c,1,(i-start_frame-1)]-xcoords[k])**2 + (ARs[c,2,(i-start_frame-1)]-ycoords[k])**2)
    #            if dr < 5:
    #                ARs[c,0,i-start_frame] = intensities[k]
    #                ARs[c,1,i-start_frame] = xcoords[k]
    #                ARs[c,2,i-start_frame] = ycoords[k]
    #                found = 1  # maybe need to say closest one if more than one found?  don't think should be issue
    #        if found == 0:
    #            ARs[count,0,i-start_frame] = intensities[k]
    #            ARs[count,1,i-start_frame] = xcoords[k]
    #            ARs[count,2,i-start_frame] = ycoords[k]
    #            count += 1
    
    
    ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date: %s' % (date_title), y=1.01, fontweight='bold', fontsize=font_size)
    canvas.restore_region(background)
    
    #x_temp = xcoords[xcoords != 0]
    #y_temp = ycoords[ycoords != 0]
    #int_ar0 = intensities[intensities != 0]
    
    #x_ar02 = xcoords[xcoords > 0]
    xN_temp = xcoords[ycoords > 0]
    xS_temp = xcoords[ycoords < 0]
    yN_temp = ycoords[ycoords > 0]
    yS_temp = ycoords[ycoords < 0]
    
    xerr1N_temp = xerr1[ycoords > 0]
    xerr1S_temp = xerr1[ycoords < 0]
    xerr2N_temp = xerr2[ycoords > 0]
    xerr2S_temp = xerr2[ycoords < 0]
    
    intN_temp = intensities[ycoords > 0]
    intS_temp = intensities[ycoords < 0]
    #int_ar02 = np.zeros((len(x_ar02)))
    #for v in range(len(x_ar02)):
    #    int_ar02[v] = int_ar0[]
    
    frm_temp = np.array([i-start_frame for y in range(len(xcoords))]) 
    frmN_temp = np.array([i-start_frame for y in range(len(xN_temp))]) 
    frmS_temp = np.array([i-start_frame for y in range(len(xS_temp))])
    
    
    int_tot = np.append(int_tot, intensities)
    intN_tot = np.append(intN_tot, intN_temp)
    intS_tot = np.append(intS_tot, intS_temp)
    x_tot = np.append(x_tot, xcoords)
    xN_tot = np.append(xN_tot, xN_temp)
    xS_tot = np.append(xS_tot, xS_temp)
    y_tot = np.append(y_tot, ycoords)
    yN_tot = np.append(yN_tot, yN_temp)
    yS_tot = np.append(yS_tot, yS_temp)
    frm_tot = np.append(frm_tot, frm_temp)
    frmN_tot = np.append(frmN_tot, frmN_temp)
    frmS_tot = np.append(frmS_tot, frmS_temp)
    
    xerr1_tot = np.append(xerr1_tot, xerr1)
    xerr2_tot = np.append(xerr2_tot, xerr2)
    xerr1N_tot = np.append(xerr1N_tot, xerr1N_temp)
    xerr1S_tot = np.append(xerr1S_tot, xerr1S_temp)
    xerr2N_tot = np.append(xerr2N_tot, xerr2N_temp)
    xerr2S_tot = np.append(xerr2S_tot, xerr2S_temp)
    
    
    #im = ax.scatter(frm_temp, xcoords, c=color[i-start_frame])
    im = ax.scatter(frm_temp, xcoords, int_tot)
    #im = ax.scatter(frm_temp, xcoords)
    #im = ax.scatter(frms, x_ar0)
    canvas.blit(ax.bbox)
    #plt.pause(0.001) # used for 1000 points, reasonable
    #plt.pause(0.1) # used for 1000 points, reasonable
    #plt.pause(0.5) # used for 1000 points, reasonable
"""



"""  
### plot North / South Hemispheres scatter

date_start = f_names[start_frame][0:8]
date_end = f_names[start_frame+images][0:8]

fig = plt.figure(figsize=(22,11))
ax = plt.gca()
ax.set_title(r'Our Data [Northern Hemisphere] | Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Longitude', fontsize=font_size)
ax.set_xlabel('Frame', fontsize=font_size)
#ax.set_xlim(0,300)
ax.set_xlim(0,images)
ax.set_ylim(0,360)  
im = ax.scatter(frmN_tot, xN_tot, c=color[i-start_frame])  

fig = plt.figure(figsize=(22,11))
ax = plt.gca()
ax.set_title(r'Our Data [Southern Hemisphere] | Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Longitude', fontsize=font_size)
ax.set_xlabel('Frame', fontsize=font_size)
#ax.set_xlim(0,300)
ax.set_xlim(0,images)
ax.set_ylim(0,360)  
im = ax.scatter(frmS_tot, xS_tot, c=color[i-start_frame])  
"""


#"""  
####### plot / fit specific AR ##########
AR_xs = []
AR_ys = []
AR_fs = []
AR_int = []
AR_xerr1 = []
AR_xerr2 = []
AR_int_sum = []
AR_fs_single = []


#"""
######### Northern ############
for a in range(len(frmN_tot)):
#for a in range(100):
    if frmN_tot[a] > 336 and frmN_tot[a] < 701:
        if xN_tot[a] > 300 and xN_tot[a] < 361:
            AR_xs = np.append(AR_xs, xN_tot[a])
            AR_ys = np.append(AR_ys, yN_tot[a])
            AR_fs = np.append(AR_fs, frmN_tot[a])
            AR_xerr1 = np.append(AR_xerr1, (xN_tot[a] - (xerr1N_tot[a]/10.)))
            AR_xerr2 = np.append(AR_xerr2, ((xerr2N_tot[a]/10.)-xN_tot[a]))
            AR_int = np.append(AR_int, intN_tot[a])
#"""


"""        
######### Southern ############
for a in range(len(frmS_tot)):
#for a in range(100):
    if frmS_tot[a] > 200 and frmS_tot[a] < 283:
        if xS_tot[a] > 237 and xS_tot[a] < 273:
            AR_xs = np.append(AR_xs, xS_tot[a])
            AR_ys = np.append(AR_ys, yS_tot[a])
            AR_fs = np.append(AR_fs, frmS_tot[a])
            AR_xerr1 = np.append(AR_xerr1, (xS_tot[a] - (xerr1S_tot[a]/10.)))
            AR_xerr2 = np.append(AR_xerr2, ((xerr2S_tot[a]/10.)-xS_tot[a]))
            AR_int = np.append(AR_int, intS_tot[a])
"""           
            
            
m, b = np.polyfit(AR_fs, AR_xs, 1)

date_start = f_names[int(AR_fs[0]+start_frame)]
date_end = f_names[int(AR_fs[-1]+start_frame)]

date_start = '%s/%s/%s' % (date_start[0:4],date_start[4:6],date_start[6:8])
date_end = '%s/%s/%s' % (date_end[0:4],date_end[4:6],date_end[6:8])

       
fig = plt.figure(figsize=(22,11))
ax = plt.gca()
ax.set_title(r'Active Region # | Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Longitude', fontsize=font_size)
ax.set_xlabel('Frame', fontsize=font_size)
ax.set_xlim(AR_fs[0]-3,AR_fs[-1]+3)
ax.set_ylim(np.min(AR_xs-AR_xerr1)-5,np.max(AR_xs+AR_xerr1)+5)  
im = ax.scatter(AR_fs, AR_xs, AR_int)  
ax.errorbar(AR_fs, AR_xs, yerr=[AR_xerr1, AR_xerr2], fmt='--')
ax.plot(AR_fs, m*AR_fs + b, '-')

fig = plt.figure(figsize=(22,11))
ax = plt.gca()
ax.set_title(r'Active Region # | Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Latitude', fontsize=font_size)
ax.set_xlabel('Frame', fontsize=font_size)
ax.set_xlim(AR_fs[0]-3,AR_fs[-1]+3)
ax.set_ylim(np.min(AR_ys)-5,np.max(AR_ys)+5)  
im = ax.scatter(AR_fs, AR_ys, AR_int)  

count_fs = 0

for p in range(len(AR_fs)):
    if count_fs == 0:
        AR_int_sum = np.append(AR_int_sum, AR_int[p])
        AR_fs_single = np.append(AR_fs_single, AR_fs[p])
        count_fs += 1
    else:
        if AR_fs[p] != AR_fs[p-1]:
            AR_int_sum = np.append(AR_int_sum, AR_int[p])
            AR_fs_single = np.append(AR_fs_single, AR_fs[p])
            count_fs += 1
        else:
            AR_int_sum[count_fs-1] +=  AR_int[p]
            
    

fig = plt.figure(figsize=(22,11))
ax = plt.gca()
ax.set_title(r'Active Region # | Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Intensity', fontsize=font_size)
ax.set_xlabel('Frame', fontsize=font_size)
ax.set_xlim(AR_fs[0]-3,AR_fs[-1]+3)
#ax.plot(AR_fs, AR_int)
ax.plot(AR_fs_single, AR_int_sum)
#"""        