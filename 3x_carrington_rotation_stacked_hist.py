# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:13:08 2017

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

int_thresh = 30

count = 0

rotations = 3
seg = ((dates[trim]-dates[11])/27.25)/rotations
  
ind_start = np.zeros((int(seg)))
ind_end = np.zeros((int(seg)))

for i in range(int(seg)):
#for i in range(3):
    
    start = dates[11] + ((27.25*i)*rotations)
    end = start + (27.25*rotations)
    
    dt_begin = 2400000.5
    dt_dif1 = start-dt_begin   
    dt_dif2 = (start+27)-dt_begin  
    dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)
    dt_greg2 = jdcal.jd2gcal(dt_begin,dt_dif2)
    
    ind_start[i] = np.searchsorted(dates,start)  # dont' think this is exactly correct, but close?
    ind_end[i] = np.searchsorted(dates,end)

rot_start = 0
rot_end = 18
#for c in range(int(seg)):
for c in range(rot_start,rot_end):
    
    date_start = f_names[int(ind_start[c])][0:8]
    date_end = f_names[int(ind_end[c])][0:8]
    
    date_start = '%s/%s/%s' % (date_start[0:4],date_start[4:6],date_start[6:8])
    date_end = '%s/%s/%s' % (date_end[0:4],date_end[4:6],date_end[6:8])
    
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 23
    
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
    
    for i in range(int(ind_start[c]),int(ind_end[c])):
    
        intensities0 = np.array(all_tot_int1[i])
        intensities = intensities0[intensities0 > int_thresh] 
        
        xcoords0 = np.array(all_cen_coords[i])[:,0]
        ycoords0 = np.array(all_cen_coords[i])[:,1]
        
        xcoords = xcoords0[intensities0 > int_thresh]
        ycoords = ycoords0[intensities0 > int_thresh]
        
        xN_temp = xcoords[ycoords > 0]
        xS_temp = xcoords[ycoords < 0]
        intN_temp = intensities[ycoords > 0]
        intS_temp = intensities[ycoords < 0]
        
        #frm_temp = np.array([i-start_frame for y in range(len(xcoords))]) 
        #frmN_temp = np.array([i-start_frame for y in range(len(xN_temp))])
        #frmS_temp = np.array([i-start_frame for y in range(len(xS_temp))])
        frm_temp = np.array([i for y in range(len(xcoords))]) 
        frmN_temp = np.array([i for y in range(len(xN_temp))])
        frmS_temp = np.array([i for y in range(len(xS_temp))])
        
        int_tot = np.append(int_tot, intensities)
        intN_tot = np.append(intN_tot, intN_temp)
        intS_tot = np.append(intS_tot, intS_temp)
        x_tot = np.append(x_tot, xcoords)
        xN_tot = np.append(xN_tot, xN_temp)
        xS_tot = np.append(xS_tot, xS_temp)
        #y_tot = np.append(y_tot, ycoords)
        #yN_tot = np.append(yN_tot, yN_temp)
        #yS_tot = np.append(yS_tot, yS_temp)
        frm_tot = np.append(frm_tot, frm_temp)
        frmN_tot = np.append(frmN_tot, frmN_temp)
        frmS_tot = np.append(frmS_tot, frmS_temp)

        #im = ax.scatter(frm_temp, xcoords)
        #canvas.blit(ax.bbox)
        #plt.pause(0.001) # used for 1000 points, reasonable
        #plt.pause(0.1) # used for 1000 points, reasonable
        #plt.pause(0.5) # used for 1000 points, reasonable
    
    
    #x_bins = [20*l for l in range(19)]
    #x_bins2 = [20*l for l in range(18)]
    x_bins = [2*l for l in range(181)]
    x_bins2 = [2*l for l in range(180)]
    x_ticks = [40*l for l in range(10)]
    
    plt.figure()
    y1, x1, _ = plt.hist(xN_tot, bins=x_bins)
    plt.close()
    #elem1 = np.argmax(y1)
    #bin_max1 = y1[elem1]
    norm1 = y1/np.sum(y1)
    #plt.figure()
    
        
        
        #y2, x2, _ = plt.hist(xS_temp, bins=x_bins)
        #elem2 = np.argmax(y2)
        #bin_max2 = y2[elem2]
        #plt.close()
        
        #bin_max = np.max([bin_max1, bin_max2])*1.1
        
        #"""  ### plot North / South Hemispheres scatter
    #"""
    if c == rot_start:
        fig = plt.figure(figsize=(10,22))
        plt.suptitle(r'Nothern Hemisphere' + '\n 3x Rotation Periods: %i-%i' % (rot_start,rot_end), y=0.97, fontweight='bold', fontsize=font_size) 
        ax1 = plt.subplot2grid((18,1),(c-rot_start, 0), colspan=1, rowspan=1)
        ax1 = plt.gca()         
        ax1.set_ylabel('%i' % (c+1), fontsize=font_size)
        ax1.set_xlim(0,360)   
        #ax1.hist(norm1, bins=x_bins) 
        #ax1.plot(x_bins2, norm1)
        ax1.bar(x_bins2, norm1)
    elif c == rot_end-1: 
        ax1 = plt.subplot2grid((18,1),(c-rot_start, 0), colspan=1, rowspan=1, sharey=ax1)
        ax1 = plt.gca()    
        ax1.set_xlim(0,360) 
        ax1.set_ylabel('%i' % (c+1), fontsize=font_size)
        ax1.set_xlabel('Longitude', fontsize=font_size)
        #ax1.plot(x_bins2, norm1)
        ax1.bar(x_bins2, norm1)
    else:
        ax1 = plt.subplot2grid((18,1),(c-rot_start, 0), colspan=1, rowspan=1, sharex=ax1, sharey=ax1)
        ax1 = plt.gca()    
        ax1.set_xlim(0,360)   
        ax1.set_ylabel('%i' % (c+1), fontsize=font_size)
        ax1.set_xticklabels([])
        #ax1.hist(norm1, bins=x_bins) 
        ax1.bar(x_bins2, norm1)
        #ax1.plot(x_bins2, norm1)
    #"""
    #plt.savefig('C:/Users/Brendan/Desktop/3x_Car_Rot_%i_%i_North.pdf' % (rot_start,rot_end), bbox_inches = 'tight')
    #plt.close()
    
    """
    fig = plt.figure(figsize=(22,10))
    plt.suptitle(r'Southern Hemisphere - Carrington Rotation Periods: %i - %i' % ((c*rotations)+1, ((c+1)*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=0.97, fontweight='bold', fontsize=font_size)
    ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
    ax1 = plt.gca()
    
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Frame', fontsize=font_size)
    ax1.set_xlim(ind_start[c],ind_end[c])
    ax1.set_ylim(0,360)  
    ax1.scatter(frmS_tot, xS_tot)  
    
    ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
    ax2 = plt.gca()
    ax2.set_ylabel('Number of ARs', fontsize=font_size)
    ax2.set_xlabel('Longitude', fontsize=font_size)
    ax2.set_ylim(0,bin_max)  
    ax2.set_xlim(0,360)
    ax2.hist(xS_tot, bins=x_bins)  
    plt.xticks(x_ticks)
    plt.savefig('C:/Users/Brendan/Desktop/Car_Rot_%i_%i_South.jpg' % ((c*rotations)+1, ((c+1)*rotations)), bbox_inches = 'tight')
    plt.close()
    """    
    #"""