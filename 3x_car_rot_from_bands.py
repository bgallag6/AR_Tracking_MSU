# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 18:16:34 2017

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

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
    
ARs = np.load('C:/Users/Brendan/Desktop/AR_bands_N.npy')
#ARs = AR_total
for i in range(500):
    if ARs[i,0,0] == 0:
        count = i
        break
    
num_bands = np.load('C:/Users/Brendan/Desktop/num_bands_N.npy')
#num_bands = num_bands

number = 0

rot_start = 0
rot_end = 18
    
for c in range(rot_start,rot_end):
#for c in range(3):    
    #date_start = f_names[int(ind_start[c])][0:8]
    #date_end = f_names[int(ind_end[c])][0:8]
    
    #date_start = '%s/%s/%s' % (date_start[0:4],date_start[4:6],date_start[6:8])
    #date_end = '%s/%s/%s' % (date_end[0:4],date_end[4:6],date_end[6:8])
       
    int_tot = []
    x_tot = []
    frm_tot = []
    first_tot = []
    
    for i in range(int(num_bands[c])):
    
        intensities = np.array(ARs[i+number,2,:])
        int_temp = intensities[intensities != 0]
        
        xcoords = np.array(ARs[i+number,1,:])
        x_temp = xcoords[xcoords != 0]
        
        frm = np.array(ARs[i+number,0,:]) 
        frm_temp = frm[frm != 0]
        
        int_tot = np.append(int_tot, int_temp)
        
        x_tot = np.append(x_tot, x_temp)
        
        frm_tot = np.append(frm_tot, frm_temp) 
        
        first_tot = np.append(first_tot, x_temp[0])
        
    number += int(num_bands[c])
    
    deg = 30
    num_bins = 360/deg
    
    x_bins = [deg*l for l in range(num_bins+1)]
    x_bins2 = [deg*l for l in range(num_bins)]
    
    x_ticks = np.array(x_bins) + (deg/2)
    
    plt.figure()
    y1, x1, _ = plt.hist(x_tot, bins=x_bins)
    #y1, x1, _ = plt.hist(first_tot, bins=x_bins)
    plt.close()
    #elem1 = np.argmax(y1)
    #bin_max1 = y1[elem1]
    norm1 = y1/np.sum(y1)
    norm2 = np.copy(norm1)
    y2 = np.sort(y1)
    max1 = y1.tolist().index(y2[-1])
    max2 = y1.tolist().index(y2[-2])
    
    for u in range(len(norm1)):
        if u != max1:
            norm1[u] = 0
            
        if u != max2:
            norm2[u] = 0
    
        
        
        #y2, x2, _ = plt.hist(xS_temp, bins=x_bins)
        #elem2 = np.argmax(y2)
        #bin_max2 = y2[elem2]
        #plt.close()
        
        #bin_max = np.max([bin_max1, bin_max2])*1.1   
        #"""  ### plot North / South Hemispheres scatter
    #"""
        
    y_lim = 0.25
    
    if c == rot_start:
        fig = plt.figure(figsize=(10,22))
        plt.suptitle(r'Southern Hemisphere' + '\n 3x Rotation Periods: %i-%i' % (rot_start,rot_end), y=0.97, fontweight='bold', fontsize=font_size) 
        #plt.suptitle(r'Northern Hemisphere' + '\n 7x Rotation Periods: %i-%i' % (rot_start,rot_end), y=0.97, fontweight='bold', fontsize=font_size) 
        ax1 = plt.subplot2grid((18,1),(c-rot_start, 0), colspan=1, rowspan=1)
        ax1 = plt.gca()         
        ax1.set_ylabel('%i' % (c+1), fontsize=font_size)
        ax1.set_xlim(0,360)   
        ax1.set_ylim(0,y_lim)
        #ax1.hist(norm1, bins=x_bins) 
        #ax1.plot(x_bins2, norm1)
        ax1.bar(x_bins2, norm1, width=deg/3)
        #ax1.bar(x_bins2, norm2, width=deg/3, color='black')
    elif c == rot_end-1: 
        ax1 = plt.subplot2grid((18,1),(c-rot_start, 0), colspan=1, rowspan=1, sharey=ax1)
        ax1 = plt.gca()      
        ax1.set_xlim(0,360) 
        ax1.set_ylim(0,y_lim)
        ax1.set_ylabel('%i' % (c+1), fontsize=font_size)
        ax1.set_xlabel('Longitude', fontsize=font_size)
        plt.xticks(x_bins)
        #ax1.plot(x_bins2, norm1)
        ax1.bar(x_bins2, norm1, width=deg/3)
        #ax1.bar(x_bins2, norm2, width=deg/3, color='black')
    else:
        ax1 = plt.subplot2grid((18,1),(c-rot_start, 0), colspan=1, rowspan=1, sharex=ax1, sharey=ax1)
        ax1 = plt.gca()    
        ax1.set_xlim(0,360)  
        ax1.set_ylim(0,y_lim)
        ax1.set_ylabel('%i' % (c+1), fontsize=font_size)
        ax1.set_xticklabels([])
        #ax1.hist(norm1, bins=x_bins) 
        ax1.bar(x_bins2, norm1, width=deg/3)
        #ax1.bar(x_bins2, norm2, width=deg/3, color='black')
        #ax1.plot(x_bins2, norm1)
    #"""
#plt.savefig('C:/Users/Brendan/Desktop/3x_Car_Rot_%i_%i_North_Emerge_Bands_Int5.pdf' % (rot_start,rot_end), bbox_inches = 'tight')
#plt.savefig('C:/Users/Brendan/Desktop/3x_Car_Rot_%i_%i_South_%ideg.pdf' % (rot_start,rot_end,deg), bbox_inches = 'tight')
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