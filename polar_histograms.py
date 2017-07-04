# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 20:23:33 2017

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
from matplotlib import cm

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
    
ARs = np.load('C:/Users/Brendan/Desktop/AR_bands_S.npy')
#ARs = AR_total
for i in range(500):
    if ARs[i,0,0] == 0:
        count = i
        break
    
num_bands = np.load('C:/Users/Brendan/Desktop/num_bands_S.npy')
#num_bands = num_bands

number = 0

rot_start = 0
rot_end = 18
    
#for c in range(rot_start,rot_end):
for c in range(3):    
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
    
    y3 = y1/(np.std(y1))
    y4 = [0 if x < 2. else x for x in y3]
        
    
    
    
    
    for u in range(len(norm1)):
        if u != max1:
            norm1[u] = 0
            
        if u != max2:
            norm2[u] = 0
    
        
    fig = plt.figure(figsize=(15,17))
    ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.75], polar=True)
    ax1.set_title(r'Southern Hemisphere' + '\n 3x Rotation Periods: %i-%i' % ((c*3)+1,((c+1)*3)), y=1.08, fontweight='bold', fontsize=font_size) 
    #ax1 = plt.subplot(111, projection='polar')
    N = num_bins
    theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
    radii = y4
    width = [np.pi/6 for z in range(12)]
    bars = ax1.bar(theta, radii, width=width, bottom=1.5)
    ax1.set_ylim(0,4)
    #ax1.set_ylabel('Sigma')
    for r,bar in zip(radii, bars):
        if r >= 2. and r < 3.:
            bar.set_facecolor('orange')
        elif r >= 3:
            bar.set_facecolor('red')
        #bar.set_facecolor( cm.jet(r/3.))
        bar.set_alpha(0.7)
    
    #plt.savefig('C:/Users/Brendan/Desktop/polar_hist/3x_Car_Rot_%i_%i_South_Polar_sigma.jpeg' % ((c*3)+1,((c+1)*3)), bbox_inches = 'tight')



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