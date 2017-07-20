# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:02:35 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib


plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
deg = 15
num_bins = 360/deg

fStart = [11,417,1115,1825,2511]
fEnd = [416,1114,1824,2510,2710]

x_bins = [deg*l for l in range(num_bins+1)]
x_bins2 = [deg*l for l in range(num_bins)]

x_ticks = np.array(x_bins) + (deg/2)

#hemi = 'N'
hemi = 'S'
smooth_x = 5  #  5, 6, 8, 10
smooth_y = 2  #  2, 3, 4, 5

AL_thresh = 8


if hemi == 'N':
    hemiF = 'North'
elif hemi == 'S':
    hemiF = 'South'
   
#num_bands = np.load('C:/Users/Brendan/Desktop/MSU_Project/num_bands_S.npy')
num_bands = np.load('C:/Users/Brendan/Desktop/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, smooth_x, smooth_y))
#num_bands = num_bands
    
#ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S.npy')
ARs = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, smooth_x, smooth_y))
#ARs = AR_total

#fit_params = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_slopes.npy')
fit_params = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_slopes_%s_3x_30int_%sx%sysmooth.npy' % (hemi, smooth_x, smooth_y))

AL_bins = np.load('C:/Users/Brendan/Desktop/3x_%s_3sigma_combined.npy' % hemiF)

for i in range(500):
    if ARs[i,0,0] == 0:
        count = i
        break
    
number = 0

rot_start = 0
rot_end = 18

AL_lat = []
AL_slopes = []

med_lat_tot = []
slopes_tot = []   
    
count_tot = 0

for c in range(rot_start,rot_end):
#for c in range(3):    
    
    count = 0
    for i in range(int(num_bands[c])):
        
        intensity0 = np.array(ARs[i+number,2,:])
        frames0 = np.array(ARs[i+number,0,:])
        int_temp = intensity0[intensity0 != 0]
        frm_temp = frames0[intensity0 != 0]
        
        ycoords = np.array(ARs[i+number,3,:])
        
        xcoords = np.array(ARs[i+number,1,:])

        x_temp = xcoords[intensity0 != 0]
        y_temp = ycoords[intensity0 != 0]
        
        med_long = np.median(x_temp)
        
        med_lat_tot = np.append(med_lat_tot, np.median(y_temp))
        slopes_tot = np.append(slopes_tot, fit_params[i+number,0])
        
        AL_bins_temp = [0 if x < AL_thresh else x for x in AL_bins[c]]
        AL_nonzero = np.array(np.nonzero(AL_bins_temp))
        
        """                
        if count_tot == 0:
            plt.figure(figsize=(9,12))
            ax1 = plt.gca()
            #plt.ylim(0,360)
            plt.xlim(0,360)  # for sideways
            plt.ylim(2700,0)  # for sideways
            plt.title('All Bands : %sern Hemisphere' % hemiF, fontsize=font_size)
            plt.hlines(fEnd[0],0,360, linestyle='dashed')
            plt.hlines(fEnd[1],0,360, linestyle='dashed')
            plt.hlines(fEnd[2],0,360, linestyle='dashed')
            plt.hlines(fEnd[3],0,360, linestyle='dashed')
            plt.xlabel('Longitude', fontsize=font_size)
            plt.ylabel('Frame', fontsize=font_size)
        #plt.scatter(frm_temp,x_temp)
        plt.scatter(x_temp, frm_temp)  # for sideways
        count_tot += 1
        """
        
        #"""
        for r in range(len(AL_nonzero[0])):
            if med_long >= AL_nonzero[0,r]*10 and med_long < (AL_nonzero[0,r]*10 + 10):
                
                AL_lat = np.append(AL_lat, np.median(y_temp))
                AL_slopes = np.append(AL_slopes, fit_params[i+number,0])
                
                """               
                if count_tot == 0:
                    plt.figure(figsize=(9,12))
                    ax1 = plt.gca()
                    #plt.ylim(0,360)
                    plt.xlim(0,360)  # for sideways
                    plt.ylim(2700,0)  # for sideways
                    plt.title('Bands Within AL Zone : %sern Hemisphere' % hemiF, fontsize=font_size)
                    plt.hlines(fEnd[0],0,360, linestyle='dashed')
                    plt.hlines(fEnd[1],0,360, linestyle='dashed')
                    plt.hlines(fEnd[2],0,360, linestyle='dashed')
                    plt.hlines(fEnd[3],0,360, linestyle='dashed')
                    plt.xlabel('Longitude', fontsize=font_size)
                    plt.ylabel('Frame', fontsize=font_size)
                #plt.scatter(frm_temp,x_temp)
                plt.scatter(x_temp, frm_temp)  # for sideways
                count_tot += 1

                if count_tot == 0:
                    plt.figure(figsize=(12,10))
                    plt.title('Slope vs Latitude: Bands in AL Zones [> 6/16]', fontsize=19)
                    plt.ylim(-1,1)
                    plt.xlim(-35,0)
                    plt.xlabel('Latitude', fontsize=19)
                    plt.ylabel('Slope [Per Frame]', fontsize=19)
                    #plt.xlim(0,360)  # for sideways
                    #plt.ylim(2900,0)  # for sideways
                plt.scatter(med_lat_tot, slopes_tot)
                #plt.scatter(x_temp, frm_temp)  # for sideways
                count_tot += 1
                """
                
        #"""
    number += int(num_bands[c])
#plt.savefig('C:/Users/Brendan/Desktop/Bands_Within_AL_Zone_%s.jpeg' % hemiF, bbox_inches='tight')    
#plt.savefig('C:/Users/Brendan/Desktop/All_Bands_%s.jpeg' % hemiF, bbox_inches='tight')    

"""   
#med_lat_sin2 = AL_lat  ## AL bands - strict latitude
med_lat_sin2 = np.sin(np.deg2rad(AL_lat))**2  ## AL bands
#med_lat_sin2 = np.sin(np.deg2rad(lat_tot))**2

slopes_days = AL_slopes*2
#slopes_days = slope_tot*2

m2, b2 = np.polyfit(med_lat_sin2, slopes_days, 1)

r_val = pearsonr(slopes_days, med_lat_sin2)[0]
print r_val

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
plt.figure(figsize=(15,10))
plt.title('Slope vs Latitude: %sern Hemisphere (AL Bands > %i/16)' % (hemiF, AL_thresh), y=1.01, fontsize=font_size)
#plt.scatter(med_lat,slopes)
plt.scatter(med_lat_sin2, slopes_days)
#plt.plot(med_lat, m*med_lat + b, 'r-')  
plt.plot(med_lat_sin2, m2*med_lat_sin2 + b2, 'r-')  
plt.text(0.2,1.,'y = %0.2fx + %0.2f' % (m2, b2), fontsize=font_size) 
plt.xlim(0,0.3)
plt.ylim(-2,2)
#plt.savefig('C:/Users/Brendan/Desktop/Slopes_AL_Bands_%iof16_thresh.jpeg' % AL_thresh, bbox_inches='tight')    
"""