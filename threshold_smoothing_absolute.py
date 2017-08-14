# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 11:42:51 2017

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

if hemi == 'N':
    hemiF = 'North'
elif hemi == 'S':
    hemiF = 'South'

smoothXY = [[5,2], [6,3], [7,4], [8,5]]
#threshAL = [5,8,10,12]
#threshAL = [5,7,9,11]
threshAL = [15,20,25,30]

count1 = 0
count2 = 0

color = ['black', 'red', 'blue', 'green']
        
for w1 in range(4):
#for w1 in range(1):
    int_array = np.zeros((4))
    int_median = np.zeros((4))
    for w2 in range(4):
        smooth_x = smoothXY[w1][0]
        smooth_y = smoothXY[w1][1]
        
        AL_thresh = threshAL[w2]
        
        #print smooth_x, smooth_y, AL_thresh     
           
        #num_bands = np.load('C:/Users/Brendan/Desktop/MSU_Project/num_bands_S.npy')
        num_bands = np.load('C:/Users/Brendan/Desktop/MSU_Project/8_3/absolute/%s/AR_Absolute_num_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, smooth_x, smooth_y))
        #num_bands = num_bands
            
        #ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S.npy')
        ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/8_3/absolute/%s/AR_Absolute_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, smooth_x, smooth_y))
        #ARs = AR_total
        
        #fit_params = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_slopes.npy')
        fit_params = np.load('C:/Users/Brendan/Desktop/MSU_Project/8_3/absolute/%s/AR_Absolute_slopes_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, smooth_x, smooth_y))
        
        AL_bins = np.load('C:/Users/Brendan/Desktop/MSU_Project/8_3/absolute/%s/3x_%s_3sigma_combined_absolute.npy' % (hemiF,hemiF))
        
        for i in range(1000):
            if ARs[i,0,0] == 0:
                count = i
                break
            
        number = 0
        
        rot_start = 0
        rot_end = 18
        
        AL_lat = []
        AL_slopes = []
        AL_int = []
        
        int_tot = []
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
                
                int_temp = intensity0[intensity0 != 0]
                
                med_long = np.median(x_temp)
                
                med_lat_tot = np.append(med_lat_tot, np.median(y_temp))
                slopes_tot = np.append(slopes_tot, fit_params[i+number,0])
                
                AL_bins_temp = [0 if x < AL_thresh else x for x in AL_bins[c]]
                AL_nonzero = np.array(np.nonzero(AL_bins_temp))
                
                int_tot = np.append(int_tot, int_temp)
                
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
                        AL_int = np.append(AL_int, int_temp)
                        
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
        
        #"""
        int_bin_size = 10
        num_int_bins = 500/int_bin_size
        
        int_bins = np.array([int_bin_size*k for k in range(3,num_int_bins+1)])
        int_bins2 = np.array([int_bin_size*k for k in range(3,num_int_bins)])
        
        font_size = 21
        
        plt.figure()
        y1, x1, _ = plt.hist(int_tot, bins = int_bins)
        y2, x2, _ = plt.hist(AL_int, bins = int_bins)
        plt.close()
        int_tot_avg = np.average(int_tot)
        AL_int_avg = np.average(AL_int)
        #print int_tot_avg, AL_int_avg
        AL_int_med = np.median(AL_int)
        int_tot_med = np.median(int_tot)
        
        elem1 = np.argmax(y1)
        elem2 = np.argmax(y2)
        bin_max1 = y1[elem1]
        bin_max2 = y2[elem2]
        bin_sum1 = np.sum(y1)
        bin_sum2 = np.sum(y2)
        
        #int_tot_norm = y1 / bin_max1
        #AL_int_norm = y2 / bin_max2
        int_tot_norm = y1 / bin_sum1
        AL_int_norm = y2 / bin_sum2
        
        #print AL_thresh, AL_int_avg
        
        int_array[w2] = AL_int_avg
        int_median[w2] = AL_int_med
        
        
        """
        plt.figure(figsize=(15,12))
        plt.title('Active Longitude Average EUV Integrated Intensity: Thresholds', fontsize=font_size+3, y=1.01)
        plt.plot(int_bins2, int_tot_norm, color='black', linewidth=2, label='All Bands: Avg. = %i | Med. = %i' % (int_tot_avg,int_tot_med))
        plt.plot(int_bins2, AL_int_norm, color='blue', linewidth=2, label='AL Bands: Avg. = %i | Med. = %i' % (AL_int_avg,AL_int_med))
        #plt.xlim(4,13)
        plt.ylim(0,0.2)
        #plt.vlines(int_tot_avg,0,1.1,color='green',label='All Bands: Average = %i' % int_tot_avg, linestyle='solid', linewidth=2)
        #plt.vlines(int_tot_med,0,1.1,color='green',label='All Bands: Median = %i' % int_tot_med, linestyle='dashed', linewidth=2)
        #plt.vlines(AL_int_avg,0,1.1,color='red',label='AL Bands: Average = %i' % AL_int_avg, linestyle='solid', linewidth=2)
        #plt.vlines(AL_int_med,0,1.1,color='red',label='AL Bands: Median = %i' % AL_int_med, linestyle='dashed', linewidth=2)
        plt.vlines(30,0,1.1,label='Intensity Threshold > 30', linestyle='dashed', linewidth=2)
        plt.ylabel('EUV Integrated Intensity', fontsize=font_size)
        plt.xlabel('Active Longitude Threshold', fontsize=font_size)
        plt.legend(fontsize=font_size, loc='upper right')
        #plt.savefig('C:/Users/Brendan/Desktop/AL_Intensity_Vs_Threshold_Smoothing_%s_%iC.jpeg' % (hemi,w2), bbox_inches='tight')    
        """
        
        #"""
        if w1 == 0 and w2 == 0:
            plt.figure(figsize=(15,12))
            plt.title('Active Region Integrated Intensity in Active Longitude Zones: %sern Hemisphere' % hemiF, fontsize=font_size+3, y=1.01)
            #plt.bar(int_bins2, int_tot_norm, width=10, color='black', alpha=0.5)
            #plt.bar(int_bins2, AL_int_norm, width=10, color='blue', alpha=0.5)
            #plt.xlim(4,12)
            plt.xlim(10,35)
            #plt.ylim(0,1.1)
            #plt.ylim(80,180)
            #plt.ylim(60,130)
            plt.ylim(60,125)
            #plt.vlines(30,0,1.1,label='Intensity Threshold > 30', linestyle='dashed', linewidth=2)
            plt.ylabel('EUV Integrated Intensity', fontsize=font_size)
            plt.xlabel('Active Longitude Threshold', fontsize=font_size)
        if count1 == count2:
            plt.hlines(int_tot_avg, 0, 100, color='%s' % color[w1], linewidth=2, linestyle='dashed')   
            plt.hlines(int_tot_med, 0, 100, color='%s' % color[w1], linewidth=2, linestyle='solid')
            count1 += 5
        count2 += 1
    plt.plot(0,0, color='%s' % color[w1], linestyle='solid', linewidth=2, label='Smoothing: %ix,%iy' % (smooth_x,smooth_y))
    plt.plot(threshAL, int_array, color='%s' % color[w1], linestyle='dashed', linewidth=2)
    plt.plot(threshAL, int_median, color='%s' % color[w1], linestyle='solid', linewidth=2)
    plt.scatter(threshAL, int_array, 40, color='%s' % color[w1])
    plt.scatter(threshAL, int_median, 40, color='%s' % color[w1])
    
    print smooth_x, smooth_y
    print int_array, int_median
    print int_tot_avg, int_tot_med

plt.plot(0,0, color='%s' % color[w1], linestyle='dashed', linewidth=2, label='Average')
plt.plot(0,0, color='%s' % color[w1], linestyle='solid', linewidth=2, label='Median')
plt.legend(fontsize=font_size, loc='upper left')
#plt.text(9.4,92,'All Bands -- Average Intensity', fontsize=font_size)
#plt.text(9.4,72.5,'All Bands -- Median Intensity', fontsize=font_size)
plt.text(27,92,'All Bands -- Average Intensity', fontsize=font_size)
plt.text(27,71,'All Bands -- Median Intensity', fontsize=font_size)
plt.xticks(fontsize=font_size-3)
plt.yticks(fontsize=font_size-3)
#plt.text(9.8,71,'All Bands -- Median Intensity', fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/AL_Intensity_Vs_Threshold_Smoothing_%s_absolute.jpeg' % hemi, bbox_inches='tight')    
#"""