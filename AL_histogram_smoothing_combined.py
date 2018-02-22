# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:31:41 2017

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

#"""
plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
#deg = 10
deg0 = [10,20,30]
#deg0 = [20]
#deg0 = [10,15,20,30]

hemi = 'N'
#hemi = 'S'

AL_thresh = 8


if hemi == 'N':
    hemiF = 'North'
elif hemi == 'S':
    hemiF = 'South'
  
#num_bands5 =  np.load('C:/Users/Brendan/Desktop/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 5, 2))
#num_bands6 =  np.load('C:/Users/Brendan/Desktop/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 6, 3))
#num_bands8 =  np.load('C:/Users/Brendan/Desktop/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 8, 4))
#num_bands10 =  np.load('C:/Users/Brendan/Desktop/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 10, 5))
    
#ARs5 = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 5, 2))
#ARs6 = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 6, 3))
#ARs8 = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 8, 4))
#ARs10 = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 10, 5))

#"""
num_bands5 =  np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/%s/AR_Absolute_num_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, 5, 2))
num_bands6 =  np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/%s/AR_Absolute_num_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, 6, 3))
num_bands7 =  np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/%s/AR_Absolute_num_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, 8, 4))
num_bands8 =  np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/%s/AR_Absolute_num_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, 10, 5))
    
ARs5 = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/%s/AR_Absolute_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, 5, 2))
ARs6 = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/%s/AR_Absolute_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, 6, 3))
ARs7 = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/%s/AR_Absolute_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, 8, 4))
ARs8 = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/%s/AR_Absolute_bands_%s_3x_%sx%sysmooth.npy' % (hemiF, hemi, 10, 5))
#"""

"""
num_bands5 =  np.load('C:/Users/Brendan/Desktop/Inbox/AL2/AR_Absolute_num_bands_%s_3x_%sx%sysmooth.npy' % (hemi, 5, 3))
num_bands6 =  np.load('C:/Users/Brendan/Desktop/Inbox/AL2/AR_Absolute_num_bands_%s_3x_%sx%sysmooth.npy' % (hemi, 8, 3))
num_bands7 =  np.load('C:/Users/Brendan/Desktop/Inbox/AL2/AR_Absolute_num_bands_%s_3x_%sx%sysmooth.npy' % (hemi, 5, 6))
num_bands8 =  np.load('C:/Users/Brendan/Desktop/Inbox/AL2/AR_Absolute_num_bands_%s_3x_%sx%sysmooth.npy' % (hemi, 8, 6))
    
ARs5 = np.load('C:/Users/Brendan/Desktop/Inbox/AL2/AR_Absolute_bands_%s_3x_%sx%sysmooth.npy' % (hemi, 5, 3))
ARs6 = np.load('C:/Users/Brendan/Desktop/Inbox/AL2/AR_Absolute_bands_%s_3x_%sx%sysmooth.npy' % (hemi, 8, 3))
ARs7 = np.load('C:/Users/Brendan/Desktop/Inbox/AL2/AR_Absolute_bands_%s_3x_%sx%sysmooth.npy' % (hemi, 5, 6))
ARs8 = np.load('C:/Users/Brendan/Desktop/Inbox/AL2/AR_Absolute_bands_%s_3x_%sx%sysmooth.npy' % (hemi, 8, 6))
"""

ARs_arr = [ARs6, ARs7, ARs8]
num_bands_arr = [num_bands6, num_bands7, num_bands8]

#ARs_arr = [ARs7]
#num_bands_arr = [num_bands7]

#AL_total = np.zeros((4,18,37))
AL_combined = np.zeros((18,36))

#y_lim = 4.
#y_lim = 16.
#y_lim = 32.
#y_lim = 64.

#for n in range(1):
for n in range(len(ARs_arr)):
    colors = ['black','blue','red','green']
    ARs = ARs_arr[n]
    num_bands = num_bands_arr[n]

    for i in range(1000):
        if ARs[i,0,0] == 0:
            count = i
            break
    
    rot_start = 0
    rot_end = 18
    
    for k in range(len(deg0)):
        number = 0
        deg = deg0[k]
        num_bins = 360/deg
        
        x_bins = [deg*l for l in range(num_bins+1)]
        x_bins2 = [deg*l for l in range(num_bins)]
        
        x_ticks = np.array(x_bins) + (deg/2)
        
        #AL_array = np.zeros((rot_end-rot_start,num_bins))
            
        for c in range(rot_start,rot_end):
        #for c in range(8):    
        
            nbins = deg/10
               
            int_tot = []
            x_tot = []
            frm_tot = []
            first_tot = []
            
            x_scaled = np.zeros((36))
            
            for i in range(int(num_bands[c])):
            
                intensities = np.array(ARs[i+number,2,:])
                int_temp = intensities[intensities != 0]
                int_avg = np.average(int_temp)
                
                xcoords = np.array(ARs[i+number,1,:])
                x_temp = xcoords[xcoords != 0]
                x_count = len(x_temp)
                x_avg = np.average(x_temp)  # for each band counts as one
                
                frm = np.array(ARs[i+number,0,:]) 
                frm_temp = frm[frm != 0]
                
                int_tot = np.append(int_tot, int_temp)
                
                x_tot = np.append(x_tot, x_temp)
                
                frm_tot = np.append(frm_tot, frm_temp) 
                
                first_tot = np.append(first_tot, x_temp[0])
                
                #"""
                for l1 in range(num_bins):
                    for l2 in range(nbins):
                        if x_avg > l1*deg and x_avg < (l1+1)*deg:
                            #x_scaled[(l1*nbins)+l2] += np.sum(int_tot)
                            x_scaled[(l1*nbins)+l2] += (x_count / int_avg)
                            #x_scaled[(l1*nbins)+l2] += (int_avg / x_count)
                            #x_scaled[(l1*nbins)+l2] += 1
                #"""
                
            number += int(num_bands[c])  # could have better way -- summing all values below
            
            ### add / subtract 360 degrees if band was corrected to below 0 / above 360
            x_tot = np.array([x-360 if x > 360 else x for x in x_tot])
            x_tot = np.array([x+360 if x < 0 else x for x in x_tot])
            
            
            y1, x1 = np.histogram(x_tot, bins=x_bins)
            
            """
            ### If using the top 1 or 2 bins only
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
            """
             
            #"""
            #y3 = y1/np.mean(y1)
            #y3 = y1/(np.std(y1))
            y3 = (y1-np.mean(y1))/(np.std(y1))
            #y4 = [0 if x < 2. else x for x in y3]
            #y5 = [0 if x >= 3. else x for x in y4]
            #y6 = [0 if x < 3. else x for x in y4]
            y6 = [0 if x < 1. else x for x in y3]
            #y5 = [0 if x >= 2. else x for x in y4]
            #y6 = [0 if x < 2. else x for x in y4]
            
            #AL_total[n,c,0:len(y6)] = y6
            #AL_array[c] = y6
            
            
            y7 = [1 if x != 0 else x for x in y6]
            #y7 = [1 if x != 0 else x for x in y3]
            for h3 in range(num_bins):
                for h4 in range(nbins):
                    #AL_combined[c,(h3*nbins)+h4] += y7[h3]
                    #AL_combined[c,(h3*nbins)+h4] += y6[h3]
                    AL_combined[c,(h3*nbins)+h4] += y3[h3]
            #"""    
            
            #x_mean = np.mean(x_scaled)
            #x_std = np.std(x_scaled)
            #x_scaled = [0 if ((x-x_mean)/x_std) < 0.5 else x for x in x_scaled]
            #x_scaled = [0 if x < x_mean else x for x in x_scaled]
            #x_scaled /= np.mean(x_scaled)
            
            #AL_combined[c] += x_scaled  # added 8_25
                    
#np.save('C:/Users/Brendan/Desktop/3x_%s_3sigma_combined_absolute.npy' % hemiF, AL_combined)
#"""

"""
nbin = [11]

y_lim = 4.

fig = plt.figure(figsize=(10,22))
#plt.suptitle(r'Southern Hemisphere ( > 3$\sigma$)' + '\n 3x Rotation Periods', y=0.97, fontweight='bold', fontsize=font_size)
plt.suptitle(r'Southern Hemisphere' + '\n 3x Rotation Periods', y=0.97, fontweight='bold', fontsize=font_size)           

for h1 in range(18):
    for h2 in range(4):
           
        num_bins = 12
    
        y6 = np.array(AL_total[h2,h1,0:num_bins])        
        
        if h2 == 0:
            if h1 == 0:
                ax1 = plt.subplot2grid((18,1),(h1, 0), colspan=1, rowspan=1)
            elif h1 == 17:
                ax1 = plt.subplot2grid((18,1),(h1, 0), colspan=1, rowspan=1, sharey=ax1)
                ax1.set_xlabel('Longitude', fontsize=font_size)
                plt.xticks(x_bins)
            else:
                ax1 = plt.subplot2grid((18,1),(h1, 0), colspan=1, rowspan=1, sharex=ax1, sharey=ax1)
                ax1.set_xticklabels([])
            
        ax1 = plt.gca()         
        ax1.set_ylabel('%i' % (h1+1), fontsize=font_size)   
        ax1.set_ylim(0,y_lim)
        ax1.set_xlim(0,360)
        #ax1.bar(x_bins2, y6, width=deg/3, color=colors[h2], alpha=0.5)
        ax1.bar(x_bins2, y6, width=4+(3*(h2+1)), color=colors[h2], alpha=0.5)
                 
#plt.savefig('C:/Users/Brendan/Desktop/3x_Car_Rot_North_3plus_30deg_30int_combined_colors.jpeg', bbox_inches = 'tight')
#plt.savefig('C:/Users/Brendan/Desktop/3x_Car_Rot_North_full_%ideg_30int_%ix%iy.jpeg' % (deg,smooth_x,smooth_y), bbox_inches = 'tight')
"""

"""   
AL_combined = np.zeros((18,37))
for h1 in range(18):
    for h2 in range(4):
        num_bins = 12
        deg = 360/num_bins
        bins = deg/10
        y6 = np.array(AL_total[h2,h1,0:num_bins])
        y7 = [1 if x != 0 else x for x in y6]
        for h3 in range(num_bins):
            for h4 in range(bins):
                AL_combined[h1,(h3*bins)+h4] += y7[h3]
    
"""
xbins36 = np.array([10*i for i in range(36)])
xbins72 = np.array([5*i for i in range(36)])
            
fig = plt.figure(figsize=(10,22))
#plt.suptitle(r'Southern Hemisphere ( > 3$\sigma$)' + '\n 3x Rotation Periods', y=0.97, fontweight='bold', fontsize=font_size)
#plt.suptitle(r'%sern Hemisphere: 3x Rotation Periods'  % hemiF + '\n Black = 6/16+ | Red = 9/16+', y=0.97, fontweight='bold', fontsize=font_size)   
plt.suptitle(r'%sern Hemisphere: 3x Rotation Periods'  % hemiF + '\n Points Count Each Point', y=0.97, fontweight='bold', fontsize=font_size)           

for h1 in range(18):
    
    yrow = AL_combined[h1,:]
    #y0 = [0 if x < 6. else x for x in yrow]
    #y8 = [0 if x >= 9. else x for x in y0]
    #y9 = [0 if x < 9. else x for x in y0]
    #y0 = [0 if x < 4. else x for x in yrow]   
    #y8 = [0 if x >= 8. else x for x in y0]
    #y9 = [0 if x < 8. else x for x in y0]
    y0 = [0 if x < 10. else x for x in yrow]   
    y8 = [0 if x >= 20. else x for x in y0]
    y9 = [0 if x < 20. else x for x in y0]
    
    y00 = np.append(AL_combined[h1,:], AL_combined[h1,:])
    y88 = np.append(y8,y8)
    y99 = np.append(y9,y9)
    
    if h1 == 0:
        ax1 = plt.subplot2grid((18,1),(h1, 0), colspan=1, rowspan=1)
    elif h1 == 17:
        ax1 = plt.subplot2grid((18,1),(h1, 0), colspan=1, rowspan=1, sharey=ax1)
        ax1.set_xlabel('Longitude', fontsize=font_size)
        #plt.xticks(x_bins)
        #plt.xticks(xbins36)
    else:
        ax1 = plt.subplot2grid((18,1),(h1, 0), colspan=1, rowspan=1, sharex=ax1, sharey=ax1)
        ax1.set_xticklabels([])
    
    ax1 = plt.gca()         
    ax1.set_ylabel('%i' % (h1+1), fontsize=font_size)   
    #ax1.set_ylim(0,y_lim)
    ax1.set_xlim(0,360)
    #ax1.bar(x_bins2, y6, width=deg/3, color=colors[h2], alpha=0.5)
    #ax1.bar(xbins36, AL_combined[h1,:], width=10, color='blue')
    #ax1.bar(xbins36, yrow, width=10, color='black')
    #ax1.bar(xbins36, yrow, width=10, color='black')
    
    ax1.bar(xbins36, y0, width=10, color='blue')
    ax1.bar(xbins36, y8, width=10, color='black')
    ax1.bar(xbins36, y9, width=10, color='red')
    
    #ax1.bar(xbins72, y00, width=10, color='blue')
    #ax1.bar(xbins72, y88, width=10, color='black')
    #ax1.bar(xbins72, y99, width=10, color='red')

plt.savefig('C:/Users/Brendan/Desktop/North_Combined_Hist_count_each_point_9combin_raw_std.pdf', bbox_inches = 'tight')