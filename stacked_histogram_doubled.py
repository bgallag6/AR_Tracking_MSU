# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:41:11 2017

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
#deg0 = [30]
deg0 = [10,15,20,30]

hemi = 'N'
#hemi = 'S'
smooth_x = 5  #  5, 6, 8, 10
smooth_y = 2  #  2, 3, 4, 5

AL_thresh = 8


if hemi == 'N':
    hemiF = 'North'
elif hemi == 'S':
    hemiF = 'South'
  
num_bands5 =  np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 5, 2))
num_bands6 =  np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 6, 3))
num_bands8 =  np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 8, 4))
num_bands10 =  np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 10, 5))
    
ARs5 = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 5, 2))
ARs6 = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 6, 3))
ARs8 = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 8, 4))
ARs10 = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 10, 5))

ARs_arr_N = [ARs5, ARs6, ARs8, ARs10]
num_bands_arr_N = [num_bands5, num_bands6, num_bands8, num_bands10]

hemi = 'S'

num_bands5 =  np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 5, 2))
num_bands6 =  np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 6, 3))
num_bands8 =  np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 8, 4))
num_bands10 =  np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 10, 5))
    
ARs5 = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 5, 2))
ARs6 = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 6, 3))
ARs8 = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 8, 4))
ARs10 = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, 10, 5))

ARs_arr_S = [ARs5, ARs6, ARs8, ARs10]
num_bands_arr_S = [num_bands5, num_bands6, num_bands8, num_bands10]

#AL_total = np.zeros((4,18,37))
AL_combined = np.zeros((18,36))

y_lim = 5.
#y_lim = 16.
#y_lim = 64.
#y_lim = 32.
#y_lim = 128.

for n in range(4):
    colors = ['black','blue','red','green']
    ARs = ARs_arr_N[n]
    num_bands = num_bands_arr_N[n]

    for i in range(500):
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
                
            number += int(num_bands[c])  # could have better way -- summing all values below
            
            ### add / subtract 360 degrees if band was corrected to below 0 / above 360
            x_tot = np.array([x-360 if x > 360 else x for x in x_tot])
            x_tot = np.array([x+360 if x < 0 else x for x in x_tot])
            
              
            plt.figure()
            y1, x1, _ = plt.hist(x_tot, bins=x_bins)
            plt.close()
               
            #y3 = y1/(np.std(y1))
            y3 = y1
            
            nbins = deg/10
            for h3 in range(num_bins):
                for h4 in range(nbins):
                    #AL_combined[c,(h3*nbins)+h4] += y7[h3]
                    AL_combined[c,(h3*nbins)+h4] += y3[h3]
                    

for n in range(4):
    colors = ['black','blue','red','green']
    ARs = ARs_arr_S[n]
    num_bands = num_bands_arr_S[n]

    for i in range(500):
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
                
            number += int(num_bands[c])  # could have better way -- summing all values below
            
            ### add / subtract 360 degrees if band was corrected to below 0 / above 360
            x_tot = np.array([x-360 if x > 360 else x for x in x_tot])
            x_tot = np.array([x+360 if x < 0 else x for x in x_tot])
            
              
            plt.figure()
            y1, x1, _ = plt.hist(x_tot, bins=x_bins)
            plt.close()
               
            #y3 = y1/(np.std(y1))
            y3 = y1
            
            nbins = deg/10
            for h3 in range(num_bins):
                for h4 in range(nbins):
                    #AL_combined[c,(h3*nbins)+h4] += y7[h3]
                    AL_combined[c,(h3*nbins)+h4] += y3[h3]
                    
#np.save('C:/Users/Brendan/Desktop/3x_North_3sigma_combined.npy', AL_combined)
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
#"""
xbins36 = np.array([10*i for i in range(36)])
xbins72 = np.array([10*i for i in range(72)])
            
fig = plt.figure(figsize=(10,22))
#plt.suptitle(r'Southern Hemisphere ( > 3$\sigma$)' + '\n 3x Rotation Periods', y=0.97, fontweight='bold', fontsize=font_size)
plt.suptitle(r'Combined Hemispheres: 3x Rotation Periods' + r'\n Black = 2$\sigma$ | Red = 3$\sigma$', y=0.97, fontweight='bold', fontsize=font_size)           

for h1 in range(18):
    
    yrow = AL_combined[h1,:]/np.std(AL_combined[h1,:])
    y0 = [0 if x < 2. else x for x in yrow]
    y8 = [0 if x >= 3. else x for x in y0]
    y9 = [0 if x < 3. else x for x in y0]
    #y0 = [0 if x < 24. else x for x in yrow]   
    #y8 = [0 if x >= 36. else x for x in y0]
    #y9 = [0 if x < 36. else x for x in y0]
    
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
    ax1.set_ylim(0,y_lim)
    ax1.set_xlim(0,720)
    #ax1.bar(x_bins2, y6, width=deg/3, color=colors[h2], alpha=0.5)
    #ax1.bar(xbins36, AL_combined[h1,:], width=10, color='blue')
    #ax1.bar(xbins36, y8, width=10, color='black')
    #ax1.bar(xbins36, y9, width=10, color='red')
    #ax1.bar(xbins72, y00, width=10, color='blue')
    ax1.bar(xbins72, y88, width=10, color='black')
    ax1.bar(xbins72, y99, width=10, color='red')

#plt.savefig('C:/Users/Brendan/Desktop/3x_Car_Rot_Combined_30int_doubled.jpeg', bbox_inches = 'tight')
#"""