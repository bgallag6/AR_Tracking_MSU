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
#deg0 = [30]
deg0 = [10,15,20,30]
  
num_bands5 =  np.load('C:/Users/Brendan/Desktop/AL_smoothing/num_bands_N_3x_30int_5x2ysmooth.npy')
num_bands6 =  np.load('C:/Users/Brendan/Desktop/AL_smoothing/num_bands_N_3x_30int_6x3ysmooth.npy')
num_bands8 =  np.load('C:/Users/Brendan/Desktop/AL_smoothing/num_bands_N_3x_30int_8x4ysmooth.npy')
num_bands10 =  np.load('C:/Users/Brendan/Desktop/AL_smoothing/num_bands_N_3x_30int_10x5ysmooth.npy')
    
ARs5 = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_bands_N_3x_30int_5x2ysmooth.npy')
ARs6 = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_bands_N_3x_30int_6x3ysmooth.npy')
ARs8 = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_bands_N_3x_30int_8x4ysmooth.npy')
ARs10 = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AR_bands_N_3x_30int_10x5ysmooth.npy')

ARs_arr = [ARs5, ARs6, ARs8, ARs10]
num_bands_arr = [num_bands5, num_bands6, num_bands8, num_bands10]

#AL_total = np.zeros((4,18,37))
AL_combined = np.zeros((18,36))

#y_lim = 16.
y_lim = 64.

for n in range(4):
    colors = ['black','blue','red','green']
    ARs = ARs_arr[n]
    num_bands = num_bands_arr[n]

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
                
            ### If using max 1/2 bins ###
            #norm1 = y1/np.sum(y1)
            #norm2 = np.copy(norm1)
            #y2 = np.sort(y1)
            #max1 = y1.tolist().index(y2[-1])
            #max2 = y1.tolist().index(y2[-2])
            
            #for u in range(len(norm1)):
            #    if u != max1:
            #        norm1[u] = 0
            #        
            #    if u != max2:
            #        norm2[u] = 0
               
            y3 = y1/(np.std(y1))
            y4 = [0 if x < 2. else x for x in y3]
            y5 = [0 if x >= 3. else x for x in y4]
            y6 = [0 if x < 3. else x for x in y4]
            
            #AL_total[n,c,0:len(y6)] = y6
            #AL_array[c] = y6
            
            nbins = deg/10
            y7 = [1 if x != 0 else x for x in y6]
            #y7 = [1 if x != 0 else x for x in y3]
            for h3 in range(num_bins):
                for h4 in range(nbins):
                    AL_combined[c,(h3*nbins)+h4] += y7[h3]
                    #AL_combined[c,(h3*nbins)+h4] += y3[h3]
                    
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
xbins36 = np.array([10*i for i in range(36)])
            
fig = plt.figure(figsize=(10,22))
#plt.suptitle(r'Southern Hemisphere ( > 3$\sigma$)' + '\n 3x Rotation Periods', y=0.97, fontweight='bold', fontsize=font_size)
plt.suptitle(r'Northern Hemisphere: 3x Rotation Periods' + '\n Black = 6/16+ | Red = 9/16+', y=0.97, fontweight='bold', fontsize=font_size)           

for h1 in range(18):
    
    yrow = AL_combined[h1,:]
    #y0 = [0 if x < 6. else x for x in yrow]
    #y8 = [0 if x >= 9. else x for x in y0]
    #y9 = [0 if x < 9. else x for x in y0]
    y0 = [0 if x < 24. else x for x in yrow]
    y8 = [0 if x >= 36. else x for x in y0]
    y9 = [0 if x < 36. else x for x in y0]
    
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
    ax1.set_xlim(0,360)
    #ax1.bar(x_bins2, y6, width=deg/3, color=colors[h2], alpha=0.5)
    ax1.bar(xbins36, AL_combined[h1,:], width=10, color='blue')
    ax1.bar(xbins36, y8, width=10, color='black')
    ax1.bar(xbins36, y9, width=10, color='red')

#plt.savefig('C:/Users/Brendan/Desktop/3x_Car_Rot_North_3plus_30int_combined_smooth_bins_points_full.jpeg', bbox_inches = 'tight')