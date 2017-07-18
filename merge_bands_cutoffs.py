# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:32:19 2017

@author: Brendan
"""

"""
#################################################
### interactive tool so you can click ###########
### on a scatter point of duration/intensity  ###
### and displays AR life statistics #############
#################################################
* build in latitude for plot verification *
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.idl import readsav
import jdcal

#ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_lat.npy')
#ARs = np.load('C:/Users/Brendan/Desktop/AR_bands_N_3x_15int_3smooth.npy')
ARs = AR_total
#fit_paramsS = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_slopes.npy')
fit_params = fit_params

count = 0
    

#for i in range(500):
for i in range(1000):
    if ARs[i,0,0] == 0:
        count = i
        break

slopes = fit_params[:,0]
slopes = slopes[slopes != 0]

"""               
global frames, avg_int, area
frames = np.zeros((count))
first = np.zeros((count))
x_form = np.zeros((count))
y_form = np.zeros((count))
x_end = np.zeros((count))
y_end = np.zeros((count))
x_avg = np.zeros((count))
y_avg = np.zeros((count))
avg_int = np.zeros((count))
med_int = np.zeros((count))
max_int = np.zeros((count))
distance = np.zeros((count))
area = np.zeros((count))

for c in range(count): 

    ar0 = ARs[c,:,:]
    
    frms = ar0[0][ar0[0] != 0]
    frames[c] = (np.max(frms) - np.min(frms)) # - how many frames AR lasts
    
    first[c] = np.min(frms)
    
    #for d in range(count):
    #    if ar0[0,d] != 0:
    #        first[c] = d
    #        break
    
    ### could use a sort for these next few
    #x_form[c] = ar0[1][ar0[1] != 0][0]
    
    #y_form[c] = ar0[2][ar0[2] != 0][0]
    #x_end[c] = ar0[1][ar0[1] != 0][-1]
    
    #y_end[c] = ar0[2][ar0[2] != 0][-1]
    
    x_coords = ar0[1][ar0[1] != 0]
    x_form[c] = np.average(ar0[1][np.where(ar0[0,:] == np.min(frms))])
    x_end[c] = np.average(ar0[1][np.where(ar0[0,:] == np.max(frms))])
    #x_avg[c] = np.average(x_coords)
    
    y_coords = ar0[3][ar0[3] != 0]
    #y_avg[c] = np.average(y_coords)
    
    avg_int[c] = np.average(ar0[2][ar0[2] != 0])  
    
    #area = ar0[3][ar0[3] != 0]
    
    #dist_steps = np.zeros((int(frames[c]-1)))    
    
    #for r in range(int(frames[c])-1):
    #    dist_steps[r] = np.sqrt((x_coords[r+1]-x_coords[r])**2 + (y_coords[r+1]-y_coords[r])**2)
    #distance[c] = np.sum(dist_steps)
"""

#num_bands = np.load('C:/Users/Brendan/Desktop/MSU_Project/num_bands_S.npy')
num_bands = num_bands
    
number = 0

rot_start = 0
rot_end = 18
    
#for c in range(rot_start,rot_end):
for c in range(3):    
       
    int_tot = []
    x_tot = []
    frm_tot = []
    first_tot = []
    AR_prev = np.zeros((50,5,250))
    
    for i in range(int(num_bands[c])):
    
        intensities = np.array(ARs[i+number,2,:])
        int_temp = intensities[intensities != 0]
        
        xcoords = np.array(ARs[i+number,1,:])
        x_temp = xcoords[xcoords != 0]       
        
        ycoords = np.array(ARs[i+number,3,:])
        y_temp = ycoords[ycoords != 0]
        
        frm = np.array(ARs[i+number,0,:]) 
        frm_temp = frm[frm != 0]
        #print np.min(frm_temp), np.max(frm_temp), y_temp[np.where(frm_temp == np.min(frm_temp))], y_temp[np.where(frm_temp == np.max(frm_temp))] 
        
        int_tot = np.append(int_tot, int_temp)
        
        x_tot = np.append(x_tot, x_temp)
        
        frm_tot = np.append(frm_tot, frm_temp) 
        
        first_tot = np.append(first_tot, x_temp[0])
        
        AR_prev
        
    number += int(num_bands[c])
    
    if c > 0:
        prev_end = int(number - num_bands[c])
        prev_start = int(prev_end - num_bands[c-1])
        
        #print prev_start, prev_end
        for k in range(prev_end, number):
            ARs_int_temp0 = ARs[k,2,:]
            ARs_frm_temp0 = ARs[k,0,:]
            ARs_long_temp0 = ARs[k,1,:]
            ARs_int_tempA = ARs_int_temp0[ARs_int_temp0 != 0]
            ARs_frm_tempA = ARs_frm_temp0[ARs_int_temp0 != 0]
            ARs_long_tempA = ARs_long_temp0[ARs_int_temp0 != 0]
            
            #print np.min(ARs_frm_tempA)
            #print ARs_long_tempA[np.where(ARs_frm_tempA == np.min(ARs_frm_tempA))]
        
            for j in range(prev_start, prev_end):
                
                ARs_int_temp0 = ARs[j,2,:]
                ARs_frm_temp0 = ARs[j,0,:]
                ARs_long_temp0 = ARs[j,1,:]
                ARs_int_tempB = ARs_int_temp0[ARs_int_temp0 != 0]
                ARs_frm_tempB = ARs_frm_temp0[ARs_int_temp0 != 0]
                ARs_long_tempB = ARs_long_temp0[ARs_int_temp0 != 0]
                ARs_slopeB = slopes[j] 
                ARs_long_tempB_corrected = ARs_long_tempB[np.where(ARs_frm_tempB == np.max(ARs_frm_tempB))] + np.max(ARs_frm_tempB)*ARs_slopeB
                
                print np.min(ARs_frm_tempA), np.max(ARs_frm_tempB)
                #print np.max(ARs_long_tempB)
                print ARs_long_tempA[np.where(ARs_frm_tempA == np.min(ARs_frm_tempA))], ARs_long_tempB_corrected
                

                if ((np.min(ARs_frm_tempA) - np.max(ARs_frm_tempB)) < 5) and (np.abs((ARs_long_tempA[np.where(ARs_frm_tempA == np.min(ARs_frm_tempA))] - ARs_long_tempB_corrected)) < 10):
                    print np.min(ARs_frm_tempA), np.max(ARs_frm_tempB)
                    print ARs_long_tempA[np.where(ARs_frm_tempA == np.min(ARs_frm_tempA))], ARs_long_tempB_corrected
                    print k, j

                    plt.figure()
                    plt.scatter(ARs_frm_tempB,ARs_long_tempB)
                    plt.scatter(ARs_frm_tempA,ARs_long_tempA)
                    plt.xlim(0,360)
                    plt.ylim(0,360)
    
