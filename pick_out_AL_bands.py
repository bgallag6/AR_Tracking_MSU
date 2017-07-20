# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:29:20 2017

@author: Brendan
"""

"""
#########################################
#########################################
"""

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
deg = 15
num_bins = 360/deg

x_bins = [deg*l for l in range(num_bins+1)]
x_bins2 = [deg*l for l in range(num_bins)]

x_ticks = np.array(x_bins) + (deg/2)

### Southern Hemisphere ###
   
num_bands = np.load('C:/Users/Brendan/Desktop/MSU_Project/num_bands_S.npy')
#num_bands = num_bands
    
ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S.npy')
#ARs = AR_total

fit_params = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_slopes.npy')
#ARs = AR_total

for i in range(500):
    if ARs[i,0,0] == 0:
        count = i
        break
    
number = 0

rot_start = 0
rot_end = 18

AL_array = np.zeros((rot_end-rot_start,num_bins))

x_AL_tot = []
frm_AL_tot = []
int_AL_tot = []
duration_tot = []
slopes_tot = []
    
for c in range(rot_start,rot_end):
#for c in range(3):    
       
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
        
        duration = np.max(frm_temp) - np.min(frm_temp)
        
        int_tot = np.append(int_tot, int_temp)
        
        x_tot = np.append(x_tot, x_temp)
        
        frm_tot = np.append(frm_tot, frm_temp) 
        
        first_tot = np.append(first_tot, x_temp[0])
        
        duration_tot = np.append(duration_tot, duration)
        
        #slopes = 
        
    number += int(num_bands[c])
    
    x_tot = np.array([x-360 if x > 360 else x for x in x_tot])  # add this to main program
    x_tot = np.array([x+360 if x < 0 else x for x in x_tot])
    
    
    plt.figure()
    y1, x1, _ = plt.hist(x_tot, bins=x_bins)
    plt.close()
    
    #elem1 = np.argmax(y1)
    #bin_max1 = y1[elem1]
        
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
    
    if np.count_nonzero(y6) != 0:
        xbin_AL = np.array(x_bins2)[np.array(y6) != 0]
        for s in range(xbin_AL.size):
            x_AL0 = x_tot[x_tot > xbin_AL[s]]
            #x_AL = x_tot[x_tot < xbin_AL[0] + deg]
            x_AL = x_AL0[x_AL0 < xbin_AL[s] + deg]
            frm_AL = frm_tot[x_tot > xbin_AL[s]]
            #frm_AL = frm_tot[x_tot < xbin_AL[0] + deg]
            frm_AL = frm_AL[x_AL0 < xbin_AL[s] + deg]
            int_AL = int_tot[x_tot > xbin_AL[s]]
            #int_AL = int_tot[x_tot < xbin_AL[0] + deg]
            int_AL = int_AL[x_AL0 < xbin_AL[s] + deg]
            
            x_AL_tot = np.append(x_AL_tot, x_AL)
            frm_AL_tot = np.append(frm_AL_tot, frm_AL)
            int_AL_tot = np.append(int_AL_tot, int_AL)
    
    #plt.figure()
    #plt.scatter(frm_AL,x_AL)    
    
    #AL_array[c] = y6
plt.figure()
plt.scatter(frm_AL_tot, x_AL_tot)
"""    
    y_lim = 4.
    
    if c == rot_start:
        fig = plt.figure(figsize=(10,22))
        plt.suptitle(r'Southern Hemisphere ( > 3$\sigma$)' + '\n 3x Rotation Periods', y=0.97, fontweight='bold', fontsize=font_size) 
        ax1 = plt.subplot2grid((18,1),(c-rot_start, 0), colspan=1, rowspan=1)
        ax1 = plt.gca()         
        ax1.set_ylabel('%i' % (c+1), fontsize=font_size)
        #ax1.set_xlim(0,360)   
        ax1.set_ylim(0,y_lim)
        #ax1.bar(x_bins2, y5, width=deg/3)
        ax1.bar(x_bins2, y6, width=deg/3, color='black')
    elif c == rot_end-1: 
        ax1 = plt.subplot2grid((18,1),(c-rot_start, 0), colspan=1, rowspan=1, sharey=ax1)
        ax1 = plt.gca()      
        #ax1.set_xlim(0,360) 
        ax1.set_ylim(0,y_lim)
        ax1.set_ylabel('%i' % (c+1), fontsize=font_size)
        ax1.set_xlabel('Longitude', fontsize=font_size)
        #ax1.set_ylabel('Carrington Rotations', fontsize=font_size)
        plt.xticks(x_bins)
        #ax1.bar(x_bins2, y5, width=deg/3)
        ax1.bar(x_bins2, y6, width=deg/3, color='black')
    else:
        ax1 = plt.subplot2grid((18,1),(c-rot_start, 0), colspan=1, rowspan=1, sharex=ax1, sharey=ax1)
        ax1 = plt.gca()    
        #ax1.set_xlim(0,360)  
        ax1.set_ylim(0,y_lim)
        ax1.set_ylabel('%i' % (c+1), fontsize=font_size)
        ax1.set_xticklabels([])
        #ax1.bar(x_bins2, y5, width=deg/3)
        ax1.bar(x_bins2, y6, width=deg/3, color='black')
"""
    
"""
AL_array[AL_array > 0] = 1.
AL_array = np.transpose(AL_array)

phase_ticks = [(num_bins/5.)*i for i in range(6)]
car_ticks = [(18./17.)*i for i in range(rot_end-rot_start)]

phase_ind = [0.2*i for i in range(6)]
car_ind = [1+(3*i) for i in range(rot_end-rot_start)]

aspect = np.float(num_bins)/np.float((rot_end-rot_start))

aspect_shift = (2./3.)/aspect

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
plt.title('Southern Hemisphere: Active Longitude', y=1.01, fontsize=font_size+3)
plt.xlabel('Carrington Rotation [2096+]', fontsize=font_size)
plt.ylabel('Carrington Phase', fontsize=font_size)
ax.imshow(AL_array, cmap='Greys', interpolation='none')
plt.yticks(phase_ticks, phase_ind, fontsize=font_size-3)
plt.xticks(car_ticks, car_ind, fontsize=font_size-3)
ax.set_aspect(aspect_shift)

plt.figure()
plt.scatter(frm_AL_tot, x_AL_tot)
"""

#plt.savefig('C:/Users/Brendan/Desktop/3x_Car_Rot_South_2_3.pdf', bbox_inches = 'tight')
#plt.savefig('C:/Users/Brendan/Desktop/3x_Car_Rot_South_3plus_30deg_car.jpeg', bbox_inches = 'tight')
#plt.close()