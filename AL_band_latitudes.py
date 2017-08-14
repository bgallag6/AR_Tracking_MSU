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
import scipy.signal


plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
deg = 30
num_bins = 360/deg

fStart = [11,417,1115,1825,2511]
fEnd = [416,1114,1824,2510,2710]

x_bins = [deg*l for l in range(num_bins+1)]
x_bins2 = [deg*l for l in range(num_bins)]

x_ticks = np.array(x_bins) + (deg/2)

hemi = 'N'
#hemi = 'S'
smooth_x = 6  #  5, 6, 8, 10
smooth_y = 3  #  2, 3, 4, 5

AL_thresh = 6


if hemi == 'N':
    hemiF = 'North'
elif hemi == 'S':
    hemiF = 'South'
   
#num_bands = np.load('C:/Users/Brendan/Desktop/MSU_Project/num_bands_S.npy')
num_bands = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/num_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, smooth_x, smooth_y))
#num_bands = num_bands
    
#ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S.npy')
ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_bands_%s_3x_30int_%sx%sysmooth.npy' % (hemi, smooth_x, smooth_y))
#ARs = AR_total

#fit_params = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_slopes.npy')
fit_params = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/AR_slopes_%s_3x_30int_%sx%sysmooth.npy' % (hemi, smooth_x, smooth_y))

AL_bins = np.load('C:/Users/Brendan/Desktop/MSU_Project/AL_smoothing/3x_%s_3sigma_combined.npy' % hemiF)

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
        
        #"""                
        if count_tot == 0:
            #plt.figure(figsize=(9,12))
            plt.figure(figsize=(20,10))
            ax1 = plt.gca()
            plt.ylim(0,360)
            plt.xlim(0,2750)
            #plt.xlim(0,360)  # for sideways
            #plt.ylim(2700,0)  # for sideways
            plt.title('All Bands [AL Bands = Red]: %sern Hemisphere' % hemiF, fontsize=font_size)
            plt.hlines(fEnd[0],0,360, linestyle='dashed')
            plt.hlines(fEnd[1],0,360, linestyle='dashed')
            plt.hlines(fEnd[2],0,360, linestyle='dashed')
            plt.hlines(fEnd[3],0,360, linestyle='dashed')
            #plt.xlabel('Longitude', fontsize=font_size)
            #plt.ylabel('Frame', fontsize=font_size)
            plt.xlabel('Frame', fontsize=font_size)
            plt.ylabel('Longitude', fontsize=font_size)
        plt.scatter(frm_temp,x_temp)
        #plt.scatter(x_temp, frm_temp)  # for sideways
        count_tot += 1
        #"""

        
        #"""
        for r in range(len(AL_nonzero[0])):
            if med_long >= AL_nonzero[0,r]*10 and med_long < (AL_nonzero[0,r]*10 + 10):
                
                AL_lat = np.append(AL_lat, np.median(y_temp))
                AL_slopes = np.append(AL_slopes, fit_params[i+number,0])
                AL_int = np.append(AL_int, int_temp)
                #print c, med_long, np.median(y_temp), fit_params[i+number,0]
                
                """              
                if count_tot == 0:
                    #plt.figure(figsize=(9,12))
                    plt.figure(figsize=(20,10))
                    ax1 = plt.gca()
                    plt.ylim(0,360)
                    #plt.xlim(0,360)  # for sideways
                    #plt.ylim(2700,0)  # for sideways
                    plt.title('Bands Within AL Zone : %sern Hemisphere' % hemiF, fontsize=font_size)
                    plt.hlines(fEnd[0],0,360, linestyle='dashed')
                    plt.hlines(fEnd[1],0,360, linestyle='dashed')
                    plt.hlines(fEnd[2],0,360, linestyle='dashed')
                    plt.hlines(fEnd[3],0,360, linestyle='dashed')
                    plt.xlabel('Longitude', fontsize=font_size)
                    plt.ylabel('Frame', fontsize=font_size)
                """
                plt.scatter(frm_temp,x_temp,color='red')
                #plt.scatter(x_temp, frm_temp)  # for sideways
                #count_tot += 1
                """
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
plt.savefig('C:/Users/Brendan/Desktop/All_Bands_%s_AL_Marked.jpeg' % hemiF, bbox_inches='tight')    

"""
## intensity histogram comparison
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
print int_tot_avg, AL_int_avg

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

plt.figure(figsize=(15,12))
plt.title('All Bands vs Active Longitude Bands: EUV Integrated Intensity', fontsize=font_size+3, y=1.01)
#plt.bar(int_bins2, int_tot_norm, width=10, color='black', alpha=0.5)
#plt.bar(int_bins2, AL_int_norm, width=10, color='blue', alpha=0.5)
plt.plot(int_bins2, int_tot_norm, linewidth=2, color='black', label='All Bands: Average = %i' % int_tot_avg)
plt.plot(int_bins2, AL_int_norm, linewidth=2, color='blue', label='AL Bands: Average = %i' % AL_int_avg)
plt.xlim(0,500)
#plt.ylim(0,1.1)
plt.ylim(0,0.17)
plt.vlines(30,0,1.1,label='Intensity Threshold > 30', linestyle='dashed', linewidth=2)
plt.xlabel('EUV Integrated Intensity', fontsize=font_size)
plt.ylabel('Fraction of Total', fontsize=font_size)
plt.legend(fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/AL_Intensity_Histogram_%s_%iof16_thresh.jpeg' % (hemi,AL_thresh), bbox_inches='tight')    
"""

""" 
## plot slope vs latitude  
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
#plt.text(0.2,1.,'y = %0.2fx + %0.2f' % (m2, b2), fontsize=font_size)
plt.text(-30,1.,'y = %0.2fx + %0.2f' % (m2, b2), fontsize=font_size)  
#plt.xlim(0,0.3)
plt.xlim(0,-40)
plt.ylim(-2,2)
#plt.savefig('C:/Users/Brendan/Desktop/Slopes_AL_Bands_%iof16_thresh.jpeg' % AL_thresh, bbox_inches='tight')    
"""


""" 
## plot latitude vs slope
#med_lat_sin2 = AL_lat  ## AL bands - strict latitude
#med_lat_sin2 = np.sin(np.deg2rad(AL_lat))**2  ## AL bands
#med_lat_sin2 = np.sin(np.deg2rad(lat_tot))**2
med_lat_sin2 = np.sin(np.deg2rad(med_lat_tot))**2  ## all bands

#slopes_days = AL_slopes*2  # AL bands
#slopes_days = slope_tot*2
slopes_days = slopes_tot*2  # all bands

#m2, b2 = np.polyfit(slopes_days, med_lat_sin2, 1)
#o2, m2, b2 = np.polyfit(slopes_days, med_lat_sin2, 2)
#o2, m2, b2 = np.polyfit(med_lat_sin2, slopes_days, 2)

def diffRotLinear(l, a, b):
    return a + b*l
    
nlfit_gp1, nlpcov_gp1 = scipy.optimize.curve_fit(diffRotLinear, med_lat_sin2, slopes_days, method='dogbox', max_nfev=3000)
a1,b1 = nlfit_gp1


if hemi == 'S':
    lat_full = np.array([v for v in range(-40,0)])
elif hemi == 'N':
    lat_full = np.array([v for v in range(0,40)])


lat_full2 = np.sin(np.deg2rad(lat_full))**2
r_val = pearsonr(med_lat_sin2, slopes_days)[0]
print r_val

med_lat_sin0 = np.sort(med_lat_sin2)

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
plt.figure(figsize=(15,10))
#plt.title('Slope vs Latitude: %sern Hemisphere (AL Bands > %i/16)' % (hemiF, AL_thresh), y=1.01, fontsize=font_size)
plt.title('Drift Rate vs Latitude: %sern Hemisphere: All Bands (Smoothing: %ix,%iy)' % (hemiF, smooth_x, smooth_y), y=1.01, fontsize=font_size)
#plt.scatter(med_lat,slopes)
#plt.scatter(slopes_days, med_lat_sin2)
plt.scatter(med_lat_sin2, slopes_days)
#plt.plot(med_lat, m*med_lat + b, 'r-')  
#plt.plot(slopes_days, m2*slopes_days + b2, 'r-')  
#plt.plot(lat_full2, c2*(lat_full2**2) + b2*lat_full2 + a2, color='black', linestyle='solid',linewidth=2.,  label=r'$\omega$ = %0.2f + %0.2f$\sin{^2}\varphi$ + %0.2f$\sin{^4}\varphi$' % (a2, b2, c2))  
#plt.plot(lat_full2, 14.7 - 14.53 - 1.8*lat_full2**2 - 2.4*lat_full2, color='black', linestyle='dashed',linewidth=2.,  label=r'$\omega$ = 0.17 - 2.4$\sin{^2}\varphi$ - 1.8$\sin{^4}\varphi$')
plt.plot(lat_full2, b1*lat_full2 + a1, color='red', linestyle='solid',linewidth=2.,  label=r'$\omega$ = %0.2f + %0.2f$\sin{^2}\varphi$' % (a1, b1))  
#plt.plot(lat_full, c*(lat_full**4) + b*lat_full**2 + a, 'r-')  
#plt.plot(slopes_days, o2*(slopes_days**2) + m2*slopes_days + b2, 'r-')  
#plt.plot(lat_full, 14.7 - 1.8*np.sin(np.deg2rad(lat_full))**4 - 2.4*np.sin(np.deg2rad(lat_full))**2, 'k')
#plt.plot(lat_full, 14.7 - 14.53 - 1.8*np.sin(np.deg2rad(lat_full))**4 - 2.4*np.sin(np.deg2rad(lat_full))**2, 'k')
plt.plot(lat_full2, 14.7 - 14.53 - 2.4*lat_full2, color='red', linestyle='dashed', linewidth=2., label=r'$\omega$ = 0.17 - 2.4$\sin{^2}\varphi$')
#plt.text(0.2,1.,'y = %0.2fx + %0.2f' % (m2, b2), fontsize=font_size)
#plt.text(0.8,-20.,'y = %0.2fx + %0.2f' % (m2, b2), fontsize=font_size)  
#plt.text(0.17,0.8,r'$\omega$ = %0.2f + %0.2f$\sin{^2}\varphi$ + %0.2f$\sin{^4}\varphi$' % (a2, b2, c2), fontsize=font_size)  
#plt.text(0.17,1.,r'$\omega$ = 0.17 - 2.4$\sin{^2}\varphi$ - 1.8$\sin{^4}\varphi$', fontsize=font_size)  
#plt.text(0.17,0.8,r'$\omega$ = %0.2f + %0.2f$\sin{^2}\varphi$ + %0.2f$\sin{^4}\varphi$' % (a2, b2, c2), fontsize=font_size)  
#plt.text(0.17,1.,r'$\omega$ = 0.17 - 2.4$\sin{^2}\varphi$ - 1.8$\sin{^4}\varphi$', fontsize=font_size)  
plt.legend(loc='upper right', fontsize=20)
#plt.xlim(0,0.3)
#plt.xlim(0,-40)
plt.ylim(-1.5,1.5)
plt.xlim(0,0.3)
plt.tick_params(labelsize=font_size)
plt.xlabel(r'$\sin{^2}\varphi$', fontsize=font_size)
plt.ylabel(r'$\omega$  [${^\circ}$/day]', fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/Slopes_%s_AL_Bands_%iof16_thresh_full_equationC.jpeg' % (hemi, AL_thresh), bbox_inches='tight')  
#plt.savefig('C:/Users/Brendan/Desktop/Slopes_%s_All_Bands_full_equation_smooth_%ix%iyD.jpeg' % (hemi, smooth_x, smooth_y), bbox_inches='tight')    
"""