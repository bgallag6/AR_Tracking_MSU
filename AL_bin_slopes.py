# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:22:26 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
import jdcal
import csv
import urllib2
import urllib
from astropy.time import Time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
import jdcal
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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


rot_start = 0
rot_end = 18

number = 0

AL_slopes = np.zeros((18,36))

for c in range(rot_start,rot_end):
#for c in range(3):    

    AL_bins_temp = [0 if x < AL_thresh else x for x in AL_bins[c]]
    AL_nonzero = np.array(np.nonzero(AL_bins_temp))
    print AL_nonzero[0]
    
    AL_slopes_temp = np.zeros((36))
    AL_slopes_count = np.zeros((36))
    count = 0
    for i in range(int(num_bands[c])):
        
        intensity0 = np.array(ARs[i+number,2,:])
        
        xcoords = np.array(ARs[i+number,1,:])

        x_temp = xcoords[intensity0 != 0]
        
        med_long = np.median(x_temp)
        print AL_nonzero[0]
        for r in range(len(AL_nonzero[0])):
            if med_long >= AL_nonzero[0,r]*10 and med_long < (AL_nonzero[0,r]*10 + 10):

                AL_slopes_temp[AL_nonzero[0,r]] += fit_params[i+number,0]
                AL_slopes_count[AL_nonzero[0,r]] += 1
                #print c, med_long
                print c, med_long
    #print AL_slopes_temp
    #print AL_slopes_count
    
    AL_slopes0 = AL_slopes_temp / AL_slopes_count
    #print AL_slopes0
    
    AL_slopes[c] = AL_slopes0
                
    number += int(num_bands[c])
    
#np.save('C:/Users/Brendan/Desktop/AL_slopes_%s.npy' % hemi, AL_slopes)