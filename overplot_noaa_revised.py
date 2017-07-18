# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:40:51 2017

@author: Brendan
"""

"""
###############################
### Comparison NOAA AR Data ###
### and our data  #############
###############################
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

#"""
s = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)
f_names = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/ar_filenames.npy')

#trim = 2922  # image before jump 20140818-20151103
trim = 2872  # last index for end of Carrington rotation

fStart = [11,417,1115,1825,2511]
fEnd = [416,1114,1824,2510,2710]

coord = s.STRS.coordinates  # rectangular box in pixels
cen_coord = s.STRS.centroid_cord  # centroid in degrees
n_regions = s.STRS.n_region
med_inten = s.STRS.median_intensity
tot_int1 = s.STRS.tot_int1
tot_area1 = s.STRS.tot_area1

all_cen_coords = cen_coord.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()

int_thresh = 0

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

fmt = '%Y%m%d'

datesNOAA = []
datesD = []
AR_num = []
Latitude = []
Longitude = []

count = 0

with open('C:/Users/Brendan/Desktop/MSU_Project/Week3/NOAA AR/Full_NOAA_AR.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     
     for row in spamreader:
        if count > 0:
             date = '%i%0.2i%0.2i' % (int(row[0]),int(row[1]),int(row[2]))
             datesD = np.append(datesD, date)
             dt = datetime.datetime.strptime(date[0:8],fmt)
             jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
             datesNOAA = np.append(datesNOAA, jul_date)
             Latitude = np.append(Latitude, int(row[5]))
             Longitude = np.append(Longitude, int(row[4]))
        count += 1
             
dates0 = np.array(datesNOAA)
#dates = dates0*2
datesNOAA -= dates[0]
Latitude0 = np.array(Latitude)
Longitude0 = np.array(Longitude)
             
#frms = dates - dates[[0]]
#frms = frms*2

#frmN = frms[Latitude > 0]
#frmS = frms[Latitude < 0]

x_bins = [20*l for l in range(19)]
x_ticks = [40*l for l in range(10)]


for c in range(5):
#for c in range(1,2):
    
    yr_ind_start = np.searchsorted(datesD,'%i' % (2010+c))
    yr_ind_end = np.searchsorted(datesD,'%i' % (2010+1+c))
    
    date_start = datesD[yr_ind_start]
    date_end = datesD[yr_ind_end-1]
    date_start = '%s/%s/%s' % (date_start[0:4],date_start[4:6],date_start[6:8])
    date_end = '%s/%s/%s' % (date_end[0:4],date_end[4:6],date_end[6:8])
    
    dates_temp = datesNOAA[yr_ind_start:yr_ind_end]
    
    Longitude = Longitude0[yr_ind_start:yr_ind_end]
    Latitude = Latitude0[yr_ind_start:yr_ind_end]
    xN = Longitude[Latitude > 0]
    datesN = dates_temp[Latitude > 0]
    datesN -= datesN[0]
    datesN *= 2
    xS = Longitude[Latitude < 0]
    datesS = dates_temp[Latitude < 0]
    datesS -= datesS[0]
    datesS *= 2
    
    ### our data ###
    #start = dates[11] + (365*c)
    #end = start + (365)
    
    #ind_start = int(np.searchsorted(dates,start))  # dont' think this is exactly correct, but close?
    #ind_end = int(np.searchsorted(dates,end))
    
    ind_start = fStart[c]
    ind_end = fEnd[c]
   
    int_tot = []
    intN_tot = []
    intS_tot = []
    x_tot = []
    xN_tot = []
    xS_tot = []
    y_tot = []
    yN_tot = []
    yS_tot = []
    frm_tot = []
    frmN_tot = []
    frmS_tot = []
    
    date_num_check = 0
    add_half = 0
    
    for i in range(ind_start,ind_end):
        
        if dates[i] == date_num_check:
            frm_num = ((dates[i]-dates[ind_start])*2)+1
        else: 
            frm_num = ((dates[i]-dates[ind_start])*2)
        
        date_num_check = dates[i]
    
        intensities0 = np.array(all_tot_int1[i])
        intensities = intensities0[intensities0 > int_thresh] 
        
        xcoords0 = np.array(all_cen_coords[i])[:,0]
        ycoords0 = np.array(all_cen_coords[i])[:,1]
        
        xcoords = xcoords0[intensities0 > int_thresh]
        ycoords = ycoords0[intensities0 > int_thresh]
        
        xN_temp = xcoords[ycoords > 0]
        xS_temp = xcoords[ycoords < 0]
        intN_temp = intensities[ycoords > 0]
        intS_temp = intensities[ycoords < 0]
        
        #frm_temp = np.array([i-start_frame for y in range(len(xcoords))]) 
        #frmN_temp = np.array([i-start_frame for y in range(len(xN_temp))])
        #frmS_temp = np.array([i-start_frame for y in range(len(xS_temp))])
        frm_temp = np.array([frm_num for y in range(len(xcoords))]) 
        frmN_temp = np.array([frm_num for y in range(len(xN_temp))])
        frmS_temp = np.array([frm_num for y in range(len(xS_temp))])
        
        int_tot = np.append(int_tot, intensities)
        intN_tot = np.append(intN_tot, intN_temp)
        intS_tot = np.append(intS_tot, intS_temp)
        x_tot = np.append(x_tot, xcoords)
        xN_tot = np.append(xN_tot, xN_temp)
        xS_tot = np.append(xS_tot, xS_temp)
        #y_tot = np.append(y_tot, ycoords)
        #yN_tot = np.append(yN_tot, yN_temp)
        #yS_tot = np.append(yS_tot, yS_temp)
        frm_tot = np.append(frm_tot, frm_temp)
        frmN_tot = np.append(frmN_tot, frmN_temp)
        frmS_tot = np.append(frmS_tot, frmS_temp)
    
    frmN_tot -= frmN_tot[0]
    frmS_tot -= frmS_tot[0]
    
    datesN /= 2
    datesS /= 2    
    
    frmN_tot /= 2
    frmS_tot /= 2
    
    frm_bins = [30*i for i in range(13)]   
    long_bins = [60*i for i in range(7)]
    
    tick_size = 19
    
    ### plot North / South Hemispheres scatter
    fig = plt.figure(figsize=(22,10))
    ax1 = plt.gca() 
    ax1.set_title(r'NOAA Nothern Hemisphere' + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)     
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Day', fontsize=font_size)
    #ax1.set_xlim(0,730)
    ax1.set_xlim(0,365)
    ax1.set_ylim(0,360)   
    #ax1.scatter(frmN_tot, xN_tot,color='blue', label='Our Data') 
    ax1.scatter(datesN, xN,color='red', label='NOAA Data') 
    plt.xticks(frm_bins, fontsize=tick_size)
    plt.yticks(long_bins, fontsize=tick_size)
    plt.legend(fontsize=25)
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Data_Overplot_%i_North.pdf' % (2010+c), bbox_inches = 'tight')
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Data_Overplot_%i_North.jpeg' % (2010+c), bbox_inches = 'tight')
    #plt.close()
    
    #"""
    fig = plt.figure(figsize=(22,10))
    ax1 = plt.gca()
    ax1.set_title(r'Southern Hemisphere' + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)    
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Day', fontsize=font_size)
    #ax1.set_xlim(0,730)
    ax1.set_xlim(0,365)
    ax1.set_ylim(0,360)  
    ax1.scatter(frmS_tot, xS_tot,color='blue', label='Our Data')  
    ax1.scatter(datesS, xS, color='red', label='NOAA Data')
    plt.xticks(frm_bins,fontsize=tick_size)
    plt.yticks(long_bins, fontsize=tick_size)
    plt.legend(fontsize=25)
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Data_Overplot_%i_South.pdf' % (2010+c), bbox_inches = 'tight')
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Data_Overplot_%i_South.jpeg' % (2010+c), bbox_inches = 'tight')
    #plt.close()    
    #"""