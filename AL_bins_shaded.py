# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 12:59:08 2017

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

AL_bins = np.load('C:/Users/Brendan/Desktop/AL_smoothing/3x_%s_3sigma_combined.npy' % hemiF)

AL_slopes = np.load('C:/Users/Brendan/Desktop/AL_smoothing/AL_slopes_%s.npy' % hemi)

AL_slopes = np.nan_to_num(AL_slopes)


for i in range(1000):
    if ARs[i,0,0] == 0:
        count = i
        break
    
#"""
s = readsav('C:/Users/Brendan/Desktop/AR_Tracking_MSU/fits_strs_20161219v7.sav')
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

int_thresh = 30

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

rot_start = [0,2,7,11,16]
rot_end = [2,7,11,16,17]  # last is 17+1

frm_correct = [-139,62,9,57,15]  # days in the previous period  (so 80-this)

number = 0

for c in range(5):
#for c in range(1):
    
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
    datesN -= dates_temp[0]
    datesN *= 2
    xS = Longitude[Latitude < 0]
    datesS = dates_temp[Latitude < 0]
    datesS -= dates_temp[0]
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
    
    patches = []
    
    AL_lat = []
    
    med_lat_tot = []
    slopes_tot = []   
        
    count_tot = 0
     
    for c1 in range(rot_start[c], rot_end[c]+1): 
        
        AL_bins_temp = [0 if x < AL_thresh else x for x in AL_bins[c1]]
        AL_nonzero = np.array(np.nonzero(AL_bins_temp))
        frames_rot = 164
        days_rot = frames_rot/2
        deg_bin = 10           
        
        """
        for c2 in range(int(num_bands[c1])):
        
                intensityAL = np.array(ARs[c2+number,2,:])
                framesAL = np.array(ARs[c2+number,0,:])
        
                xcoordsAL = np.array(ARs[c2+number,1,:])
        
                x_tempAL = xcoordsAL[intensityAL != 0]
                
                med_longAL = np.median(x_tempAL)
                
                #print (number+c2), med_longAL
                for r1 in range(len(AL_nonzero[0])):
                    if med_longAL >= AL_nonzero[0,r1]*10 and med_longAL < (AL_nonzero[0,r1]*10 + 10):               
                        AL_slopes = np.append(AL_slopes, fit_params[c2+number,0])
                        print med_longAL, c1, AL_nonzero[0,r1], c2+number
        """
                       
        #number += int(num_bands[c1])        
        #print AL_slopes
        
        print AL_nonzero[0]
        for r in range(len(AL_nonzero[0])):
                       
            long_temp = AL_nonzero[0,r]
            AL_slope = AL_slopes[c1,AL_nonzero[0,r]]
            
            x0 = np.zeros((4))
            y0 = np.zeros((4))
            
            x0[0] = days_rot*c1 - days_rot*rot_start[c] - frm_correct[c]
            x0[1] = days_rot*c1 - days_rot*rot_start[c] - frm_correct[c]
            x0[2] = days_rot*(c1+1) - days_rot*rot_start[c] - frm_correct[c]
            x0[3] = days_rot*(c1+1) - days_rot*rot_start[c] - frm_correct[c]
            y0[0] = long_temp*deg_bin + deg_bin - 9
            y0[1] = long_temp*deg_bin - 9
            y0[2] = long_temp*deg_bin + (days_rot*AL_slope) - 9
            y0[3] = long_temp*deg_bin + deg_bin + (days_rot*AL_slope) - 9
            #print x0,y0
            
            points = zip(x0,y0)
    
            polygon = Polygon(points, True)
            patches.append(polygon)
        
     
    pN = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.3)   
    
   
    
    x00 = 12.5  # 2011=12.5, 2012=2
    rot_rate = 27.25
    """
    for k in range(-1,13):
        x0 = np.zeros((4))
        y0 = np.zeros((4))
        
        x0[0] = x00+(rot_rate*k)
        x0[1] = x00+(rot_rate)+(rot_rate*k)
        x0[2] = x00+(rot_rate*1.5)+(rot_rate*k)
        x0[3] = x00+(rot_rate/2)+(rot_rate*k)
        y0[0] = 360
        y0[1] = 0
        y0[2] = 0
        y0[3] = 360
           
        points = zip(x0,y0)
    
        polygon = Polygon(points, True)
        patches.append(polygon)
    """
    
    AL_deg = 30
    AL_len = 45
    AL_bin = [120,150,150,120]
    """
    for k2 in range(4):
        x0 = np.zeros((4))
        y0 = np.zeros((4))
        
        x0[0] = k2*AL_len
        x0[1] = k2*AL_len
        x0[2] = (k2+1)*AL_len
        x0[3] = (k2+1)*AL_len
        y0[0] = AL_bin[k2]
        y0[1] = AL_bin[k2]+AL_deg
        y0[2] = AL_bin[k2]+AL_deg
        y0[3] = AL_bin[k2]

           
        points = zip(x0,y0)
    
        polygon = Polygon(points, True)
        patches.append(polygon)
        

    pN = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.3)
    pS = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.3)
    """
    
    s = readsav('C:/Users/Brendan/Desktop/AR_Tracking_MSU/2011_sm_euv_str_20170415v9.sav')

    smid = s.STRSMEUV.SMID
    ndetection = s.STRSMEUV.NDETECTION
    smdate = s.STRSMEUV.SMDATE
    smdoy = s.STRSMEUV.SMDOY
    smlon = s.STRSMEUV.SMLON
    smlat = s.STRSMEUV.SMLAT
    smint = s.STRSMEUV.SMINT
    smfrmlm = s.STRSMEUV.SMDFRMLM
    
    
    all_smid = smid.tolist()
    all_ndetection = ndetection.tolist()
    all_smdate = smdate.tolist()
    all_smdoy = smdoy.tolist()
    all_smlon = smlon.tolist()
    all_smlat = smlat.tolist()
    all_smint = smint.tolist()
    all_smfrmlm = smfrmlm.tolist()
    
    #font_size = 21
    
    
    HMI_doy_totN = []
    HMI_lon_totN = []
    HMI_doy_totS = []
    HMI_lon_totS = []
    
    ARs_HMI = np.zeros((25,4,50))
    
    for i in range(len(all_ndetection)):
    #for i in range(1):
        
        ndetect_temp = all_ndetection[i]
        
        smlon_temp = all_smlon[i][0:ndetect_temp]
        smlat_temp = all_smlat[i][0:ndetect_temp]
        smint_temp = all_smint[i][0:ndetect_temp]
        smdate_temp = all_smdate[i][0:ndetect_temp]
        smdoy_temp = all_smdoy[i][0:ndetect_temp]
        smdoy_temp -= 1
        
        #smlon_temp = smlon_temp[smlat_temp > 0]
        #smint_temp = smint_temp[smlat_temp > 0]
        #smdate_temp = smdate_temp[smlat_temp > 0]
        #smdoy_temp = smdoy_temp[smlat_temp > 0]
        #smlat_temp = smlat_temp[smlat_temp > 0]
        
        fmt = '%Y-%m-%d'
    
        datesARs = []
        for q in range(ndetect_temp):
            date = smdate_temp[q][0:10]
            dt = datetime.datetime.strptime(date,fmt)
            jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day)) + float(smdate_temp[q][10:12])
            datesARs = np.append(datesARs, jul_date)
        
        ARs_HMI[i,0,0:ndetect_temp] = datesARs
        ARs_HMI[i,1,0:ndetect_temp] = smlon_temp
        ARs_HMI[i,2,0:ndetect_temp] = smlat_temp
        ARs_HMI[i,3,0:ndetect_temp] = smint_temp
        
        start_date = np.min(ARs_HMI[0,0,0:all_ndetection[0]])
        #dates = ARs_HMI[i,0,0:ndetect_temp] - start_date
        
        if smlat_temp[0] > 0:
            HMI_doy_totN = np.append(HMI_doy_totN, smdoy_temp)
            HMI_lon_totN = np.append(HMI_lon_totN, smlon_temp)
        
        if smlat_temp[0] < 0:
            HMI_doy_totS = np.append(HMI_doy_totS, smdoy_temp)
            HMI_lon_totS = np.append(HMI_lon_totS, smlon_temp)
        
        #plt.figure()
        #plt.plot(smlon_temp,smint_temp)
        #plt.scatter(dates,smlon_temp)
        
    if c == 0:
        datesN += 139
        datesS += 139
        frmN_tot += 139
        frmS_tot += 139
        HMI_doy_totN += 139
        HMI_doy_totS += 139
  
    """
    ### plot North / South Hemispheres scatter
    fig = plt.figure(figsize=(22,10))
    ax1 = plt.gca() 
    ax1.set_title(r'NOAA Northern Hemisphere' + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)     
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Day', fontsize=font_size)
    #ax1.set_xlim(0,730)
    #ax1.set_xlim(0,365)
    ax1.set_ylim(0,360)   
    ax1.scatter(frmN_tot, xN_tot,color='blue', label='Our Data') 
    #ax1.scatter(frmN_tot, xN_tot, intN_tot, color='blue', label='Our Data')
    ax1.scatter(datesN, xN,color='yellow', label='NOAA Data') 
    ax1.scatter(HMI_doy_totN, HMI_lon_totN, color='red', label='HMI Data')    
    #ax1.add_collection(pN)
    plt.xticks(frm_bins, fontsize=tick_size)
    plt.yticks(long_bins, fontsize=tick_size)
    plt.legend(fontsize=25)
    #ax1.set_xlim(0,180)
    ax1.set_xlim(0,365)
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Ours_HMI_Overplot_%i_North_AL_shaded_Example.jpeg' % (2010+c), bbox_inches = 'tight')
    """
    frms_full = np.array([u for u in range(int(np.min(datesS)),int(np.max(datesS)))])
    frms_eq = np.array([(fStart[c]/2)+u for u in range(len(frms_full))])
    
    ### plot North / South Hemispheres scatter
    fig = plt.figure(figsize=(22,10))
    ax1 = plt.gca() 
    ax1.set_title(r'Southern Hemisphere' + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)     
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Day', fontsize=font_size)
    #ax1.set_xlim(0,730)
    #ax1.set_xlim(0,365)
    ax1.set_ylim(0,360)   
    ax1.scatter(frmS_tot, xS_tot,color='blue', label='Our Data') 
    ax1.scatter(datesS, xS,color='orange', label='NOAA Data') 
    #ax1.scatter(HMI_doy_totN, HMI_lon_totN, color='red', label='HMI Data')
    ax1.add_collection(pN)
    ax1.plot(frms_full, 227 - 23.7*(frms_eq/82.) + 1.2*(frms_eq/82.)**2, linewidth=40., alpha=0.3)
    plt.xticks(frm_bins, fontsize=tick_size)
    plt.yticks(long_bins, fontsize=tick_size)
    #plt.legend(fontsize=25)
    #ax1.set_xlim(0,180)
    ax1.set_xlim(0,365)
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Ours_HMI_Overplot_%i_South_AL_zones_fit.jpeg' % (2010+c), bbox_inches = 'tight')
