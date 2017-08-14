# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 22:56:34 2017

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

abs_dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/EUV_Absolute_Dates.npy')
abs_ARs = np.load('C:/Users/Brendan/Desktop/MSU_Project/EUV_Absolute_ARs.npy')

fStart = [0,406,1104,1814,2500]
fEnd = [405,1103,1813,2499,2784]

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
datesNOAA -= dates0[0]
Latitude0 = np.array(Latitude)
Longitude0 = np.array(Longitude)
             
#frms = dates - dates[[0]]
#frms = frms*2

#frmN = frms[Latitude > 0]
#frmS = frms[Latitude < 0]

x_bins = [20*l for l in range(19)]
x_ticks = [40*l for l in range(10)]


#for c in range(5):
for c in range(1,2):
    
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
    #datesN *= 2
    xS = Longitude[Latitude < 0]
    datesS = dates_temp[Latitude < 0]
    datesS -= datesS[0]
    #datesS *= 2
    
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

    
    for i in range(int(ind_start),int(ind_end)):  # off slightly, not all frames are represented?
                
        longitude = abs_ARs[i,0,:]
        latitude = abs_ARs[i,1,:]
        intensity = abs_ARs[i,2,:]
        frames = abs_ARs[i,3,:] - abs_ARs[ind_start,3,0]
        longitude = longitude[intensity != 0]
        latitude = latitude[intensity != 0]
        frames = frames[intensity != 0]
        intensity = intensity[intensity != 0]
        lonN = longitude[latitude > 0]
        frmN = frames[latitude > 0]
        intN = intensity[latitude > 0]
        latN = latitude[latitude > 0]
        lonS = longitude[latitude < 0]
        frmS = frames[latitude < 0]     
        intS = intensity[latitude < 0]
        latS = latitude[latitude < 0]
        
        int_tot = np.append(int_tot, intensity)
        intN_tot = np.append(intN_tot, intN)
        intS_tot = np.append(intS_tot, intS)
        x_tot = np.append(x_tot, longitude)
        xN_tot = np.append(xN_tot, lonN)
        xS_tot = np.append(xS_tot, lonS)
        y_tot = np.append(y_tot, latitude)
        yN_tot = np.append(yN_tot, latN)
        yS_tot = np.append(yS_tot, latS)
        frm_tot = np.append(frm_tot, frames)
        frmN_tot = np.append(frmN_tot, frmN)
        frmS_tot = np.append(frmS_tot, frmS)    
    
    #datesN /= 2
    #datesS /= 2    
    
    #frmN_tot /= 2
    #frmS_tot /= 2
    
    frm_bins = [30*i for i in range(13)]   
    long_bins = [60*i for i in range(7)]
    
    tick_size = 19
    
    ### plot North / South Hemispheres scatter
    fig = plt.figure(figsize=(22,10))
    ax1 = plt.gca() 
    ax1.set_title(r'NOAA Southern Hemisphere' + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)     
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Day', fontsize=font_size)
    #ax1.set_xlim(0,730)
    #ax1.set_xlim(0,365)
    ax1.set_xlim(0,180)
    ax1.set_ylim(0,360)   
    ax1.scatter(frmS_tot, xS_tot,color='blue', label='Our Data') 
    #ax1.scatter(frmS_tot, xS_tot, intS_tot, color='blue', label='Our Data')
    ax1.scatter(datesS, xS,color='orange', label='NOAA Data') 
    plt.xticks(frm_bins, fontsize=tick_size)
    plt.yticks(long_bins, fontsize=tick_size)

    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Data_Overplot_%i_North.pdf' % (2010+c), bbox_inches = 'tight')
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Data_Overplot_%i_North.jpeg' % (2010+c), bbox_inches = 'tight')
    #plt.close()
    
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

font_size = 21


HMI_doy_tot = []
HMI_lon_tot = []

ARs_HMI = np.zeros((25,4,50))

for i in range(len(all_ndetection)):
#for i in range(1):
    
    ndetect_temp = all_ndetection[i]
    
    smlon_temp = all_smlon[i][0:ndetect_temp]
    smlat_temp = all_smlat[i][0:ndetect_temp]
    smint_temp = all_smint[i][0:ndetect_temp]
    smdate_temp = all_smdate[i][0:ndetect_temp]
    smdoy_temp = all_smdoy[i][0:ndetect_temp]
    
    #smlon_temp = smlon_temp[smlat_temp < 0]
    #smint_temp = smint_temp[smlat_temp < 0]
    #smdate_temp = smdate_temp[smlat_temp < 0]
    #smdoy_temp = smdoy_temp[smlat_temp < 0]
    #smlat_temp = smlat_temp[smlat_temp < 0]
    
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
    dates = ARs_HMI[i,0,0:ndetect_temp] - start_date
    
    
    if smlat_temp[0] < 0:
        HMI_doy_tot = np.append(HMI_doy_tot, smdoy_temp)
        HMI_lon_tot = np.append(HMI_lon_tot, smlon_temp)
    
    #plt.figure()
    #plt.plot(smlon_temp,smint_temp)
    #plt.scatter(dates,smlon_temp)

ax1.scatter(HMI_doy_tot, HMI_lon_tot, color='red', label='HMI Data')
plt.legend(fontsize=25)
#ax1.set_xlim(0,180)
ax1.set_xlim(0,365)
#plt.savefig('C:/Users/Brendan/Desktop/NOAA_Ours_HMI_Overplot_%i_South_Int.jpeg' % (2010+c), bbox_inches = 'tight')