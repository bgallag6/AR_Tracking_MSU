# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:47:26 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.idl import readsav
import jdcal
import datetime

#"""
s = readsav('2012_sm_euv_str_20170415v9.sav')

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

fig = plt.figure(figsize=(22,10))
ax1 = plt.gca() 
ax1.set_title(r'NOAA Nothern Hemisphere' + '\n Date Range: 2011', y=1.01, fontweight='bold', fontsize=font_size)     
ax1.set_ylabel('Longitude', fontsize=font_size)
ax1.set_xlabel('Day', fontsize=font_size)
ax1.set_xlim(0,365)
ax1.set_ylim(0,360)  
frm_bins = [30*i for i in range(13)]   
long_bins = [60*i for i in range(7)]

ARs_HMI = np.zeros((25,4,50))

for i in range(len(all_ndetection)):
#for i in range(1):
    
    ndetect_temp = all_ndetection[i]
    
    smlon_temp = all_smlon[i][0:ndetect_temp]
    smlat_temp = all_smlat[i][0:ndetect_temp]
    smint_temp = all_smint[i][0:ndetect_temp]
    smdate_temp = all_smdate[i][0:ndetect_temp]
    smdoy_temp = all_smdoy[i][0:ndetect_temp]
    
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
    
    #plt.figure()
    #plt.plot(smlon_temp,smint_temp)
    #plt.scatter(dates,smlon_temp)
    ax1.scatter(smdoy_temp,smlon_temp)
    plt.xticks(frm_bins, fontsize=font_size)
    plt.yticks(long_bins, fontsize=font_size)

    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Data_Overplot_%i_North.pdf' % (2010+c), bbox_inches = 'tight')
    #plt.close()

#plt.savefig('C:/Users/Brendan/Desktop/HMI_ARs2012.jpeg', bbox_inches='tight')
    

"""
### need to separate by hemispheres ###
"""