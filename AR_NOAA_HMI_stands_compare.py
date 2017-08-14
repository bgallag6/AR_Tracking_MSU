# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:53:20 2017

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
from mpl_toolkits.mplot3d import Axes3D

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

int_thresh = 0

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

fmt = '%Y%m%d'

datesNOAA = []
datesD = []
AR_num = []
Latitude = []
Longitude = []
Area = []

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
             Area = np.append(Area, float(row[6]))
        count += 1
             
dates0 = np.array(datesNOAA)
#dates = dates0*2
datesNOAA -= dates[0]
Latitude0 = np.array(Latitude)
Longitude0 = np.array(Longitude)
Area0 = np.array(Area)
             
#frms = dates - dates[[0]]
#frms = frms*2

#frmN = frms[Latitude > 0]
#frmS = frms[Latitude < 0]

x_bins = [20*l for l in range(19)]
x_ticks = [40*l for l in range(10)]


#for c in range(5):
for c in range(2,3):
    
    yr_ind_start = np.searchsorted(datesD,'%i' % (2010+c))
    yr_ind_end = np.searchsorted(datesD,'%i' % (2010+1+c))
    
    date_start = datesD[yr_ind_start]
    date_end = datesD[yr_ind_end-1]
    date_start = '%s/%s/%s' % (date_start[0:4],date_start[4:6],date_start[6:8])
    date_end = '%s/%s/%s' % (date_end[0:4],date_end[4:6],date_end[6:8])
    
    dates_temp = datesNOAA[yr_ind_start:yr_ind_end]
    
    Longitude = Longitude0[yr_ind_start:yr_ind_end]
    Latitude = Latitude0[yr_ind_start:yr_ind_end]
    Area = Area0[yr_ind_start:yr_ind_end]
    xN = Longitude[Latitude > 0]
    datesN = dates_temp[Latitude > 0]
    #datesN -= datesN[0]
    datesN -= dates_temp[0]
    #datesN *= 2
    xS = Longitude[Latitude < 0]
    datesS = dates_temp[Latitude < 0]
    #datesS -= datesS[0]
    datesS -= dates_temp[0]
    #datesS *= 2
    yN = Latitude[Latitude > 0]
    yS = Latitude[Latitude < 0]
    areaN = Area[Latitude > 0]
    areaS = Area[Latitude < 0]
    
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
        
        yN_temp = ycoords[ycoords > 0]
        yS_temp = ycoords[ycoords < 0]
        
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
        y_tot = np.append(y_tot, ycoords)
        yN_tot = np.append(yN_tot, yN_temp)
        yS_tot = np.append(yS_tot, yS_temp)
        frm_tot = np.append(frm_tot, frm_temp)
        frmN_tot = np.append(frmN_tot, frmN_temp)
        frmS_tot = np.append(frmS_tot, frmS_temp)
    
    frmN_tot -= frmN_tot[0]  # should change to subtracting overall first point (otherwise intensity thresh causes shift)
    frmS_tot -= frmS_tot[0]
    
    #datesN /= 2
    #datesS /= 2    
    
    frmN_tot /= 2
    frmS_tot /= 2
    
    frm_bins = [30*i for i in range(13)]   
    long_bins = [60*i for i in range(7)]
    
    tick_size = 19
    
    patches = []
    
   
    
    x00 = 1.5  # 2011=12, 2012=1.5
    rot_rate = 27.25
    
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

    pN = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.3)
    pS = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.3)
    
    s = readsav('C:/Users/Brendan/Desktop/AR_Tracking_MSU/2012_sm_euv_str_20170415v9.sav')

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
    HMI_int_totN = []
    HMI_lat_totN = []
    
    HMI_doy_totS = []
    HMI_lon_totS = []
    HMI_int_totS = []
    HMI_lat_totS = []
    
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
        #print smdoy_temp
        
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
        dates = ARs_HMI[i,0,0:ndetect_temp] - start_date
        
        if smlat_temp[0] > 0:
            HMI_doy_totN = np.append(HMI_doy_totN, smdoy_temp)
            HMI_lon_totN = np.append(HMI_lon_totN, smlon_temp)
            HMI_int_totN = np.append(HMI_int_totN, smint_temp)
            HMI_lat_totN = np.append(HMI_lat_totN, smlat_temp)
        
        if smlat_temp[0] < 0:
            HMI_doy_totS = np.append(HMI_doy_totS, smdoy_temp)
            HMI_lon_totS = np.append(HMI_lon_totS, smlon_temp)
            HMI_int_totS = np.append(HMI_int_totS, smint_temp)
            HMI_lat_totS = np.append(HMI_lat_totS, smlat_temp)
        
        #plt.figure()
        #plt.plot(smlon_temp,smint_temp)
        #plt.scatter(dates,smlon_temp)
    
    ARx_overlap = []
    ARy_overlap = []
    ARint_overlap = []
    ARfrm_overlap = []
    
    NOAAx_overlap = []
    #NOAAy_overlap = []
    NOAAint_overlap = []
    NOAAfrm_overlap = []
  
    
    frm_boundsN_H = [30,30,67,45,68,67,78,120,58,67]  # 2012 North (Longer EUV)
    frm_boundsN_L = [0,0,0,0,30,40,28,60,0,0]  
    long_boundsN_H = [318,220,137,67,210,65,43,308,220,67]  
    long_boundsN_L = [296,204,117,43,187,51,18,280,187,43]  
    lat_boundsN_H = [23,90,90,90,90,90,26,90,90,90]
    lat_boundsN_L = [17,19,0,14,20,0,0,10,19,0]
    
    #frm_boundsN_H = [70,98,110,126,180,135,95,140,165,103,140,165]  # 2011 North
    #frm_boundsN_L = [45,70,55,105,143,100,75,112,145,0,12,12]  
    #long_boundsN_H = [175,168,103,42,177,290,350,354,340,191,352,352]  
    #long_boundsN_L = [150,146,87,18,150,255,320,330,327,148,329,327]  
    #lat_boundsN_H = [30,90,90,26,90,90,90,90,90,90,90,90]
    #lat_boundsN_L = [15,0,0,0,0,8.5,0,0,0,0,0,0]
    
    #frm_boundsN_H = [101,149]  # 2011 South
    #frm_boundsN_L = [40,55]  
    #long_boundsN_H = [56,206]  
    #long_boundsN_L = [27,169]  
    #lat_boundsN_H = [0,0]
    #lat_boundsN_L = [-90,-24]
    
    #frm_boundsN_H = [47,43,153]  # 2012 South (one doesn't have NOAA - error)
    #frm_boundsN_L = [0,13,79]  
    #long_boundsN_H = [111,78,180]  
    #long_boundsN_L = [79,60,120]  
    #lat_boundsN_H = [0,0,0]
    #lat_boundsN_L = [-90,-90,-90]
    
    #frm_boundsN_H = [47,153]  # 2012 South
    #frm_boundsN_L = [0,79]  
    #long_boundsN_H = [111,180]  
    #long_boundsN_L = [79,120]  
    #lat_boundsN_H = [0,0]
    #lat_boundsN_L = [-90,-90]
    
    #frm_boundsN_H = [20]  
    #frm_boundsN_L = [0]  
    #long_boundsN_H = [325]  
    #long_boundsN_L = [285]  
    #lat_boundsN_H = [23]
    #lat_boundsN_L = [17]
    
    AR_complete = np.zeros((len(frm_boundsN_H),12,500))  # ARlong, ARlat, ARint, ARfrm, NOAAlong, NOAAlat, NOAAarea, NOAAfrm, HMIlong, HMIlat, HMIint, HMIfrm

    #"""
    for h in range(len(frm_boundsN_H)):
        countAR = 0
        countNOAA = 0
        countHMI = 0
        
        #"""
        for t in range(len(xN_tot)):
            if (frmN_tot[t] > frm_boundsN_L[h]) and (frmN_tot[t] < frm_boundsN_H[h]) and (xN_tot[t] > long_boundsN_L[h]) and (xN_tot[t] < long_boundsN_H[h]) and (yN_tot[t] > lat_boundsN_L[h]) and (yN_tot[t] < lat_boundsN_H[h]):
            #if (frmN_tot[t] > frm_boundsN_L[h]) and (frmN_tot[t] < frm_boundsN_H[h]) and (xN_tot[t] > long_boundsN_L[h]) and (xN_tot[t] < long_boundsN_H[h]):
                ARx_overlap = np.append(ARx_overlap, xN_tot[t])
                ARy_overlap = np.append(ARy_overlap, yN_tot[t])
                ARint_overlap = np.append(ARint_overlap, intN_tot[t])
                ARfrm_overlap = np.append(ARfrm_overlap, frmN_tot[t])
                AR_complete[h,0,countAR] = xN_tot[t]
                AR_complete[h,1,countAR] = yN_tot[t]
                AR_complete[h,2,countAR] = intN_tot[t]
                AR_complete[h,3,countAR] = frmN_tot[t]
                countAR += 1
        for t in range(len(datesN)):
            if (datesN[t] > frm_boundsN_L[h]) and (datesN[t] < frm_boundsN_H[h]) and (xN[t] > long_boundsN_L[h]) and (xN[t] < long_boundsN_H[h]) and (yN[t] > lat_boundsN_L[h]) and (yN[t] < lat_boundsN_H[h]):
            #if (datesN[t] > frm_boundsN_L[h]) and (datesN[t] < frm_boundsN_H[h]) and (xN[t] > long_boundsN_L[h]) and (xN[t] < long_boundsN_H[h]):
                NOAAx_overlap = np.append(NOAAx_overlap, xN[t])
                #ARy_overlap = np.append(ARy_overlap, yN_tot[t])
                NOAAint_overlap = np.append(NOAAint_overlap, areaN[t])
                NOAAfrm_overlap = np.append(NOAAfrm_overlap, datesN[t])   
                AR_complete[h,4,countNOAA] = xN[t]
                AR_complete[h,5,countNOAA] = yN[t]
                AR_complete[h,6,countNOAA] = areaN[t]
                AR_complete[h,7,countNOAA] = datesN[t]
                countNOAA += 1
        for t in range(len(HMI_doy_totN)):
            if (HMI_doy_totN[t] > frm_boundsN_L[h]) and (HMI_doy_totN[t] < frm_boundsN_H[h]) and (HMI_lon_totN[t] > long_boundsN_L[h]) and (HMI_lon_totN[t] < long_boundsN_H[h]): 
                AR_complete[h,8,countHMI] = HMI_lon_totN[t]
                AR_complete[h,9,countHMI] = HMI_lat_totN[t]
                AR_complete[h,10,countHMI] = HMI_int_totN[t]
                AR_complete[h,11,countHMI] = HMI_doy_totN[t]
                countHMI += 1
        #"""
        
        """
        for t in range(len(xS_tot)):
            if (frmS_tot[t] > frm_boundsN_L[h]) and (frmS_tot[t] < frm_boundsN_H[h]) and (xS_tot[t] > long_boundsN_L[h]) and (xS_tot[t] < long_boundsN_H[h]) and (yS_tot[t] > lat_boundsN_L[h]) and (yS_tot[t] < lat_boundsN_H[h]):
            #if (frmN_tot[t] > frm_boundsN_L[h]) and (frmN_tot[t] < frm_boundsN_H[h]) and (xN_tot[t] > long_boundsN_L[h]) and (xN_tot[t] < long_boundsN_H[h]):
                ARx_overlap = np.append(ARx_overlap, xS_tot[t])
                ARy_overlap = np.append(ARy_overlap, yS_tot[t])
                ARint_overlap = np.append(ARint_overlap, intS_tot[t])
                ARfrm_overlap = np.append(ARfrm_overlap, frmS_tot[t])
                AR_complete[h,0,countAR] = xS_tot[t]
                AR_complete[h,1,countAR] = yS_tot[t]
                AR_complete[h,2,countAR] = intS_tot[t]
                AR_complete[h,3,countAR] = frmS_tot[t]
                countAR += 1
        for t in range(len(datesS)):
            if (datesS[t] > frm_boundsN_L[h]) and (datesS[t] < frm_boundsN_H[h]) and (xS[t] > long_boundsN_L[h]) and (xS[t] < long_boundsN_H[h]) and (yS[t] > lat_boundsN_L[h]) and (yS[t] < lat_boundsN_H[h]):
            #if (datesN[t] > frm_boundsN_L[h]) and (datesN[t] < frm_boundsN_H[h]) and (xN[t] > long_boundsN_L[h]) and (xN[t] < long_boundsN_H[h]):
                NOAAx_overlap = np.append(NOAAx_overlap, xS[t])
                #ARy_overlap = np.append(ARy_overlap, yN_tot[t])
                NOAAint_overlap = np.append(NOAAint_overlap, areaS[t])
                NOAAfrm_overlap = np.append(NOAAfrm_overlap, datesS[t])   
                AR_complete[h,4,countNOAA] = xS[t]
                AR_complete[h,5,countNOAA] = yS[t]
                AR_complete[h,6,countNOAA] = areaS[t]
                AR_complete[h,7,countNOAA] = datesS[t]
                countNOAA += 1
        for t in range(len(HMI_doy_totS)):
            if (HMI_doy_totS[t] > frm_boundsN_L[h]) and (HMI_doy_totS[t] < frm_boundsN_H[h]) and (HMI_lon_totS[t] > long_boundsN_L[h]) and (HMI_lon_totS[t] < long_boundsN_H[h]): 
                AR_complete[h,8,countHMI] = HMI_lon_totS[t]
                AR_complete[h,9,countHMI] = HMI_lat_totS[t]
                AR_complete[h,10,countHMI] = HMI_int_totS[t]
                AR_complete[h,11,countHMI] = HMI_doy_totS[t]
                countHMI += 1
        """
    #"""            
    
    """
    #for qNOAA in range(len(xN)):    
    #    for qAR in range(len(ARx_overlap)):
    #        if (np.abs(xN[qNOAA] - ARx_overlap[qAR]) < 15) and (datesN[qNOAA] == ARfrm_overlap[qAR]): 
    #            NOAAx_overlap = np.append(NOAAx_overlap, xN[qNOAA])
    #            NOAAfrm_overlap = np.append(NOAAfrm_overlap, datesN[qNOAA])
    
    #frm_boundsN_H = [20,27,35,40,55,67,65,105]  # 2012 North
    #frm_boundsN_L = [0,10,10,10,30,40,50,75]  
    #long_boundsN_H = [325,220,135,70,210,65,43,310]  
    long_boundsN_L = [296,204,120,40,185,51,20,280]  
    #lat_boundsN_H = [23,90,90,90,90,90,90,90]
    #at_boundsN_L = [17,0,0,0,0,0,0,0]
    
    for h in range(len(frm_boundsN_H)):
       for t in range(len(datesN)):
           if (datesN[t] > frm_boundsN_L[h]) and (datesN[t] < frm_boundsN_H[h]) and (xN[t] > long_boundsN_L[h]) and (xN[t] < long_boundsN_H[h]):
               NOAAx_overlap = np.append(NOAAx_overlap, xN[t])
               #ARy_overlap = np.append(ARy_overlap, yN_tot[t])
               #ARint_overlap = np.append(ARint_overlap, intN_tot[t])
               NOAAfrm_overlap = np.append(NOAAfrm_overlap, datesN[t])   
    """
     
    ### plot North / South Hemispheres scatter
    fig = plt.figure(figsize=(22,10))
    ax1 = plt.gca() 
    #ax1 = fig.add_subplot(111,projection='3d')
    ax1.set_title(r'Northern Hemisphere' + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)     
    #ax1.set_title(r'Northern Hemisphere' + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)     
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Day', fontsize=font_size)
    #ax1.set_zlabel('Intensity', fontsize=font_size)
    #ax1.set_xlim(0,730)
    #ax1.set_xlim(0,365)
    ax1.set_ylim(0,360)   
    #ax1.set_ylim(-45,45)
    ax1.scatter(frmN_tot, xN_tot,color='blue', label='Our Data') 
    #ax1.scatter(ARfrm_overlap, ARx_overlap, color='blue', label='Our Data') 
    #ax1.scatter(frmS_tot, xS_tot, intS_tot, color='blue', label='Our Data')
    ax1.scatter(datesN, xN,color='red', label='NOAA Data') 
    #ax1.scatter(datesS, xS, areaS, color='orange', label='NOAA Data') 
    #ax1.scatter(NOAAfrm_overlap, NOAAx_overlap,color='orange', label='NOAA Data') 
    #ax1.scatter(HMI_doy_totN, HMI_lon_totN, HMI_int_totN, color='red', label='HMI Data')
    #ax1.scatter(frmN_tot, xN_tot, intN_tot, marker='.', color='blue')   
    #ax1.scatter(xN_tot, yN_tot, intN_tot, marker='.', color='blue')   
    #ax1.scatter(HMI_doy_totN, HMI_lon_totN, color='orange', label='HMI Data') 
    #ax1.scatter(HMI_lon_totN, HMI_lat_totN, HMI_int_totN, marker='.', color='red')       
    #ax1.add_collection(pN)
    #plt.xticks(frm_bins, fontsize=tick_size)
    #plt.yticks(long_bins, fontsize=tick_size)
    plt.legend(fontsize=25)
    #ax1.set_xlim(0,180)
    ax1.set_xlim(0,365)
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Ours_HMI_%i_North_trimB.jpeg' % (2010+c), bbox_inches = 'tight')
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Ours_HMI_%i_South_trim.jpeg' % (2010+c), bbox_inches = 'tight')
    
    """
    ### plot North / South Hemispheres scatter
    fig = plt.figure(figsize=(22,10))
    #ax1 = plt.gca() 
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.set_title(r'NOAA Southern Hemisphere' + '\n Date Range: %s - %s' % (date_start, date_end), y=1.01, fontweight='bold', fontsize=font_size)     
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Day', fontsize=font_size)
    #ax1.set_xlim(0,730)
    #ax1.set_xlim(0,365)
    ax1.set_ylim(0,360)   
    #ax1.scatter(frmS_tot, xS_tot,color='blue', label='Our Data') 
    ax1.scatter(frmS_tot, xS_tot, intS_tot, color='blue', label='Our Data')
    #ax1.scatter(datesS, xS,color='yellow', label='NOAA Data') 
    ax1.scatter(HMI_doy_totS, HMI_lon_totS, HMI_int_totS, color='red', label='HMI Data')
    ax1.add_collection(pS)
    plt.xticks(frm_bins, fontsize=tick_size)
    plt.yticks(long_bins, fontsize=tick_size)
    plt.legend(fontsize=25)
    ax1.set_xlim(0,180)
    #plt.savefig('C:/Users/Brendan/Desktop/NOAA_Ours_HMI_Overplot_%i_South_NearSideB_Int.jpeg' % (2010+c), bbox_inches = 'tight')
    """