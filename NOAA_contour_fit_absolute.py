# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 14:44:38 2017

@author: Brendan
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.convolution import Box2DKernel, Box1DKernel, convolve
from astropy.modeling.models import Gaussian2D
from shapely import geometry
import matplotlib.path as mplPath
from scipy.io.idl import readsav
import jdcal

abs_dates = np.load('C:/Users/Brendan/Desktop/8_1/NOAA_Absolute_Dates_1977_2016.npy')
abs_ARs = np.load('C:/Users/Brendan/Desktop/8_1/NOAA_Absolute_ARs_1977_2016.npy')
#abs_dates = np.load('C:/Users/Brendan/Desktop/8_1/NOAA_Absolute_Dates.npy')
#abs_ARs = np.load('C:/Users/Brendan/Desktop/8_1/NOAA_Absolute_ARs.npy')

hemi = 'N'
#hemi = 'S'
smooth_x = 6
smooth_y = 3

AL_thresh = 8


if hemi == 'N':
    hemiF = 'North'
elif hemi == 'S':
    hemiF = 'South'

box_1D_kernelX = Box1DKernel(smooth_x) # for full dataset
box_1D_kernelY = Box1DKernel(smooth_y) # for full dataset

count = 0

rotations = 3
car_days = rotations*27.25
#seg = 18
seg = 155

#AR_total = np.zeros((1000,5,1000))  # frame, longitude, intensity, latitude, area
#AR_total = np.zeros((1000,4,1000))  # frame, longitude, intensity, latitude
#AR_total_raw = np.zeros((1000,5,1000))
#AR_total_start_band = np.zeros((1000,5,1000))
AR_total = np.zeros((5000,4,1000))  # frame, longitude, intensity, latitude
cumulative_frames = 0
cumulative_count = 0
cumulative_bands = 0
#fit_params = np.zeros((1000,2))
fit_params = np.zeros((5000,2))

num_bands = np.zeros((int(seg)))

for c in range(int(seg)):
#for c in range(len(ind_start)):
#for c in range(5):
    
    day_start = int(car_days*c)
    day_end = int(car_days*(c+1))
    
    ind_start = np.searchsorted(abs_dates[0],day_start)  # index corresponding to carrington rotations
    ind_end = np.searchsorted(abs_dates[0],day_end)  # index corresponding to carrington rotations
    #print ind_start, ind_end
    
    #date_start = f_names[int(ind_start[c])][0:8]
    #date_end = f_names[int(ind_end[c])][0:8]
    
    date_start = '%s/%s/%s' % (str(abs_dates[2,ind_start])[0:4],str(abs_dates[2,ind_start])[4:6],str(abs_dates[2,ind_start])[6:8])
    date_end = '%s/%s/%s' % (str(abs_dates[2,ind_end])[0:4],str(abs_dates[2,ind_end])[4:6],str(abs_dates[2,ind_end])[6:8])
    
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 23

    # for 4 year data
    #longitude = abs_ARs[0,ind_start:ind_end]
    #latitude = abs_ARs[1,ind_start:ind_end]
    #intensity = abs_ARs[2,ind_start:ind_end]
    #frames = abs_ARs[3,ind_start:ind_end] - abs_ARs[3,ind_start]
    
    longitude = abs_ARs[1,ind_start:ind_end]
    latitude = abs_ARs[2,ind_start:ind_end]
    intensity = abs_ARs[3,ind_start:ind_end]
    frames = abs_ARs[0,ind_start:ind_end] - abs_ARs[3,ind_start]    
    
    #longitude = longitude[intensity != 0]
    #latitude = latitude[intensity != 0]
    #frames = frames[intensity != 0]
    #intensity = intensity[intensity != 0]
    lonN = longitude[latitude > 0]
    frmN = frames[latitude > 0]
    intN = intensity[latitude > 0]
    latN = latitude[latitude > 0]
    lonS = longitude[latitude < 0]
    frmS = frames[latitude < 0]     
    intS = intensity[latitude < 0]
    latS = latitude[latitude < 0]

    """
    for i in range(int(ind_start),int(ind_end)):  # off slightly, not all frames are represented?
                
        longitude = abs_ARs[0,i]
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
    """
    
    x_bins = [20*l for l in range(19)]
    x_ticks = [40*l for l in range(10)]
    
    plt.figure()
    y1, x1, _ = plt.hist((lonN), bins=x_bins)
    elem1 = np.argmax(y1)
    bin_max1 = y1[elem1]
    
    y2, x2, _ = plt.hist(lonS, bins=x_bins)
    elem2 = np.argmax(y2)
    bin_max2 = y2[elem2]
    plt.close()
    
    bin_max = np.max([bin_max1, bin_max2])*1.1
    
    ### plot North / South Hemispheres scatter
    fig = plt.figure(figsize=(22,10))
    plt.suptitle(r'Southern Hemisphere - Carrington Rotation Periods: %i - %i' % ((c*rotations)+1, ((c+1)*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=0.97, fontweight='bold', fontsize=font_size)
    ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
    ax1 = plt.gca()    
    ax1.set_ylabel('Longitude', fontsize=font_size)
    ax1.set_xlabel('Frame', fontsize=font_size)
    ax1.set_ylim(0,360)   
    #ax1.set_xlim(ind_start,ind_end)
    ax1.set_xlim(0,81)
    ax1.scatter(frmS, lonS)  
    
    ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
    ax2 = plt.gca()
    ax2.set_ylabel('Number of ARs', fontsize=font_size)
    ax2.set_xlabel('Longitude', fontsize=font_size)
    #ax2.set_ylim(0,bin_max)  
    ax2.set_xlim(0,360)
    ax2.hist(lonS) 
    #plt.xticks(x_ticks)
    #plt.savefig('C:/Users/Brendan/Desktop/absolute/%s/Car_Rot_%i_%i_%s_%ix%iysmooth.jpg' % (hemiF,(c*rotations)+1, ((c+1)*rotations),hemiF, smooth_x,smooth_y), bbox_inches = 'tight')
    plt.close()

    pad = 18
    
    """
    ### ***
    """
    lonS = [0 if x > 360 else x for x in lonS]
    lonN = [0 if x > 360 else x for x in lonN]
    
    if hemi == "N": 
        matrx = np.zeros((360+pad,int(frmN[-1])-int(frmN[0])+pad+1))
        matrx_lat = np.zeros((360+pad,int(frmN[-1])-int(frmN[0])+pad+1))
        #matrx_area = np.zeros((360+pad,int(frmN_tot[-1])-int(frmN_tot[0])+pad+1))
        
        for i in range(len(frmN)):
            matrx[int(lonN[i])+(pad/2),int(frmN[i])-int(frmN[0])+(pad/2)] = intN[i]
            matrx_lat[int(lonN[i])+(pad/2),int(frmN[i])-int(frmN[0])+(pad/2)] = latN[i]
            #matrx_area[int(xN_tot[i])+(pad/2),int(frmN_tot[i])-int(frmN_tot[0])+(pad/2)] = areaN_tot[i]
    
    elif hemi == "S":
        matrx = np.zeros((360+pad,int(frmS[-1])-int(frmS[0])+pad+1))
        matrx_lat = np.zeros((360+pad,int(frmS[-1])-int(frmS[0])+pad+1))
        #matrx_area = np.zeros((360+pad,int(frmS_tot[-1])-int(frmS_tot[0])+pad+1))
        
        for i in range(len(frmS)):
            matrx[int(lonS[i])+(pad/2),int(frmS[i])-int(frmS[0])+(pad/2)] = intS[i]
            matrx_lat[int(lonS[i])+(pad/2),int(frmS[i])-int(frmS[0])+(pad/2)] = latS[i]
            #matrx_area[int(xS_tot[i])+(pad/2),int(frmS_tot[i])-int(frmS_tot[0])+(pad/2)] = areaS_tot[i]
    
    matrx = np.flipud(matrx)
    matrx_lat = np.flipud(matrx_lat)
    #matrx_area = np.flipud(matrx_area)
        
    matrx[matrx < 1] = np.NaN
    matrx0 = matrx
    matrx = np.nan_to_num(matrx)
    
    smoothed_data_box = np.zeros_like(matrx)
    
    for j in range(matrx.shape[0]):
        smoothed_data_box[j] = convolve(matrx[j], box_1D_kernelX)
        matrx[j] = convolve(matrx[j], box_1D_kernelX)
    
    for i in range(matrx.shape[1]):
        smoothed_data_box[:,i] = convolve(matrx[:,i], box_1D_kernelY)
        matrx[:,i] = convolve(matrx[:,i], box_1D_kernelX)
        
        
    #fig = plt.figure(figsize=(10,10))
    #plt.imshow(matrx,vmin=0,vmax=1.)
    
    #plt.figure()
    #plt.imshow(smoothed_data_box)
    
    
    delta = 1
    x = np.arange(0, matrx.shape[1], delta)
    y = np.arange(0, matrx.shape[0], delta)
    X, Y = np.meshgrid(x, y)
   
    fig = plt.figure(figsize=(22,10))
    #ax = fig.add_subplot(111) # using this gives different results (below 0?)
    ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
    plt.suptitle('%sern Hemisphere: Contours' % hemiF, y=1.01, fontsize=font_size)
    ax1.scatter(frmS, lonS)  
    ax1.set_ylim(-60,420)
    ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
    #plt.setp(ax2.get_xticklabels(), visible=True)
    #plt.setp(ax2.get_yticklabels(), visible=True)
    #ax2.set_xlabel('Frame', fontsize=font_size-3)
    #ax2.set_ylabel('Longitude', fontsize=font_size-3)    
    ax2.imshow(matrx0)
    plt.yticks([-60,0,60,120,180,240,300,360,420],[420,360,300,240,180,120,60,0,-60])
    CS = plt.contour(X, Y, smoothed_data_box, levels=[0])
    #CS = plt.contour(X, Y, matrx, levels=[0])
    plt.clabel(CS, inline=1, fontsize=10)
    ax2.set_ylim(420,-60)
    ax2.set_aspect(0.5)
    
    #plt.savefig('C:/Users/Brendan/Desktop/absolute/%s/%s_contours_%i_%ix%iysmooth.jpeg' % (hemiF, hemiF, c, smooth_x, smooth_y), bbox_inches='tight')
    plt.close()       
       
    level0 = CS.levels[0]
    c0 = CS.collections[0]
    paths = c0.get_paths()
    
    frm_arr = np.zeros((matrx.shape[0],matrx.shape[1]))
    long_arr = np.zeros((matrx.shape[0],matrx.shape[1]))
    
    for k1 in range(matrx.shape[0]):
        long_arr[k1] = [k1 for s in range(matrx.shape[1])]    
        
    for k2 in range(matrx.shape[1]):
        frm_arr[:,k2] = [k2 for s in range(matrx.shape[0])] 
        
    long_arr = np.flipud(long_arr)
    
    #ARs = np.zeros((len(paths),4,500))
    #ARs_corrected = np.zeros((len(paths),4,500))
    ARs = np.zeros((len(paths),5,1000))  # frame, longitude, intensity, latitude, area
    ARs_corrected = np.zeros((len(paths),5,1000))
    
    count = 0  # account for # of bands not significant
    
    for i in range(len(paths)):
    #for i in range(9):
       path = mplPath.Path(paths[i].vertices)
       within = []
       for c1 in range(matrx.shape[0]):
           #points = zip([r for r in range(matrx.shape[0])],[c1 for q in range(matrx.shape[1])])  #works
           points = zip([r for r in range(matrx.shape[1])],[c1 for q in range(matrx.shape[1])])  #if 2nd dim > 1st dim size
           within_temp = path.contains_points(points)
           within = np.append(within, within_temp)
       within = np.reshape(within, (matrx.shape[0],matrx.shape[1]))
       frm = frm_arr[np.where(within == True)]
       lon = long_arr[np.where(within == True)]
       inten = np.nan_to_num(matrx0)[np.where(within == True)]
       lat = matrx_lat[np.where(within == True)]
       #are = matrx_area[np.where(within == True)]
       frmP = frm[np.where(inten > 0)]
       #print paths[i].vertices.shape[0], len(frmP)
       lonP = lon[np.where(inten > 0)]
       intenP = inten[np.where(inten > 0)]
       latP = lat[np.where(inten > 0)]
       #areaP = are[np.where(inten > 0)]
       if len(frmP) < 7:
           count += 1
       else:
           #ARs[i-count,0,0:len(frmP)] = frmP
           ARs[i-count,0,0:len(frmP)] = frmP - (pad/2)  # account for padding introduce originally
           ARs[i-count,1,0:len(frmP)] = lonP - (pad/2)
           ARs[i-count,2,0:len(frmP)] = intenP
           ARs[i-count,3,0:len(frmP)] = latP
           #ARs[i-count,4,0:len(frmP)] = areaP
           
           ARs_corrected[i-count,0,0:len(frmP)] = frmP - (pad/2)
           ARs_corrected[i-count,2,0:len(frmP)] = intenP
           ARs_corrected[i-count,3,0:len(frmP)] = latP
           #ARs_corrected[i-count,4,0:len(frmP)] = areaP
           
           #AR_total[i-count+cumulative_bands,0,0:len(frmP)] = frmP + cumulative_frames
           AR_total[i-count+cumulative_bands,0,0:len(frmP)] = frmP + cumulative_frames - (pad/2)
           AR_total[i-count+cumulative_bands,2,0:len(frmP)] = intenP
           AR_total[i-count+cumulative_bands,3,0:len(frmP)] = latP
           #AR_total[i-count+cumulative_bands,4,0:len(frmP)] = areaP
           
           #AR_total_raw[i-count+cumulative_bands,0,0:len(frmP)] = frmP + cumulative_frames - (pad/2)
           #AR_total_raw[i-count+cumulative_bands,2,0:len(frmP)] = intenP
           #AR_total_raw[i-count+cumulative_bands,3,0:len(frmP)] = latP
           #AR_total_raw[i-count+cumulative_bands,4,0:len(frmP)] = areaP
           
           #AR_total_start_band[i-count+cumulative_bands,0,0:len(frmP)] = frmP + cumulative_frames - (pad/2)
           #AR_total_start_band[i-count+cumulative_bands,2,0:len(frmP)] = intenP
           #AR_total_start_band[i-count+cumulative_bands,3,0:len(frmP)] = latP
           #AR_total_start_band[i-count+cumulative_bands,4,0:len(frmP)] = areaP
    
    cumulative_frames += (np.max(frmN) - np.min(frmN))
    cumulative_count += count
    
    for k in range(len(paths)-count):
    #for k in range(1):
       
       if k == 0:
           fig = plt.figure(figsize=(22,10))
           plt.suptitle(r'%sern Hemisphere - Carrington Rotation Periods: %i - %i' % (hemiF, (c*rotations)+1, ((c+1)*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=0.99, fontweight='bold', fontsize=font_size)
           ax1 = plt.gca()
           ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
           ax2 = plt.gca()
           ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
       #plt.scatter(frmP,lonP,intenP)
       frames = ARs[k,0,:]
       longitudes = ARs[k,1,:]
       intensity = ARs[k,2,:]
       #frames = frames[frames > 0]
       frames = frames[intensity != 0]
       #longitudes = longitudes[longitudes > 0]
       longitudes = longitudes[intensity != 0]
       intensity = intensity[intensity != 0]
       ax1.set_title('Uncorrected', y = 1.01, fontsize = font_size)
       ax1.scatter(frames, longitudes)
       #ax1.set_xlim(0,175)
       ax1.set_ylim(-50,410)
       ax1.set_xlabel('Frame', fontsize = font_size)
       ax1.set_ylabel('Longitude', fontsize = font_size)
        
       #m0, b0 = np.polyfit(frames, longitudes, 1)
       m0, b0 = np.polyfit(frames, longitudes, 1, w=intensity)  # assign higher weight to stronger points
       fit_params[cumulative_bands + k] = [m0,b0]
        
       ax1.plot(frames, m0*frames + b0, 'r-')  
       
       for t in range(len(frames)):
           #ARs_corrected[k,1,t] = ARs[k,1,t] - (ARs[k,0,t] - np.min(frames)) * m  # correct to start of band
           #print np.min(ARs[k,0,:])
           ARs_corrected[k,1,t] = ARs[k,1,t] - (ARs[k,0,t] - np.min(ARs[k,0,:])) * m0  # correct to start of plot
           AR_total[k+cumulative_bands,1,t] = ARs[k,1,t] - (ARs[k,0,t] - np.min(ARs[k,0,:])) * m0  # correct to start of plot
           #AR_total_raw[k+cumulative_bands,1,t] = ARs[k,1,t]
           #AR_total_start_band[k+cumulative_bands,1,t] = ARs[k,1,t]- (ARs[k,0,t] - np.min(frames)) * m0
               
       framesC = ARs_corrected[k,0,:]
       longitudesC = ARs_corrected[k,1,:]
       intensityC = ARs_corrected[k,2,:]
       framesC = framesC[intensityC != 0]
       longitudesC = longitudesC[intensityC != 0.]
       ax2.set_title('Corrected', y = 1.01, fontsize = font_size)
       ax2.scatter(framesC, longitudesC)
       #ax2.set_xlim(0,175)
       ax2.set_ylim(0,378)
       ax2.set_xlabel('Frame', fontsize = font_size)
       ax2.set_ylabel('Longitude', fontsize = font_size)
        
       m, b = np.polyfit(framesC, longitudesC, 1)
        
       ax2.plot(framesC, m*framesC + b, 'r-')   

    #"""
    #plt.savefig('C:/Users/Brendan/Desktop/absolute/%s/%s_AR_Bands_Compare_%i_%ix%iysmooth.jpeg' % (hemiF, hemiF, c,smooth_x,smooth_y))
    plt.close()    
     
    print (len(paths) - count) 
    cumulative_bands += (len(paths) - count)    
    
    num_bands[c] = (len(paths) - count)

np.save('C:/Users/Brendan/Desktop/NOAA_Absolute_bands_%s_3x_%ix%iysmooth_1977.npy' % (hemi,smooth_x,smooth_y), AR_total)
np.save('C:/Users/Brendan/Desktop/NOAA_Absolute_num_bands_%s_3x_%ix%iysmooth_1977.npy' % (hemi,smooth_x,smooth_y), num_bands)
np.save('C:/Users/Brendan/Desktop/NOAA_Absolute_slopes_%s_3x_%ix%iysmooth_1977.npy' % (hemi,smooth_x,smooth_y), fit_params)

"""
    num_bins = 36 
    x_bins = [(360/num_bins)*w for w in range(num_bins+1)]
       
    long_tot = []   
    long_tot_cor = []   
    
    #xbin_scaled = np.zeros((num_bins+1))
    xbin_scaled = np.zeros((num_bins+1+16))  # 16 = allow for 80 deg below 0, 80 deg above 360
       
    for k in range(len(paths)-count):
    #for k in range(1):
    
       long_temp = ARs[k,1,:]
       long_temp = long_temp[long_temp != 0]
       long_tot = np.append(long_tot, long_temp)
         
       long_temp_cor = ARs_corrected[k,1,:]
       long_temp_cor = long_temp_cor[long_temp_cor != 0]
       long_tot_cor = np.append(long_tot_cor, long_temp_cor)
       
       int_temp = ARs_corrected[k,0,:]
       int_temp = int_temp[ARs_corrected[k,1,:] != 0]
       
       for i1 in range(len(int_temp)):
           long_bin = int(np.floor(long_temp_cor[i1]/(360/num_bins)))
           xbin_scaled[long_bin] += int_temp[i1]
           
                   
    fig = plt.figure(figsize=(22,10))
    plt.suptitle(r'Southern Hemisphere - Carrington Rotation Periods: %i - %i' % ((c*rotations)+1, ((c+1)*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=0.99, fontweight='bold', fontsize=font_size)
    
    ax1 = plt.gca()
    ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
    
    ax1.set_title('Uncorrected', y = 1.01, fontsize = font_size)
    ax1.hist(long_tot, x_bins)
    ax1.set_xlabel('Longitude', fontsize = font_size)
    ax1.set_ylabel('Bin Count', fontsize = font_size)
    plt.xlim(0,360)
    
    ax2 = plt.gca()
    ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)  
    
    ax2.set_title('Corrected', y = 1.01, fontsize = font_size)
    ax2.hist(long_tot_cor, x_bins)
    ax2.set_xlabel('Longitude', fontsize = font_size)
    ax2.set_ylabel('Bin Count', fontsize = font_size)
    plt.xlim(0,360)
    
    #plt.savefig('C:/Users/Brendan/Desktop/Car_rots/AR_Histogram_Compare_Car_Rot_%i_%i.pdf' % ((c*rotations)+1, ((c+1)*rotations)))
    plt.close()
    """
    #np.save('C:/Users/Brendan/Desktop/AR_bands_S_lat.npy', AR_total)
    #np.save('C:/Users/Brendan/Desktop/AR_bands_S_lat_raw.npy', AR_total_raw)
    #np.save('C:/Users/Brendan/Desktop/AR_bands_S_lat_start_band.npy', AR_total_start_band)

"""
slopes = fit_params[:,0]
slopes = slopes[slopes != 0.]    
    
fig = plt.figure(figsize=(22,10))
ax1 = plt.gca()
ax1.set_title('Band Slopes: Pre-Correction', y = 1.01, fontsize = font_size)
ax1.hist(slopes)
ax1.set_xlabel('Longitude', fontsize = font_size)
ax1.set_ylabel('Bin Count', fontsize = font_size)
plt.xlim(-1,1)
"""

"""
fig = plt.figure(figsize=(11,10))
ax1 = plt.gca()
ax1.set_title('Active Region Bands [Corrected]', y = 1.01, fontsize = font_size)
ax1.bar(x_bins, xbin_scaled, width=5)
ax1.set_xlabel('Longitude', fontsize = font_size)
ax1.set_ylabel('Total Intensity', fontsize = font_size)
plt.xlim(0,360)

plt.savefig('C:/Users/Brendan/Desktop/AR_Histogram_Intensity.pdf')   
"""

"""
*** add and then subtract the padding allowing for contours to close
* for every frame, find distance from beginning, multiply by slope, bring point back that amount
* get rid of contours that are less than certain points
* increase longitude range to maybe -60-420 deg
"""