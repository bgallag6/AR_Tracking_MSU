# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 20:08:47 2017

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

#"""
s0 = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)
f_names = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/ar_filenames.npy')

#box_1D_kernelX = Box1DKernel(8) # for 3 rotations
#box_1D_kernelY = Box1DKernel(3) # for 3 rotations

box_1D_kernelX = Box1DKernel(5) # for full dataset
box_1D_kernelY = Box1DKernel(2) # for full dataset

#trim = 2922  # image before jump 20140818-20151103
trim = 2872  # last index for end of Carrington rotation

coord = s0.STRS.coordinates  # rectangular box in pixels
cen_coord = s0.STRS.centroid_cord  # centroid in degrees
n_regions = s0.STRS.n_region
med_inten = s0.STRS.median_intensity
tot_int1 = s0.STRS.tot_int1
tot_area1 = s0.STRS.tot_area1

all_cen_coords = cen_coord.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()

int_thresh = 30

count = 0

rotations = 3
seg = ((dates[trim]-dates[11])/27.25)/rotations
  
ind_start = np.zeros((int(seg)))
ind_end = np.zeros((int(seg)))

for i in range(int(seg)):
#for i in range(3):
    
    start = dates[11] + ((27.25*i)*rotations)
    end = start + (27.25*rotations)
    
    ind_start[i] = np.searchsorted(dates,start)  # carrington rotations
    ind_end[i] = np.searchsorted(dates,end)  # carrington rotations

#AR_total = np.zeros((1000,4,500))
#AR_total_raw = np.zeros((1000,4,500))
#AR_total_start_band = np.zeros((1000,4,500))
AR_total = np.zeros((1000,5,1000))  # frame, longitude, intensity, latitude, area
AR_total_raw = np.zeros((1000,5,1000))
AR_total_start_band = np.zeros((1000,5,1000))
cumulative_frames = 0
cumulative_count = 0
cumulative_bands = 0
fit_params = np.zeros((1000,2))

num_bands = np.zeros((int(seg)))

#ind_start = [11,417,1115,1825,2511] # yearly
#ind_end = [416,1114,1824,2510,2710]

for c in range(int(seg)):
#for c in range(len(ind_start)):
#for c in range(3):
    
    date_start = f_names[int(ind_start[c])][0:8]
    date_end = f_names[int(ind_end[c])][0:8]
    
    date_start = '%s/%s/%s' % (date_start[0:4],date_start[4:6],date_start[6:8])
    date_end = '%s/%s/%s' % (date_end[0:4],date_end[4:6],date_end[6:8])
    
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 23
    
    int_tot = []
    intN_tot = []
    intS_tot = []
    area_tot = []
    areaN_tot = []
    areaS_tot = []
    x_tot = []
    xN_tot = []
    xS_tot = []
    y_tot = []
    yN_tot = []
    yS_tot = []
    frm_tot = []
    frmN_tot = []
    frmS_tot = []

    
    for i in range(int(ind_start[c]),int(ind_end[c])):
    
        intensities0 = np.array(all_tot_int1[i])
        intensities = intensities0[intensities0 > int_thresh] 
        
        area0 = np.array(all_tot_area1[i])
        area = area0[intensities0 > int_thresh]
        
        xcoords0 = np.array(all_cen_coords[i])[:,0]
        ycoords0 = np.array(all_cen_coords[i])[:,1]
        
        xcoords = xcoords0[intensities0 > int_thresh]
        ycoords = ycoords0[intensities0 > int_thresh]
        
        xN_temp = xcoords[ycoords > 0]
        xS_temp = xcoords[ycoords < 0]
        yN_temp = ycoords[ycoords > 0]
        yS_temp = ycoords[ycoords < 0]
        intN_temp = intensities[ycoords > 0]
        intS_temp = intensities[ycoords < 0]
        areaN_temp = area[ycoords > 0]
        areaS_temp = area[ycoords < 0]
        
        #frm_temp = np.array([i-start_frame for y in range(len(xcoords))]) 
        #frmN_temp = np.array([i-start_frame for y in range(len(xN_temp))])
        #frmS_temp = np.array([i-start_frame for y in range(len(xS_temp))])
        frm_temp = np.array([i for y in range(len(xcoords))]) 
        frmN_temp = np.array([i for y in range(len(xN_temp))])
        frmS_temp = np.array([i for y in range(len(xS_temp))])
        
        int_tot = np.append(int_tot, intensities)
        intN_tot = np.append(intN_tot, intN_temp)
        intS_tot = np.append(intS_tot, intS_temp)
        area_tot = np.append(area_tot, area)
        areaN_tot = np.append(areaN_tot, areaN_temp)
        areaS_tot = np.append(areaS_tot, areaS_temp)
        x_tot = np.append(x_tot, xcoords)
        xN_tot = np.append(xN_tot, xN_temp)
        xS_tot = np.append(xS_tot, xS_temp)
        y_tot = np.append(y_tot, ycoords)
        yN_tot = np.append(yN_tot, yN_temp)
        yS_tot = np.append(yS_tot, yS_temp)
        frm_tot = np.append(frm_tot, frm_temp)
        frmN_tot = np.append(frmN_tot, frmN_temp)
        frmS_tot = np.append(frmS_tot, frmS_temp)
    
    
    x_bins = [20*l for l in range(19)]
    x_ticks = [40*l for l in range(10)]
    
    plt.figure()
    y1, x1, _ = plt.hist(xN_tot, bins=x_bins)
    elem1 = np.argmax(y1)
    bin_max1 = y1[elem1]
    
    y2, x2, _ = plt.hist(xS_tot, bins=x_bins)
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
    ax1.scatter(frmS_tot, xS_tot)  
    
    ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
    ax2 = plt.gca()
    ax2.set_ylabel('Number of ARs', fontsize=font_size)
    ax2.set_xlabel('Longitude', fontsize=font_size)
    #ax2.set_ylim(0,bin_max)  
    ax2.set_xlim(0,360)
    ax2.hist(xS_tot) 
    #plt.xticks(x_ticks)
    #plt.savefig('C:/Users/Brendan/Desktop/Car_Rot_%i_%i_North.jpg' % ((c*rotations)+1, ((c+1)*rotations)), bbox_inches = 'tight')
    plt.close()

    pad = 18
    
    #"""
    ### South ###
    matrx = np.zeros((360+pad,int(frmS_tot[-1])-int(frmS_tot[0])+pad+1))
    matrx_lat = np.zeros((360+pad,int(frmS_tot[-1])-int(frmS_tot[0])+pad+1))
    matrx_area = np.zeros((360+pad,int(frmS_tot[-1])-int(frmS_tot[0])+pad+1))
    
    for i in range(len(frmS_tot)):
        matrx[int(xS_tot[i])+(pad/2),int(frmS_tot[i])-int(frmS_tot[0])+(pad/2)] = intS_tot[i]
        matrx_lat[int(xS_tot[i])+(pad/2),int(frmS_tot[i])-int(frmS_tot[0])+(pad/2)] = yS_tot[i]
        matrx_area[int(xS_tot[i])+(pad/2),int(frmS_tot[i])-int(frmS_tot[0])+(pad/2)] = areaS_tot[i]
    #"""    
    
    
    """
    ### North ###    
    matrx = np.zeros((360+pad,int(frmN_tot[-1])-int(frmN_tot[0])+pad+1))
    matrx_lat = np.zeros((360+pad,int(frmN_tot[-1])-int(frmN_tot[0])+pad+1))
    matrx_area = np.zeros((360+pad,int(frmN_tot[-1])-int(frmN_tot[0])+pad+1))
    
    for i in range(len(frmN_tot)):
        matrx[int(xN_tot[i])+(pad/2),int(frmN_tot[i])-int(frmN_tot[0])+(pad/2)] = intN_tot[i]
        matrx_lat[int(xN_tot[i])+(pad/2),int(frmN_tot[i])-int(frmN_tot[0])+(pad/2)] = yN_tot[i]
        matrx_area[int(xN_tot[i])+(pad/2),int(frmN_tot[i])-int(frmN_tot[0])+(pad/2)] = areaN_tot[i]
    """

    
    matrx = np.flipud(matrx)
    matrx_lat = np.flipud(matrx_lat)
    matrx_area = np.flipud(matrx_area)
        
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
   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Southern Hemisphere: Contours', y=1.01, fontsize=font_size)
    plt.xlabel('Frame', fontsize=font_size-3)
    plt.ylabel('Longitude', fontsize=font_size-3)    
    ax.imshow(matrx0)
    CS = plt.contour(X, Y, smoothed_data_box, levels=[0])
    plt.clabel(CS, inline=1, fontsize=10)
    ax.set_aspect(0.5)
    #plt.savefig('C:/Users/Brendan/Desktop/South_contours_5x2y_%i.jpeg' % (c), bbox_inches='tight')
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
       are = matrx_area[np.where(within == True)]
       frmP = frm[np.where(inten > 0)]
       #print paths[i].vertices.shape[0], len(frmP)
       lonP = lon[np.where(inten > 0)]
       intenP = inten[np.where(inten > 0)]
       latP = lat[np.where(inten > 0)]
       areaP = are[np.where(inten > 0)]
       if len(frmP) < 7:
           count += 1
       else:
           #ARs[i-count,0,0:len(frmP)] = frmP
           ARs[i-count,0,0:len(frmP)] = frmP - (pad/2)  # account for padding introduce originally
           ARs[i-count,1,0:len(frmP)] = lonP
           ARs[i-count,2,0:len(frmP)] = intenP
           ARs[i-count,3,0:len(frmP)] = latP
           ARs[i-count,4,0:len(frmP)] = areaP
           
           ARs_corrected[i-count,0,0:len(frmP)] = frmP
           ARs_corrected[i-count,2,0:len(frmP)] = intenP
           ARs_corrected[i-count,3,0:len(frmP)] = latP
           ARs_corrected[i-count,4,0:len(frmP)] = areaP
           
           #AR_total[i-count+cumulative_bands,0,0:len(frmP)] = frmP + cumulative_frames
           AR_total[i-count+cumulative_bands,0,0:len(frmP)] = frmP + cumulative_frames - (pad/2)
           AR_total[i-count+cumulative_bands,2,0:len(frmP)] = intenP
           AR_total[i-count+cumulative_bands,3,0:len(frmP)] = latP
           AR_total[i-count+cumulative_bands,4,0:len(frmP)] = areaP
           
           AR_total_raw[i-count+cumulative_bands,0,0:len(frmP)] = frmP + cumulative_frames
           AR_total_raw[i-count+cumulative_bands,2,0:len(frmP)] = intenP
           AR_total_raw[i-count+cumulative_bands,3,0:len(frmP)] = latP
           AR_total_raw[i-count+cumulative_bands,4,0:len(frmP)] = areaP
           
           AR_total_start_band[i-count+cumulative_bands,0,0:len(frmP)] = frmP + cumulative_frames
           AR_total_start_band[i-count+cumulative_bands,2,0:len(frmP)] = intenP
           AR_total_start_band[i-count+cumulative_bands,3,0:len(frmP)] = latP
           AR_total_start_band[i-count+cumulative_bands,4,0:len(frmP)] = areaP
    
    cumulative_frames += (np.max(frmN_tot) - np.min(frmN_tot))
    cumulative_count += count
    
    for k in range(len(paths)-count):
    #for k in range(1):
       
       if k == 0:
           fig = plt.figure(figsize=(22,10))
           plt.suptitle(r'Southern Hemisphere - Carrington Rotation Periods: %i - %i' % ((c*rotations)+1, ((c+1)*rotations)) + '\n Date Range: %s - %s' % (date_start, date_end), y=0.99, fontweight='bold', fontsize=font_size)
           ax1 = plt.gca()
           ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
           ax2 = plt.gca()
           ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
       #plt.scatter(frmP,lonP,intenP)
       frames = ARs[k,0,:]
       longitudes = ARs[k,1,:]
       intensity = ARs[k,2,:]
       #frames = frames[frames > 0]
       frames = frames[intensity > 0]
       #longitudes = longitudes[longitudes > 0]
       longitudes = longitudes[intensity > 0]
       ax1.set_title('Uncorrected', y = 1.01, fontsize = font_size)
       ax1.scatter(frames, longitudes)
       #ax1.set_xlim(0,175)
       ax1.set_ylim(0,378)
       ax1.set_xlabel('Frame', fontsize = font_size)
       ax1.set_ylabel('Longitude', fontsize = font_size)
        
       m0, b0 = np.polyfit(frames, longitudes, 1)
       fit_params[cumulative_bands + k] = [m0,b0]
        
       ax1.plot(frames, m0*frames + b0, 'r-')  
       
       for t in range(len(frames)):
           #ARs_corrected[k,1,t] = ARs[k,1,t] - (ARs[k,0,t] - np.min(frames)) * m  # correct to start of band
           ARs_corrected[k,1,t] = ARs[k,1,t] - (ARs[k,0,t] - np.min(ARs[k,0,:])) * m0  # correct to start of plot
           AR_total[k+cumulative_bands,1,t] = ARs[k,1,t] - (ARs[k,0,t] - np.min(ARs[k,0,:])) * m0  # correct to start of plot
           AR_total_raw[k+cumulative_bands,1,t] = ARs[k,1,t]
           AR_total_start_band[k+cumulative_bands,1,t] = ARs[k,1,t]- (ARs[k,0,t] - np.min(frames)) * m0
               
       framesC = ARs_corrected[k,0,:]
       longitudesC = ARs_corrected[k,1,:]
       framesC = framesC[framesC > 0]
       longitudesC = longitudesC[longitudesC != 0.]
       ax2.set_title('Corrected', y = 1.01, fontsize = font_size)
       ax2.scatter(framesC, longitudesC)
       #ax2.set_xlim(0,175)
       ax2.set_ylim(0,378)
       ax2.set_xlabel('Frame', fontsize = font_size)
       ax2.set_ylabel('Longitude', fontsize = font_size)
        
       m, b = np.polyfit(framesC, longitudesC, 1)
        
       ax2.plot(framesC, m*framesC + b, 'r-')   

    #"""
    plt.savefig('C:/Users/Brendan/Desktop/south_5x2y_30int/South_AR_Bands_Compare_%i_5x2y.jpeg' % (c))
    plt.close()    
     
    print (len(paths) - count) 
    cumulative_bands += (len(paths) - count)    
    
    num_bands[c] = (len(paths) - count)
    
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