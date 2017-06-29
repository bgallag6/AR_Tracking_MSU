# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:22:53 2017

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

#trim = 2922  # image before jump 20140818-20151103
trim = 2872  # last index for end of Carrington rotation

coord = s0.STRS.coordinates  # rectangular box in pixels
cen_coord = s0.STRS.centroid_cord  # centroid in degrees
n_regions = s0.STRS.n_region
med_inten = s0.STRS.median_intensity
tot_int1 = s0.STRS.tot_int1
tot_area1 = s0.STRS.tot_area1

all_coords = coord.tolist()
all_cen_coords = cen_coord.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()

#plt.hist(all_tot_int1)
#"""
#int_thresh = 30
int_thresh = 0

count = 0

#int_tot = []
intN_tot = []
intS_tot = []
#x_tot = []  # box width
xN_tot = []
xS_tot = []
#y_tot = []  # box height
yN_tot = []
yS_tot = []
#area_tot = []
areaN_tot = []
areaS_tot = []

#for i in range(int(ind_start[c]),int(ind_end[c])):
for i in range(11,trim):
#for i in range(11,13):
 
    
    xcoords = np.array(all_cen_coords[i])[:,0]
    ycoords = np.array(all_cen_coords[i])[:,1]
    
    intensities0 = np.array(all_tot_int1[i])
    #intensities = intensities0[intensities0 != 0] 
    
    #xcoords = xcoords0[intensities0 > int_thresh]
    #ycoords = ycoords0[intensities0 > int_thresh]
    
    intN_temp = intensities0[ycoords > 0]
    intS_temp = intensities0[ycoords < 0]
    
    area0 = np.array(all_tot_area1[i])
    areaN_temp = area0[ycoords > 0]
    areaS_temp = area0[ycoords < 0]
    
    
    x1coords0 = np.array(all_coords[i])[:,0] / 10
    x2coords0 = np.array(all_coords[i])[:,1] / 10
    x1N_temp = x1coords0[ycoords > 0]
    x2N_temp = x2coords0[ycoords > 0]
    x1S_temp = x1coords0[ycoords < 0]
    x2S_temp = x2coords0[ycoords < 0]
    
    y1coords0 = np.array(all_coords[i])[:,2] / 10
    y2coords0 = np.array(all_coords[i])[:,3] / 10
    y1N_temp = y1coords0[ycoords > 0]
    y2N_temp = y2coords0[ycoords > 0]
    y1S_temp = y1coords0[ycoords < 0]
    y2S_temp = y2coords0[ycoords < 0]
    
    widN_temp = x2N_temp - x1N_temp
    widS_temp = x2S_temp - x1S_temp
    
    heightN_temp = y2N_temp - y1N_temp
    heightS_temp = y2S_temp - y1S_temp

    
    #int_tot = np.append(int_tot, intensities)
    intN_tot = np.append(intN_tot, intN_temp)
    intS_tot = np.append(intS_tot, intS_temp)
    #x_tot = np.append(x_tot, xcoords)
    xN_tot = np.append(xN_tot, widN_temp)
    xS_tot = np.append(xS_tot, widS_temp)
    #y_tot = np.append(y_tot, ycoords)
    yN_tot = np.append(yN_tot, heightN_temp)
    yS_tot = np.append(yS_tot, heightS_temp)
    
    areaN_tot = np.append(areaN_tot, areaN_temp)
    areaS_tot = np.append(areaS_tot, areaS_temp)

intN_min = np.percentile(intN_tot, 1)
intN_max = np.percentile(intN_tot, 99)
intS_min = np.percentile(intS_tot, 1)
intS_max = np.percentile(intS_tot, 99)

areaN_min = np.percentile(areaN_tot, 1)
areaN_max = np.percentile(areaN_tot, 99)
areaS_min = np.percentile(areaS_tot, 1)
areaS_max = np.percentile(areaS_tot, 99)

xN_min = np.percentile(xN_tot, 1)
xN_max = np.percentile(xN_tot, 99)
xS_min = np.percentile(xS_tot, 1)
xS_max = np.percentile(xS_tot, 99)

yN_min = np.percentile(yN_tot, 1)
yN_max = np.percentile(yN_tot, 99)
yS_min = np.percentile(yS_tot, 1)
yS_max = np.percentile(yS_tot, 99)

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

fig = plt.figure(figsize=(22,10))
ax1 = plt.subplot2grid((1,11),(0,0), colspan=5, rowspan=1)
ax1 = plt.gca()
plt.suptitle('Total Active Region Intensity', y=0.97, fontsize=font_size)
ax1.set_title('Northern Hemisphere', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel('Intensity', fontsize=font_size)
#y, x, _ = ax1.hist(intN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(intN_tot, bins=max_bin/10, color='blue')
#ax1.set_xlim(0,max_bin) 
ax1.hist(intN_tot, bins=50, range=(intN_min, intN_max))
ax1.set_xlim(0,intN_max) 

ax2 = plt.subplot2grid((1,11),(0,6), colspan=5, rowspan=1)
ax2 = plt.gca()
ax2.set_title('Southern Hemisphere', y=1.01, fontsize=font_size)
ax2.set_ylabel('Bin Count', fontsize=font_size)
ax2.set_xlabel('Intensity', fontsize=font_size)
#y, x, _ = ax2.hist(intS_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax2.hist(intS_tot, bins=max_bin/10, color='blue')
#ax2.set_xlim(0,max_bin) 
ax2.hist(intS_tot, bins=50, range=(intS_min, intS_max))
ax2.set_xlim(0,intS_max) 

plt.savefig('C:/Users/Brendan/Desktop/Total_Intensity_Histogram.pdf')


fig = plt.figure(figsize=(22,10))
ax1 = plt.subplot2grid((1,11),(0,0), colspan=5, rowspan=1)
ax1 = plt.gca()
plt.suptitle('Total Active Region Area', y=0.97, fontsize=font_size)
ax1.set_title('Northern Hemisphere', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel('Area', fontsize=font_size)
#y, x, _ = ax1.hist(areaN_tot, bins=50)  
#y, x, _ = ax1.hist(areaN_tot, bins=50, visible=False) 
#max_bin = x[-1]
#ax1.hist(areaN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
ax1.hist(areaN_tot, bins=50, range=(areaN_min, areaN_max))
ax1.set_xlim(0,areaN_max) 

ax2 = plt.subplot2grid((1,11),(0,6), colspan=5, rowspan=1)
ax2 = plt.gca()
ax2.set_title('Southern Hemisphere', y=1.01, fontsize=font_size)
ax2.set_ylabel('Bin Count', fontsize=font_size)
ax2.set_xlabel('Area', fontsize=font_size)
#y, x, _ = ax2.hist(areaS_tot, bins=50) 
#y, x, _ = ax2.hist(areaS_tot, bins=50, visible=False) 
#max_bin = x[-1]
#ax2.hist(areaS_tot, bins=50, color='blue')
#ax2.set_xlim(0,max_bin) 
ax2.hist(areaS_tot, bins=50, range=(areaS_min, areaS_max))
ax2.set_xlim(0,areaS_max) 

plt.savefig('C:/Users/Brendan/Desktop/Total_Area_Histogram.pdf')


fig = plt.figure(figsize=(22,10))
ax1 = plt.subplot2grid((1,11),(0,0), colspan=5, rowspan=1)
ax1 = plt.gca()
plt.suptitle('Total Active Region Longitude Extent', y=0.97, fontsize=font_size)
ax1.set_title('Northern Hemisphere', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel('Box Width', fontsize=font_size)
#y, x, _ = ax1.hist(xN_tot, bins=50) 
#y, x, _ = ax1.hist(xN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(xN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
ax1.hist(xN_tot, bins=50, range=(xN_min, xN_max))
ax1.set_xlim(0,xN_max) 

ax2 = plt.subplot2grid((1,11),(0,6), colspan=5, rowspan=1)
ax2 = plt.gca()
ax2.set_title('Southern Hemisphere', y=1.01, fontsize=font_size)
ax2.set_ylabel('Bin Count', fontsize=font_size)
ax2.set_xlabel('Box Width', fontsize=font_size)
#y, x, _ = ax2.hist(xS_tot, bins=50) 
#y, x, _ = ax2.hist(xS_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax2.hist(xS_tot, bins=50, color='blue')
#ax2.set_xlim(0,max_bin) 
ax2.hist(xS_tot, bins=50, range=(xS_min, xS_max))
ax2.set_xlim(0,xS_max) 

plt.savefig('C:/Users/Brendan/Desktop/Total_Width_Histogram.pdf')


fig = plt.figure(figsize=(22,10))
ax1 = plt.subplot2grid((1,11),(0,0), colspan=5, rowspan=1)
ax1 = plt.gca()
plt.suptitle('Total Active Region Latitude Extent', y=0.97, fontsize=font_size)
ax1.set_title('Northern Hemisphere', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel('Box Height', fontsize=font_size)
#y, x, _ = ax1.hist(yN_tot, bins=50) 
#y, x, _ = ax1.hist(yN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(yN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
ax1.hist(yN_tot, bins=50, range=(yN_min, yN_max))
ax1.set_xlim(0,yN_max) 

ax2 = plt.subplot2grid((1,11),(0,6), colspan=5, rowspan=1)
ax2 = plt.gca()
ax2.set_title('Southern Hemisphere', y=1.01, fontsize=font_size)
ax2.set_ylabel('Bin Count', fontsize=font_size)
ax2.set_xlabel('Box Height', fontsize=font_size)
#y, x, _ = ax2.hist(yS_tot, bins=50) 
#y, x, _ = ax2.hist(yS_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax2.hist(yS_tot, bins=50, color='blue')
#ax2.set_xlim(0,max_bin) 
ax2.hist(yS_tot, bins=50, range=(yS_min, yS_max))
ax2.set_xlim(0,yS_max) 

plt.savefig('C:/Users/Brendan/Desktop/Total_Height_Histogram.pdf')


"""   
    
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



"""