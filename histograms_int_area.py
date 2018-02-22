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
import matplotlib.path as mplPath
from scipy.io.idl import readsav
import jdcal

#"""
s0 = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/files/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)
f_names = np.load('C:/Users/Brendan/Desktop/files/MSU_Project/Active_Longitude/ar_filenames.npy')

#trim = 2922  # image before jump 20140818-20151103
#trim = 2872  # last index for end of Carrington rotation
trim = 2796  # last index for end of Carrington rotation

# import IDL arrays
coord = s0.STRS.coordinates  # rectangular box in pixels
cen_coord = s0.STRS.centroid_cord  # centroid in degrees
n_regions = s0.STRS.n_region
med_inten = s0.STRS.median_intensity
tot_int1 = s0.STRS.tot_int1
tot_area1 = s0.STRS.tot_area1

# convert IDL arrays to python lists
all_coords = coord.tolist()
all_cen_coords = cen_coord.tolist()
all_med_inten = med_inten.tolist()
all_tot_int1 = tot_int1.tolist()
all_tot_area1 = tot_area1.tolist()
all_scaled_intensity = (np.array(all_tot_int1)/np.array(all_med_inten)[:, np.newaxis]).tolist()

#plt.hist(all_tot_int1)
#"""
#int_thresh = 30
#int_thresh = 0
int_thresh = 24

count = 0

int_tot = []
intN_tot = []
intS_tot = []
x_tot = []  # box width
xN_tot = []
xS_tot = []
y_tot = []  # box height
yN_tot = []
yS_tot = []
area_tot = []
areaN_tot = []
areaS_tot = []
width_tot = []
height_tot = []
long_tot = []
lat_tot = []

# extract values into separate lists
#for i in range(int(ind_start[c]),int(ind_end[c])):
for i in range(11,trim):
#for i in range(11,13):
 
    
    xcoords = np.array(all_cen_coords[i])[:,0]
    ycoords = np.array(all_cen_coords[i])[:,1]
    
    intensities0 = np.array(all_tot_int1[i])
    #intensities = intensities0[intensities0 != 0] 
    
    #xcoords = xcoords0[intensities0 > int_thresh]
    #ycoords = ycoords0[intensities0 > int_thresh]
    
    int_temp = intensities0[ycoords != 0]
    intN_temp = intensities0[ycoords > 0]
    intS_temp = intensities0[ycoords < 0]
    
    long_temp = xcoords[ycoords != 0]
    
    lat_temp = ycoords[ycoords != 0]
     
    area0 = np.array(all_tot_area1[i])
    area_temp = area0[ycoords != 0]
    areaN_temp = area0[ycoords > 0]
    areaS_temp = area0[ycoords < 0]
    
    
    x1coords0 = np.array(all_coords[i])[:,0] / 10
    x2coords0 = np.array(all_coords[i])[:,1] / 10
    x1_temp = x1coords0[ycoords != 0]
    x2_temp = x2coords0[ycoords != 0]
    x1N_temp = x1coords0[ycoords > 0]
    x2N_temp = x2coords0[ycoords > 0]
    x1S_temp = x1coords0[ycoords < 0]
    x2S_temp = x2coords0[ycoords < 0]
    
    y1coords0 = np.array(all_coords[i])[:,2] / 10
    y2coords0 = np.array(all_coords[i])[:,3] / 10
    y1_temp = y1coords0[ycoords != 0]
    y2_temp = y2coords0[ycoords != 0]
    y1N_temp = y1coords0[ycoords > 0]
    y2N_temp = y2coords0[ycoords > 0]
    y1S_temp = y1coords0[ycoords < 0]
    y2S_temp = y2coords0[ycoords < 0]
    
    
    long_temp = long_temp[int_temp > int_thresh]
    lat_temp = lat_temp[int_temp > int_thresh]
    area_temp = area_temp[int_temp > int_thresh]
    x1_temp = x1_temp[int_temp > int_thresh]
    x2_temp = x2_temp[int_temp > int_thresh]
    y1_temp = y1_temp[int_temp > int_thresh]
    y2_temp = y2_temp[int_temp > int_thresh]
    int_temp = int_temp[int_temp > int_thresh]
    
    
    wid_temp = x2_temp - x1_temp
    widN_temp = x2N_temp - x1N_temp
    widS_temp = x2S_temp - x1S_temp
    
    height_temp = y2_temp - y1_temp
    heightN_temp = y2N_temp - y1N_temp
    heightS_temp = y2S_temp - y1S_temp

    
    int_tot = np.append(int_tot, int_temp)
    intN_tot = np.append(intN_tot, intN_temp)
    intS_tot = np.append(intS_tot, intS_temp)
    
    area_tot = np.append(area_tot, area_temp)
    areaN_tot = np.append(areaN_tot, areaN_temp)
    areaS_tot = np.append(areaS_tot, areaS_temp)
    
    width_tot = np.append(width_tot, wid_temp)
    xN_tot = np.append(xN_tot, widN_temp)
    xS_tot = np.append(xS_tot, widS_temp)
    
    height_tot = np.append(height_tot, height_temp)
    yN_tot = np.append(yN_tot, heightN_temp)
    yS_tot = np.append(yS_tot, heightS_temp)
    
    long_tot = np.append(long_tot, long_temp)
    
    lat_tot = np.append(lat_tot, lat_temp)
    



plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

"""
#xmax = np.max(int_max)
int_max = np.percentile(int_tot, 99)
int_med = np.median(int_tot)

fig = plt.figure(figsize=(12,10))
ax1 = plt.gca()
ax1.set_title('Total EUV Plage Region Integrated Intensity: Original', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Intensity [DN / s$\cdot$$\Omega$]', fontsize=font_size)
#y, x, _ = ax1.hist(intN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(intN_tot, bins=max_bin/10, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(int_tot, bins=50, range=(0, xmax))
y, x, _ = ax1.hist(int_tot, bins=50, range=(0,int_max))
#y, x, _ = ax1.hist(int_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
ax1.vlines(30,0,np.max(y)*2., color='red', linestyle='dashed',linewidth=3.,label='Threshold > 30')
ax1.vlines(int_med,0,np.max(y)*2., color='black',linestyle='dashed',linewidth=3.,label='Median = %i' % int(int_med))
#ax1.set_xlim(0,xmax) 
plt.legend(fontsize=font_size)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/Total_Intensity_Histogram_Original.pdf', bbox_inches='tight')


area_max = np.percentile(area_tot, 99)
area_med = np.median(area_tot)

fig = plt.figure(figsize=(12,10))
ax1 = plt.gca()
ax1.set_title('Total EUV Plage Region Area: Original', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Area [$\Omega$]', fontsize=font_size)
#y, x, _ = ax1.hist(areaN_tot, bins=50)  
#y, x, _ = ax1.hist(areaN_tot, bins=50, visible=False) 
#max_bin = x[-1]
#ax1.hist(areaN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(area_tot, bins=50, range=(areaN_min, areaN_max))
y, x, _ = ax1.hist(area_tot, bins=50, range=(0,area_max))
#y, x, _ = ax1.hist(area_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
ax1.vlines(area_med, 0, np.max(y)*2., color='black',linestyle='dashed',linewidth=3.,label='Median = %0.2f' % area_med)
plt.legend(fontsize=font_size)
#ax1.set_xlim(0,areaN_max) 

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/Total_Area_Histogram_Original.pdf', bbox_inches='tight')

width_count = len(width_tot)
lonN_sum = np.sum(xN_tot)
lonS_sum = np.sum(xS_tot)
width_thresh = width_tot[width_tot < 30]
lonN_thresh = xN_tot[xN_tot < 30]
lonS_thresh = xS_tot[xS_tot < 30]
width_thresh_count = len(width_thresh)  
lonN_thresh_sum = np.sum(lonN_thresh)
lonS_thresh_sum = np.sum(lonS_thresh)
width_percent = float(width_thresh_count) / float(width_count)
lonN_percent = lonN_thresh_sum / lonN_sum
lonS_percent = lonS_thresh_sum / lonS_sum


width_max = np.percentile(width_tot, 99)
width_med = np.median(width_tot)

fig = plt.figure(figsize=(12,10))
ax1 = plt.gca()
ax1.set_title('Total Plage Region Longitude Extent: Original', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Longitude Extension [Deg]', fontsize=font_size)
#y, x, _ = ax1.hist(xN_tot, bins=50) 
#y, x, _ = ax1.hist(xN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(xN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.set_ylim(0,2250)
#ax1.hist(width_tot, bins=50, range=(xN_min, xN_max))
y, x, _ = ax1.hist(width_tot, bins=50, range=(0,width_max))
#y, x, _ = ax1.hist(width_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
ax1.vlines(30,0,np.max(y)*2., color='red', linestyle='dashed', linewidth=3., label='30-Degree Bin')
ax1.vlines(width_med, 0, np.max(y)*2., color='black',linestyle='dashed',linewidth=3.,label='Median = %i' % int(width_med))
#ax1.set_xlim(0,xN_max) 
ax1.text(33, 2000, '%i%s Below' % (width_percent*100, '%'), fontsize=font_size)
plt.legend(fontsize=font_size)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/Total_Width_Histogram_Original.pdf', bbox_inches='tight')


height_max = np.percentile(height_tot, 99)
height_med = np.median(height_tot)

latN_sum = np.sum(yN_tot)
latS_sum = np.sum(yS_tot)
latN_thresh = yN_tot[yN_tot < 30]
latS_thresh = yS_tot[yS_tot < 30]
latN_thresh_sum = np.sum(latN_thresh)
latS_thresh_sum = np.sum(latS_thresh)
latN_percent = latN_thresh_sum / latN_sum
latS_percent = latS_thresh_sum / latS_sum

fig = plt.figure(figsize=(12,10))
ax1 = plt.gca()
ax1.set_title('Total Plage Region Latitude Extent: Original', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Latitude Extension [Deg]', fontsize=font_size)
#y, x, _ = ax1.hist(yN_tot, bins=50) 
#y, x, _ = ax1.hist(yN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(yN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(height_tot, bins=50, range=(yN_min, yN_max))
y, x, _ = ax1.hist(height_tot, bins=50, range=(0,height_max))
#y, x, _ = ax1.hist(height_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
ax1.vlines(height_med, 0, np.max(y)*2., color='black',linestyle='dashed',linewidth=3.,label='Median = %i' % int(height_med))
#ax1.set_xlim(0,yN_max) 
plt.legend(fontsize=font_size)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/Total_Height_Histogram_Original.pdf', bbox_inches='tight')



#long_max = np.percentile(long_tot, 99.9)
long_med = np.median(long_tot)
long_max = 360.
long_xticks = [60*i for i in range(7)]

fig = plt.figure(figsize=(15,10))
ax1 = plt.gca()
ax1.set_title('Total Plage Region Centroid Longitude Position | No Threshold: Original', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Longitude [Deg]', fontsize=font_size)
#y, x, _ = ax1.hist(yN_tot, bins=50) 
#y, x, _ = ax1.hist(yN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(yN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(height_tot, bins=50, range=(yN_min, yN_max))
y, x, _ = ax1.hist(long_tot, bins=36, range=(0,long_max))
#y, x, _ = ax1.hist(height_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
ax1.vlines(long_med, 0, np.max(y)*2., color='black',linestyle='dashed',linewidth=3.,label='Median = %i' % int(long_med))
ax1.set_xlim(0,360) 
ax1.set_xticks(long_xticks)
plt.legend(fontsize=font_size-3)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/Total_Longitude_Histogram_0thresh_original.pdf', bbox_inches='tight')


#lat_tot = np.abs(lat_tot)
#lat_max = np.percentile(lat_tot, 99.9)
lat_min = np.percentile(lat_tot, 0.01)
lat_med = np.median(lat_tot)
lat_max = 40.
lat_min = -40.
lat_xticks = [10*i for i in range(-4,5)]

lat_N = lat_tot[lat_tot > 0]
lat_S = lat_tot[lat_tot < 0]
latN_med = np.median(lat_N)
latS_med = np.median(lat_S)

print latN_med, latS_med, long_med


fig = plt.figure(figsize=(15,10))
ax1 = plt.gca()
ax1.set_title('Total Plage Region Centroid Latitude Position | No Threshold: Original', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Latitude [Deg]', fontsize=font_size)
ax1.set_xticks(lat_xticks)
#y, x, _ = ax1.hist(yN_tot, bins=50) 
#y, x, _ = ax1.hist(yN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(yN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(height_tot, bins=50, range=(yN_min, yN_max))
y, x, _ = ax1.hist(lat_tot, bins=40, range=(lat_min,lat_max))
#y, x, _ = ax1.hist(lat_tot, bins=50, range=(0,lat_max))
#y, x, _ = ax1.hist(height_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
#ax1.vlines(lat_med, 0, np.max(y)*2., color='black',linestyle='dashed',linewidth=3.,label='Median = %i' % int(lat_med))
ax1.vlines(latN_med, 0, np.max(y)*2., color='black',linestyle='dashed',linewidth=3.,label='North Median = %i' % int(latN_med))
ax1.vlines(latS_med, 0, np.max(y)*2., color='black',linestyle='dashed',linewidth=3.,label='South Median = %i' % int(latS_med))
ax1.set_xlim(-40,40) 
#ax1.set_xlim(0,40) 
plt.legend(fontsize=font_size-3)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/Total_Latitude_Histogram_0thresh_original.pdf', bbox_inches='tight')

"""

rep_arr = np.array([t for t in range(len(int_tot))])

int_tot = np.array([0 if long_tot[r] > 357 and width_tot[r] < 4 else int_tot[r] for r in rep_arr])
area_tot = np.array([0 if long_tot[r] > 357 and width_tot[r] < 4 else area_tot[r] for r in rep_arr])
height_tot = np.array([0 if long_tot[r] > 357 and width_tot[r] < 4 else height_tot[r] for r in rep_arr])
width_tot = np.array([0 if long_tot[r] > 357 and width_tot[r] < 4 else width_tot[r] for r in rep_arr]) 
lat_tot = np.array([0 if long_tot[r] > 357 and width_tot[r] < 4 else lat_tot[r] for r in rep_arr]) 
long_tot = np.array([0 if long_tot[r] > 357 and width_tot[r] < 4 else long_tot[r] for r in rep_arr]) 

int_tot = np.array([0 if long_tot[r] < 2 and width_tot[r] < 4 else int_tot[r] for r in rep_arr])
area_tot = np.array([0 if long_tot[r] < 2 and width_tot[r] < 4 else area_tot[r] for r in rep_arr])
height_tot = np.array([0 if long_tot[r] < 2 and width_tot[r] < 4 else height_tot[r] for r in rep_arr]) 
width_tot = np.array([0 if long_tot[r] < 2 and width_tot[r] < 4 else width_tot[r] for r in rep_arr]) 
lat_tot = np.array([0 if long_tot[r] < 2 and width_tot[r] < 4 else lat_tot[r] for r in rep_arr]) 
long_tot = np.array([0 if long_tot[r] < 2 and width_tot[r] < 4 else long_tot[r] for r in rep_arr]) 


int_tot = int_tot[int_tot !=0]
area_tot = area_tot[area_tot !=0]
height_tot = height_tot[height_tot !=0]
width_tot = width_tot[width_tot !=0]
lat_tot = lat_tot[lat_tot !=0]
long_tot = long_tot[long_tot !=0]

"""
#xmax = np.max(int_max)
int_min = np.percentile(int_tot, 0.5)
int_max = np.percentile(int_tot, 99.5)
int_med = np.median(int_tot)

fig = plt.figure(figsize=(13,10))
ax1 = plt.gca()
ax1.set_title('(a) Integrated Intensity | No Threshold', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Intensity [DN / s$\cdot$$\Omega$]', fontsize=font_size)
#y, x, _ = ax1.hist(intN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(intN_tot, bins=max_bin/10, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(int_tot, bins=50, range=(0, xmax))
#y, x, _ = ax1.hist(int_tot, bins=50, range=(0,int_max))
y, x, _ = ax1.hist(int_tot, bins=50, range=(int_min,int_max))
#y, x, _ = ax1.hist(int_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
#ax1.vlines(30,0,np.max(y)*2., color='red', linestyle='dashed',linewidth=3.,label='Threshold > 30')
ax1.vlines(int_med,0,np.max(y)*2., color='red',linestyle='dashed',linewidth=3.,label='Median = %i' % int(int_med))
#ax1.set_xlim(0,xmax) 
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(fontsize=font_size-2)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/AL_figures/Total_Intensity_Histogram_%i_rev.pdf' % int_thresh, bbox_inches='tight')


area_min = np.percentile(area_tot, 0.5)
area_max = np.percentile(area_tot, 99.5)
area_med = np.median(area_tot)

fig = plt.figure(figsize=(13,10))
ax1 = plt.gca()
ax1.set_title('(b) Area | No Threshold', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Area [$\Omega$]', fontsize=font_size)
#y, x, _ = ax1.hist(areaN_tot, bins=50)  
#y, x, _ = ax1.hist(areaN_tot, bins=50, visible=False) 
#max_bin = x[-1]
#ax1.hist(areaN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(area_tot, bins=50, range=(areaN_min, areaN_max))
#y, x, _ = ax1.hist(area_tot, bins=50, range=(0,area_max))
y, x, _ = ax1.hist(area_tot, bins=50, range=(area_min,area_max))
#y, x, _ = ax1.hist(area_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
ax1.vlines(area_med, 0, np.max(y)*2., color='red',linestyle='dashed',linewidth=3.,label='Median = %0.2f' % area_med)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(fontsize=font_size-2)
#ax1.set_xlim(0,areaN_max) 

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/AL_figures/Total_Area_Histogram_%i_rev.pdf' % int_thresh, bbox_inches='tight')

width_count = len(width_tot)
lonN_sum = np.sum(xN_tot)
lonS_sum = np.sum(xS_tot)
width_thresh = width_tot[width_tot < 30]
lonN_thresh = xN_tot[xN_tot < 30]
lonS_thresh = xS_tot[xS_tot < 30]
width_thresh_count = len(width_thresh)  
lonN_thresh_sum = np.sum(lonN_thresh)
lonS_thresh_sum = np.sum(lonS_thresh)
width_percent = float(width_thresh_count) / float(width_count)
lonN_percent = lonN_thresh_sum / lonN_sum
lonS_percent = lonS_thresh_sum / lonS_sum


width_min = np.percentile(width_tot, 0.5)
width_max = np.percentile(width_tot, 99.5)
width_med = np.median(width_tot)

fig = plt.figure(figsize=(13,10))
ax1 = plt.gca()
ax1.set_title('(c) Longitude Extent | No Threshold', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Longitude Extension [Deg]', fontsize=font_size)
#y, x, _ = ax1.hist(xN_tot, bins=50) 
#y, x, _ = ax1.hist(xN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(xN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.set_ylim(0,2250)
#ax1.hist(width_tot, bins=50, range=(xN_min, xN_max))
#y, x, _ = ax1.hist(width_tot, bins=50, range=(0,width_max))
y, x, _ = ax1.hist(width_tot, bins=50, range=(width_min,width_max))
#y, x, _ = ax1.hist(width_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
#ax1.vlines(30,0,np.max(y)*2., color='red', linestyle='dashed', linewidth=3., label='30-Degree Bin')
ax1.vlines(width_med, 0, np.max(y)*2., color='red',linestyle='dashed',linewidth=3.,label='Median = %i' % int(width_med))
#ax1.set_xlim(0,xN_max) 
#ax1.text(33, 2000, '%i%s Below' % (width_percent*100, '%'), fontsize=font_size)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(fontsize=font_size-2)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/AL_figures/Total_Width_Histogram_%i_rev.pdf' % int_thresh, bbox_inches='tight')


height_min = np.percentile(height_tot, 0.5)
height_max = np.percentile(height_tot, 99.5)
height_med = np.median(height_tot)

latN_sum = np.sum(yN_tot)
latS_sum = np.sum(yS_tot)
latN_thresh = yN_tot[yN_tot < 30]
latS_thresh = yS_tot[yS_tot < 30]
latN_thresh_sum = np.sum(latN_thresh)
latS_thresh_sum = np.sum(latS_thresh)
latN_percent = latN_thresh_sum / latN_sum
latS_percent = latS_thresh_sum / latS_sum

fig = plt.figure(figsize=(13,10))
ax1 = plt.gca()
ax1.set_title('(d) Latitude Extent | No Threshold', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Latitude Extension [Deg]', fontsize=font_size)
#y, x, _ = ax1.hist(yN_tot, bins=50) 
#y, x, _ = ax1.hist(yN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(yN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(height_tot, bins=50, range=(yN_min, yN_max))
#y, x, _ = ax1.hist(height_tot, bins=50, range=(0,height_max))
y, x, _ = ax1.hist(height_tot, bins=50, range=(height_min,height_max))
#y, x, _ = ax1.hist(height_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
ax1.vlines(height_med, 0, np.max(y)*2., color='red',linestyle='dashed',linewidth=3.,label='Median = %i' % int(height_med))
#ax1.set_xlim(0,yN_max) 
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(fontsize=font_size-2)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/AL_figures/Total_Height_Histogram_%i_rev.pdf' % int_thresh, bbox_inches='tight')
"""


#long_max = np.percentile(long_tot, 99.9)
long_med = np.median(long_tot)
long_max = 360.
long_xticks = [60*i for i in range(7)]

fig = plt.figure(figsize=(13,10))
ax1 = plt.gca()
ax1.set_title('(g) Centroid Longitude Position | Threshold = 24', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Longitude [Deg]', fontsize=font_size)
#y, x, _ = ax1.hist(yN_tot, bins=50) 
#y, x, _ = ax1.hist(yN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(yN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(height_tot, bins=50, range=(yN_min, yN_max))
y, x, _ = ax1.hist(long_tot, bins=36, range=(0,long_max))
#y, x, _ = ax1.hist(height_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
ax1.vlines(long_med, 0, np.max(y)*2., color='red',linestyle='dashed',linewidth=3.,label='Median = %i' % int(long_med))
ax1.set_xlim(0,360) 
ax1.set_xticks(long_xticks)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(fontsize=font_size-3)

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/AL_figures/Total_Longitude_Histogram_%i_rev.pdf' % int_thresh, bbox_inches='tight')


#lat_tot = np.abs(lat_tot)
#lat_max = np.percentile(lat_tot, 99.9)
#lat_min = np.percentile(lat_tot, 0.01)
lat_med = np.median(lat_tot)
lat_max = 40.
lat_min = -40.
lat_xticks = [10*i for i in range(-4,5)]

lat_N = lat_tot[lat_tot > 0]
lat_S = lat_tot[lat_tot < 0]
latN_med = np.median(lat_N)
latS_med = np.median(lat_S)

print latN_med, latS_med, long_med


fig = plt.figure(figsize=(13,10))
ax1 = plt.gca()
ax1.set_title('(h) Centroid Latitude Position | Threshold = 24', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel(r'Latitude [Deg]', fontsize=font_size)
ax1.set_xticks(lat_xticks)
#y, x, _ = ax1.hist(yN_tot, bins=50) 
#y, x, _ = ax1.hist(yN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(yN_tot, bins=50, color='blue')
#ax1.set_xlim(0,max_bin) 
#ax1.hist(height_tot, bins=50, range=(yN_min, yN_max))
y, x, _ = ax1.hist(lat_tot, bins=40, range=(lat_min,lat_max))
#y, x, _ = ax1.hist(lat_tot, bins=50, range=(0,lat_max))
#y, x, _ = ax1.hist(height_tot, bins=50)
ax1.set_ylim(0,np.max(y)*1.1)
#ax1.vlines(lat_med, 0, np.max(y)*2., color='black',linestyle='dashed',linewidth=3.,label='Median = %i' % int(lat_med))
ax1.vlines(latN_med, 0, np.max(y)*2., color='red',linestyle='dashed',linewidth=3.,label='North Median = %i' % int(latN_med))
ax1.vlines(latS_med, 0, np.max(y)*2., color='red',linestyle='dashed',linewidth=3.,label='South Median = %i' % int(latS_med))
ax1.set_xlim(-40,40) 
#ax1.set_xlim(0,40) 
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(fontsize=font_size-3, loc='upper center')

#plt.savefig('C:/Users/Brendan/Desktop/Inbox/AL_figures/Total_Latitude_Histogram_%i_rev.pdf' % int_thresh, bbox_inches='tight')




"""
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


xmax = np.max([intN_max,intS_max])

fig = plt.figure(figsize=(22,10))
ax1 = plt.subplot2grid((1,11),(0,0), colspan=5, rowspan=1)
ax1 = plt.gca()
plt.suptitle('Total EUV Active Region Integrated Intensity', y=0.97, fontsize=font_size)
ax1.set_title('Northern Hemisphere', y=1.01, fontsize=font_size)
ax1.set_ylabel('Bin Count', fontsize=font_size)
ax1.set_xlabel('Intensity', fontsize=font_size)
ax1.vlines(30,0,4000,linestyle='dashed',linewidth=3.,label='Threshold > 30')
#y, x, _ = ax1.hist(intN_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax1.hist(intN_tot, bins=max_bin/10, color='blue')
#ax1.set_xlim(0,max_bin) 
ax1.hist(intN_tot, bins=50, range=(0, xmax))
ax1.set_xlim(0,xmax) 
plt.legend(fontsize=font_size)

ax2 = plt.subplot2grid((1,11),(0,6), colspan=5, rowspan=1)
ax2 = plt.gca()
ax2.set_title('Southern Hemisphere', y=1.01, fontsize=font_size)
ax2.set_ylabel('Bin Count', fontsize=font_size)
ax2.set_xlabel('Intensity', fontsize=font_size)
ax2.vlines(30,0,4000,linestyle='dashed',linewidth=3.,label='Threshold > 30')
#y, x, _ = ax2.hist(intS_tot, bins=50, visible=False) 
#max_bin = int(x[-1])
#ax2.hist(intS_tot, bins=max_bin/10, color='blue')
#ax2.set_xlim(0,max_bin) 
ax2.hist(intS_tot, bins=50, range=(0, xmax))
ax2.set_xlim(0,xmax) 
plt.legend(fontsize=font_size)

#plt.savefig('C:/Users/Brendan/Desktop/Total_Intensity_Histogram_Threshold.pdf', bbox_inches='tight')


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

#plt.savefig('C:/Users/Brendan/Desktop/Total_Area_Histogram.pdf')

lonN_sum = np.sum(xN_tot)
lonS_sum = np.sum(xS_tot)
lonN_thresh = xN_tot[xN_tot < 30]
lonS_thresh = xS_tot[xS_tot < 30]
lonN_thresh_sum = np.sum(lonN_thresh)
lonS_thresh_sum = np.sum(lonS_thresh)
lonN_percent = lonN_thresh_sum / lonN_sum
lonS_percent = lonS_thresh_sum / lonS_sum


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
ax1.vlines(30,0,2250, linestyle='dashed', linewidth=2., label='30-Degree Bin')
ax1.set_ylim(0,2250)
ax1.hist(xN_tot, bins=50, range=(xN_min, xN_max))
ax1.set_xlim(0,xN_max) 
ax1.text(33, 2000, '%i%s Below' % (lonN_percent*100, '%'), fontsize=font_size)

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
ax2.vlines(30,0,2250, linestyle='dashed', linewidth=2., label='30-Degree Bin')
ax2.set_ylim(0,2250)
ax2.hist(xS_tot, bins=50, range=(xS_min, xS_max))
ax2.set_xlim(0,xS_max) 
ax2.text(33, 2000, '%i%s Below' % (lonS_percent*100, '%'), fontsize=font_size)

#plt.savefig('C:/Users/Brendan/Desktop/Total_Width_Histogram_Thresh.jpeg', bbox_inches='tight')


latN_sum = np.sum(yN_tot)
latS_sum = np.sum(yS_tot)
latN_thresh = yN_tot[yN_tot < 30]
latS_thresh = yS_tot[yS_tot < 30]
latN_thresh_sum = np.sum(latN_thresh)
latS_thresh_sum = np.sum(latS_thresh)
latN_percent = latN_thresh_sum / latN_sum
latS_percent = latS_thresh_sum / latS_sum

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

#plt.savefig('C:/Users/Brendan/Desktop/Total_Height_Histogram.pdf')
"""

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