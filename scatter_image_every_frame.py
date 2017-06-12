# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:56:58 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.idl import readsav
import jdcal
import matplotlib.image as im

"""
s = readsav('fits_strs_20161219v7.sav')
dates = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/image_jul_dates.npy')
dates = np.array(dates)
f_names = np.load('C:/Users/Brendan/Desktop/MSU_Project/Active_Longitude/ar_filenames.npy')

#trim = 2922  # image before jump 20140818-20151103
trim = 2872
start_frame = 11

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

images = trim-start_frame
num_ar = images

ARs = np.zeros((num_ar,3,images))
count = 0
  
plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

### need to add time component to AR array? - or just put in which image it is
"""

#for i in range(start_frame,trim):
for i in range(1076,trim):
#for i in range(start_frame,20):
    
    start = dates[0] + 0.5*i
    
    dt_begin = 2400000.5
    dt_dif1 = start-dt_begin    
    dt_greg1 = jdcal.jd2gcal(dt_begin,dt_dif1)
    
    file_date = '%s' % f_names[i][0:8]
    frame_date = '%s/%s/%s' % (file_date[0:4], file_date[4:6], file_date[6:8])
    
    intensities = np.array(all_tot_int1[i])
    intensities = [0 if x < 35 else x for x in intensities]  # maybe also eliminate zeros
    xcoords = np.array(all_cen_coords[i])[:,0]
    ycoords = np.array(all_cen_coords[i])[:,1]
    
    num_reg = n_regions[i]
    for k in range(num_reg):  # if got rid zeros, just range(intensities.size)
        if intensities[k] > 0:
            found = 0
            for c in range(count):
                dr = np.sqrt((ARs[c,1,(i-1)]-xcoords[k])**2 + (ARs[c,2,(i-1)]-ycoords[k])**2)
                if dr < 5:
                    ARs[c,0,i] = intensities[k]
                    ARs[c,1,i] = xcoords[k]
                    ARs[c,2,i] = ycoords[k]
                    found = 1  # maybe need to say closest one if more than one found?  don't think should be issue
            if found == 0:
                ARs[count,0,i] = intensities[k]
                ARs[count,1,i] = xcoords[k]
                ARs[count,2,i] = ycoords[k]
                count += 1
    
    fig = plt.figure(figsize=(9,11))

    ax = plt.subplot2grid((11,11),(0, 0), colspan=11, rowspan=5)
    ax = plt.gca()
    ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date: %s' % frame_date, y=1.01, fontweight='bold', fontsize=font_size)
    ax.set_ylabel('Latitude', fontsize=font_size)
    ax.set_xlabel('Longitude', fontsize=font_size)
    img = ax.scatter(xcoords, ycoords,intensities)
    ax.set_xlim(0,360)
    ax.set_ylim(-45,45)
    
    ax2 = plt.subplot2grid((11,11),(6, 0), colspan=11, rowspan=5)
    ax2 = plt.gca()
    image=im.imread('C:/Users/Brendan/Desktop/MSU_Project/fits_images/%s.jpg' % f_names[i])
    ax2.imshow(image)
    ax2.axis('off')
    plt.savefig('C:/Users/Brendan/Desktop/frames/frame_%i_%s.jpeg' % (i,file_date))
    plt.close()
