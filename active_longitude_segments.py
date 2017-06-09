# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 06:23:53 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.idl import readsav

def linear(f, m, b):
    return m*f + b
    
#s = readsav('fits_sample_strs_20161219v7.sav')
s = readsav('fits_strs_20161219v7.sav')

coord = s.STRS.coordinates  # rectangular box in pixels
cen_coord = s.STRS.centroid_cord  # centroid in degrees
n_regions = s.STRS.n_region
med_inten = s.STRS.median_intensity
tot_int1 = s.STRS.tot_int1
tot_area1 = s.STRS.tot_area1
tot_int3 = s.STRS.tot_int3
tot_area3 = s.STRS.tot_area3

all_xcoords = []
all_ycoords = []
all_med_inten = []
all_tot_int1 = []
all_tot_area1 = []
all_tot_int3 = []
all_tot_area3 = []
total_intensity = []
all_scaled_intensity = []

long_scaled_intensity = np.zeros((18))

for i in range(n_regions.size):
    num_reg = n_regions[i]
    all_med_inten = np.append(all_med_inten, med_inten[i])
    temp_int = 0
    for j in range(num_reg):
        all_xcoords = np.append(all_xcoords, cen_coord[i][j][0])
        all_ycoords = np.append(all_ycoords, cen_coord[i][j][1])       
        all_tot_int1 = np.append(all_tot_int1, tot_int1[i][j])
        all_tot_area1 = np.append(all_tot_area1, tot_area1[i][j])
        all_tot_int3 = np.append(all_tot_int3, tot_int3[i][j])
        all_tot_area3 = np.append(all_tot_area3, tot_area3[i][j])
        tempx = cen_coord[i][j][0]
        tempy = cen_coord[i][j][1]
        temp_int += tot_int1[i][j]
        all_scaled_intensity = np.append(all_scaled_intensity, tot_int1[i][j]/med_inten[i])
        for k in range(18):
            if tempx/20. >= k and tempx/20. < k+1:
                long_scaled_intensity[k] += (tot_int1[i][j]/med_inten[i])
    total_intensity = np.append(total_intensity, temp_int)
      

xticks_long = [60*i for i in range(7)]
xticks_lat = [-90+(30*i) for i in range(7)]
    
#"""
seg = 96
n = int(all_xcoords.size/seg)

lat_max = 75
long_max = 100

lat_bin_size = 1
long_bin_size = 2

lat_bins = np.arange(-90, 90 + lat_bin_size, lat_bin_size)
long_bins = np.arange(0, 360 + long_bin_size, long_bin_size)

lat_num = lat_bins.size
long_num = long_bins.size

lat_mode = np.zeros((seg,2))
long_mode = np.zeros((seg))

xticks_long = [60*i for i in range(7)]
xticks_lat = [-90+(30*i) for i in range(7)]

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

for i in range(seg):
    
    fig = plt.figure(figsize=(22,11))

    plt.suptitle(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date Range: 2010/05/13 - 2016/05/14' + '\n %i of %i' % (i+1, seg), y=0.96, fontweight='bold', fontsize=font_size)
    
    ax = plt.subplot2grid((11,11),(1, 0), colspan=5, rowspan=10)
    ax = plt.gca()
    ax.set_title(r'Longitude', y = 1.01, fontsize=25)
    ax.set_xlim(0,360)
    ax.set_ylim(0,long_max)
    ax.set_ylabel('Count', fontsize=font_size)
    ax.set_xlabel('Degrees', fontsize=font_size)
    ax.set_xticks(xticks_long)
    ax.tick_params(axis='both', labelsize=font_size, pad=7)
    y, x, _ = ax.hist(all_xcoords[n*i:n*(i+1)],bins=36)
    num=y
    bins=x
    elem = np.argmax(num)
    bin_max = bins[elem]
    long_mode[i] = bin_max
    plt.vlines(bin_max, 0, long_max, color='blue', linestyle='dotted', linewidth=1.5, label='mode = %0.1f$^\circ$' % bin_max)   
    legend = plt.legend(loc='upper right', prop={'size':20}, labelspacing=0.35)
    for label in legend.get_lines():
        label.set_linewidth(2.0)  # the legend line width    
    
    ax1 = plt.subplot2grid((11,11),(1, 6), colspan=5, rowspan=10)
    ax1 = plt.gca()
    ax1.set_title(r'Latitude', y = 1.01, fontsize=25)
    ax1.set_xlim(-90,90)
    ax1.set_ylim(0,lat_max)
    ax1.set_ylabel('Count', fontsize=font_size)
    ax1.set_xlabel('Degrees', fontsize=font_size)
    ax1.set_xticks(xticks_lat)
    ax1.tick_params(axis='both', labelsize=font_size, pad=7)
    y, x, _ = ax1.hist(all_ycoords[n*i:n*(i+1)],lat_bins)
    num1=y[0:lat_num/2]
    num2=y[lat_num/2:]
    bins1=x[0:lat_num/2]
    bins2=x[lat_num/2:]
    elem1 = np.argmax(num1)
    elem2 = np.argmax(num2)
    bin_max1 = bins1[elem1]
    bin_max2 = bins2[elem2]
    lat_mode[i,0] = bin_max1
    lat_mode[i,1] = bin_max2
    plt.vlines(bin_max1, 0, lat_max, color='blue', linestyle='dotted', linewidth=1.5, label=r'mode = %0.1f$^\circ$' % bin_max1) 
    plt.vlines(bin_max2, 0, lat_max, color='blue', linestyle='dotted', linewidth=1.5, label=r'mode =  %0.1f$^\circ$' % bin_max2)
    legend = plt.legend(loc='upper right', prop={'size':20}, labelspacing=0.35)
    for label in legend.get_lines():
        label.set_linewidth(2.0)
            
    #plt.savefig('C:/Users/Brendan/Desktop/%i_of_%i.pdf' % ((i+1),seg), format='pdf')
    #plt.savefig('C:/Users/Brendan/Desktop/96_seg/%i_of_%i.jpeg' % ((i+1),seg))
    #plt.close()
#"""    

lat_yticks = [-30+(6*i) for i in range(11)]
time_seg = [i for i in range(1,seg+1)]

f = np.array(time_seg)
s1 = lat_mode[:,0]
s2 = lat_mode[:,1]

m1, b1 = scipy.optimize.curve_fit(linear, f, s1)[0]  # replaced #'s with arrays
m2, b2 = scipy.optimize.curve_fit(linear, f, s2)[0]  # replaced #'s with arrays

line_fit1 = linear(f,m1,b1)
line_fit2 = linear(f,m2,b2)

fig = plt.figure(figsize=(22,11))

plt.title(r'304 $\AA$ 12-Hour Carrington Full-Surface Maps' + '\n Date Range: 2010/05/13 - 2016/05/14', y=1.01, fontweight='bold', fontsize=font_size)
plt.ylabel('Latitude [Deg]', fontsize=font_size)
plt.xlabel('Time Segment (out of %i)' % seg, fontsize=font_size)
plt.xlim(0,seg+1)
plt.ylim(-30,30)
plt.yticks(lat_yticks, fontsize=font_size)
#plt.xticks(axis='both', labelsize=font_size, pad=7)
plt.plot(time_seg, lat_mode[:,0], 'k', linewidth=2.)
plt.plot(time_seg, lat_mode[:,1], 'b', linewidth=2.)
plt.plot(f,line_fit1, 'k', linestyle='dashed', linewidth=1.5)
plt.plot(f,line_fit2, 'b', linestyle='dashed', linewidth=1.5)

#plt.savefig('C:/Users/Brendan/Desktop/Latitude_Time_24.pdf', format='pdf')


    
        
        