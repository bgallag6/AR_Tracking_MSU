# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:43:35 2017

@author: Brendan
"""

"""
#################################################
### interactive tool so you can click ###########
### on a scatter point of duration/intensity  ###
### and displays AR life statistics #############
#################################################
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.idl import readsav
import jdcal

def onclick(event):
    global ix, iy, c, ind, count
    ixx, iyy = event.xdata, event.ydata
    ax2.clear()
    ax3.clear()
    ax4.clear()
    plt.draw()
    print ('x = %d, y = %d' % ( ixx, iyy))  # print location of pixel
    ix = int(ixx)
    iy = int(iyy)
    #ind = 0
    
    ARs_copy = np.zeros_like((ARs))
    avg_int_copy = np.zeros_like((avg_int))
    max_int_copy = np.zeros_like((max_int))
    frames_copy = np.zeros_like((frames))
    frms_copy = np.zeros_like((frms))

    
    ARs_copy[:,0,:] = ARs[:,0,:]
    ARs_copy[:,1,:] = ARs[:,1,:]
    ARs_copy[:,2,:] = ARs[:,2,:]
    avg_int_copy[:] = avg_int[:]
    max_int_copy[:] = max_int[:]
    frames_copy[:] = frames[:]
    frms_copy[:] = frms[:]

    for q in range(count):
        if max_int_copy[q] >= ix-2 and max_int_copy[q] <= ix+2 and frames_copy[q] >= iy-2 and frames_copy[q] <= iy+2:         
            ind = q
            break
    print ind    
    ar0 = ARs_copy[ind,:,:]
    
    x_coords = ar0[1][ar0[1] != 0]
    
    #y_coords = ar0[2][ar0[2] != 0]
    int_ar0 = ar0[2][ar0[0] != 0]
    #area_ar0 = ar0[3][ar0[3] != 0]
    
    num_frm = int(frames_copy[q])
    
    frm_arr = ar0[0][ar0[0] != 0]
    
    color_copy = np.zeros((num_frm,3))
    for t in range(num_frm):
        color_copy[t][0] = t*(1./num_frm)
        color_copy[t][2] = 1. - t*(1./num_frm)
    
    print len(frm_arr), len(x_coords), len(int_ar0)
    #ax2.plot(area_ar0)
    ax2.set_ylabel('Longitude', fontsize=font_size)
    ax2.set_xlabel('Frame', fontsize=font_size)
    #ax2.scatter(frm_arr, x_coords, int_ar0, c=color_copy)
    ax2.scatter(frm_arr, x_coords, int_ar0)
      
    ax3.set_ylabel('Total Intensity', fontsize=font_size)
    ax3.set_xlabel('Frame', fontsize=font_size)
    
    ax4.plot(int_ar0)
    ax4.set_ylabel('Total Intensity', fontsize=font_size)
    ax4.set_xlabel('Frame', fontsize=font_size)
    
    #ax4.plot(x_coords, y_coords)
    print "here"
    
    plt.draw()
    return ix,iy

global count

ARs = np.load('C:/Users/Brendan/Desktop/AR_bands.npy')
for i in range(500):
    if ARs[i,0,0] == 0:
        count = i
        break
                
global frames, avg_int, area
frames = np.zeros((count))
first = np.zeros((count))
x_form = np.zeros((count))
y_form = np.zeros((count))
x_end = np.zeros((count))
y_end = np.zeros((count))
x_avg = np.zeros((count))
y_avg = np.zeros((count))
avg_int = np.zeros((count))
med_int = np.zeros((count))
max_int = np.zeros((count))
distance = np.zeros((count))
area = np.zeros((count))

for c in range(count): 

    ar0 = ARs[c,:,:]
    
    frms = ar0[0][ar0[0] != 0]
    frames[c] = (np.max(frms) - np.min(frms)) # - how many frames AR lasts
    
    first[c] = np.min(ar0[0,:])
    
    #for d in range(count):
    #    if ar0[0,d] != 0:
    #        first[c] = d
    #        break
    
    ### could use a sort for these next few
    #x_form[c] = ar0[1][ar0[1] != 0][0]
    
    #y_form[c] = ar0[2][ar0[2] != 0][0]
    #x_end[c] = ar0[1][ar0[1] != 0][-1]
    
    #y_end[c] = ar0[2][ar0[2] != 0][-1]
    
    x_coords = ar0[1][ar0[1] != 0]
    x_form[c] = np.average(ar0[1][np.where(ar0[0,:] == np.min(frms))])
    x_end[c] = np.average(ar0[1][np.where(ar0[0,:] == np.max(frms))])
    #y_coords = ar0[2][ar0[2] != 0]
    x_avg[c] = np.average(x_coords)
    #y_avg[c] = np.average(y_coords)
    
    avg_int[c] = np.average(ar0[2][ar0[2] != 0])
    max_int[c] = np.max(ar0[2][ar0[2] != 0])    
    #med_int[c] = np.median(ar0[0][ar0[0] != 0])
    
    #area = ar0[3][ar0[3] != 0]
    
    #dist_steps = np.zeros((int(frames[c]-1)))    
    
    #for r in range(int(frames[c])-1):
    #    dist_steps[r] = np.sqrt((x_coords[r+1]-x_coords[r])**2 + (y_coords[r+1]-y_coords[r])**2)
    #distance[c] = np.sum(dist_steps)



if 1:
    
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 17
    
    fig = plt.figure(figsize=(22,11))
    plt.suptitle('Active Region Statistics Summary', fontsize=23, y=0.97)
    ax1 = plt.gca()
    ax1 = plt.subplot2grid((11,11),(0, 0), colspan=5, rowspan=5)
    ax1.set_ylabel('Duration', fontsize=font_size)
    ax1.set_xlabel('Max Intensity', fontsize=font_size)
    #coll = ax1.scatter(avg_int, frames, picker = 5)
    coll = ax1.scatter(max_int, frames, picker = 5)
    
    ax2 = plt.subplot2grid((11,11),(0,6), colspan=5, rowspan=5)
    ax2 = plt.gca()
    ax2.plot(0, 0)
    ax2.set_ylabel('Longitude', fontsize=font_size)
    ax2.set_xlabel('Frame', fontsize=font_size)
    
    ax3 = plt.subplot2grid((11,11),(6, 6), colspan=5, rowspan=5)
    ax3 = plt.gca()
    ax3.plot(0, 0)
    ax3.set_ylabel('Total Intensity', fontsize=font_size)
    ax3.set_xlabel('Frame', fontsize=font_size)
    
    ax4 = plt.subplot2grid((11,11),(6, 0), colspan=5, rowspan=5)
    ax4.plot(0)
    ax4.set_ylabel('Total Intensity', fontsize=font_size)
    ax4.set_xlabel('Frame', fontsize=font_size)
    
    fig.canvas.mpl_connect('button_press_event', onclick)

plt.draw()