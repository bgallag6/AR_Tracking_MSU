# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 11:41:53 2018

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
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import scipy.signal
#matplotlib.use('TkAgg') 	# NOTE: This is a MAC/OSX thing. Probably REMOVE for linux/Win
from matplotlib.widgets import Cursor
from pylab import axvline
import sunpy
from scipy import signal
from scipy import fftpack
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button

    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
     
class Index(object):
    ind = 0   
       
    def y2010(self, event):
        ax1.set_xlim(frm_ticks[0]-20,frm_ticks[1]+20)
        ax1.set_ylim(-10,370)
        ax1.set_title('2010', y = 1.02, fontsize=font_size)
        yr_start = ax1.axvline(frm_ticks[0], color='r', linewidth=2.)
        yr_end = ax1.axvline(frm_ticks[1], color='r', linewidth=2.)
        plt.draw()

    def y2011(self, event):
        ax1.set_xlim(frm_ticks[1]-20,frm_ticks[2]+20)
        ax1.set_ylim(-10,370)
        ax1.set_title('2011', y = 1.02, fontsize=font_size)
        yr_start = ax1.axvline(frm_ticks[1], color='r', linewidth=2.)
        yr_end = ax1.axvline(frm_ticks[2], color='r', linewidth=2.)
        plt.draw()
        
    def y2012(self, event):
        ax1.set_xlim(frm_ticks[2]-20,frm_ticks[3]+20)
        ax1.set_ylim(-10,370)
        ax1.set_title('2012', y = 1.02, fontsize=font_size)
        yr_start = ax1.axvline(frm_ticks[2], color='r', linewidth=2.)
        yr_end = ax1.axvline(frm_ticks[3], color='r', linewidth=2.)
        plt.draw()

    def y2013(self, event):
        ax1.set_xlim(frm_ticks[3]-20,frm_ticks[4]+20)
        ax1.set_ylim(-10,370)
        ax1.set_title('2013', y = 1.02, fontsize=font_size)
        yr_start = ax1.axvline(frm_ticks[3], color='r', linewidth=2.)
        yr_end = ax1.axvline(frm_ticks[4], color='r', linewidth=2.)
        plt.draw()
        
    def y2014(self, event):
        ax1.set_xlim(frm_ticks[3]-20,frm_ticks[4]+20)
        ax1.set_ylim(-10,370)
        ax1.set_title('2014', y = 1.02, fontsize=font_size)
        yr_start = ax1.axvline(frm_ticks[4], color='r', linewidth=2.)
        yr_end = ax1.axvline(frm_ticks[5], color='r', linewidth=2.)
        plt.draw()

    def yFull(self, event):
        ax1.set_xlim(frm_ticks[0]-20,frm_ticks[5]+20)
        ax1.set_ylim(-10,370)
        ax1.set_title('2010-2014', y = 1.02, fontsize=font_size)
        plt.draw()
        
    def yNorth(self, event):
        global toggle
        toggle = 0
        ax1.clear()
        ax1.scatter(frmN_tot, lonN_tot, color='black')
        ax1.scatter(frmN_NOAA, lonN_NOAA, color='red')
        ax1.set_xlim(frm_ticks[0]-20,frm_ticks[5]+20)
        ax1.set_ylim(-10,370)
        ax1.set_title('2010-2014', y = 1.02, fontsize=font_size)
        plt.draw()
        return toggle

    def ySouth(self, event):
        global toggle
        toggle = 1
        ax1.clear()
        ax1.scatter(frmS_tot, lonS_tot, color='black')
        ax1.scatter(frmS_NOAA, lonS_NOAA, color='red')
        ax1.set_xlim(frm_ticks[0]-20,frm_ticks[5]+20)
        ax1.set_ylim(-10,370)
        ax1.set_title('2010-2014', y = 1.02, fontsize=font_size)
        plt.draw()
        return toggle
    
    def saveFig(self, event):
        plt.savefig('C:/Users/Brendan/Desktop/Figures/scatter_intensity.pdf', bbox_inches='tight')
        
    def scatter(self, event):
        global toggle
        global col
        if toggle == 0:
            toggle = 1
            col = ax1.scatter(x, y, s=50, c='white', picker=True)
        elif toggle == 1:
            toggle = 0
            col.remove()
            plt.draw()                
        return toggle
        
def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    global x10,y10,x20,y20
    x10, y10 = eclick.xdata, eclick.ydata
    x20, y20 = erelease.xdata, erelease.ydata
    x1 = int(x10)
    y1 = int(y10)
    x2 = int(x20)
    y2 = int(y20)
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2)) 
    
    AR_complete = np.zeros((12,1000))  # ARlong, ARlat, ARint, ARfrm, NOAAlong, NOAAlat, NOAAarea, NOAAfrm, HMIlong, HMIlat, HMIint, HMIfrm
    
    frm_boundsL = np.min([x1,x2])
    frm_boundsH = np.max([x1,x2])
    lon_boundsL = np.min([y1,y2])
    lon_boundsH = np.max([y1,y2])
    
    ## North
    if toggle == 0:
        lat_boundsL = 0
        lat_boundsH = 90
        countAR = 0
        countNOAA = 0
        countHMI = 0 
        for t in range(len(lonN_tot)):
            if (frmN_tot[t] > frm_boundsL) and (frmN_tot[t] < frm_boundsH) and (lonN_tot[t] > lon_boundsL) and (lonN_tot[t] < lon_boundsH) and (latN_tot[t] > lat_boundsL) and (latN_tot[t] < lat_boundsH):
                AR_complete[0,countAR] = lonN_tot[t]
                AR_complete[1,countAR] = latN_tot[t]
                AR_complete[2,countAR] = intN_tot[t]
                AR_complete[3,countAR] = frmN_tot[t]
                countAR += 1
        for t in range(len(frmN_NOAA)):
            if (frmN_NOAA[t] > frm_boundsL) and (frmN_NOAA[t] < frm_boundsH) and (lonN_NOAA[t] > lon_boundsL) and (lonN_NOAA[t] < lon_boundsH) and (latN_NOAA[t] > lat_boundsL) and (latN_NOAA[t] < lat_boundsH):
                AR_complete[4,countNOAA] = lonN_NOAA[t]
                AR_complete[5,countNOAA] = latN_NOAA[t]
                AR_complete[6,countNOAA] = intN_NOAA[t]
                AR_complete[7,countNOAA] = frmN_NOAA[t]
                countNOAA += 1
        for t in range(len(frmN_HMI)):
            if (frmN_HMI[t] > frm_boundsL) and (frmN_HMI[t] < frm_boundsH) and (lonN_HMI[t] > lon_boundsL) and (lonN_HMI[t] < lon_boundsH): 
                AR_complete[8,countHMI] = lonN_HMI[t]
                AR_complete[9,countHMI] = latN_HMI[t]
                AR_complete[10,countHMI] = intN_HMI[t]
                AR_complete[11,countHMI] = frmN_HMI[t]
                countHMI += 1
        
    ## South    
    elif toggle == 1:
        lat_boundsL = -90
        lat_boundsH = 0
        countAR = 0
        countNOAA = 0
        countHMI = 0 
        for t in range(len(lonS_tot)):
            if (frmS_tot[t] > frm_boundsL) and (frmS_tot[t] < frm_boundsH) and (lonS_tot[t] > lon_boundsL) and (lonS_tot[t] < lon_boundsH) and (latS_tot[t] > lat_boundsL) and (latS_tot[t] < lat_boundsH):
                AR_complete[0,countAR] = lonS_tot[t]
                AR_complete[1,countAR] = latS_tot[t]
                AR_complete[2,countAR] = intS_tot[t]
                AR_complete[3,countAR] = frmS_tot[t]
                countAR += 1
        for t in range(len(frmS_NOAA)):
            if (frmS_NOAA[t] > frm_boundsL) and (frmS_NOAA[t] < frm_boundsH) and (lonS_NOAA[t] > lon_boundsL) and (lonS_NOAA[t] < lon_boundsH) and (latS_NOAA[t] > lat_boundsL) and (latS_NOAA[t] < lat_boundsH):
                AR_complete[4,countNOAA] = lonS_NOAA[t]
                AR_complete[5,countNOAA] = latS_NOAA[t]
                AR_complete[6,countNOAA] = intS_NOAA[t]
                AR_complete[7,countNOAA] = frmS_NOAA[t]
                countNOAA += 1
        for t in range(len(frmS_HMI)):
            if (frmS_HMI[t] > frm_boundsL) and (frmS_HMI[t] < frm_boundsH) and (lonS_HMI[t] > lon_boundsL) and (lonS_HMI[t] < lon_boundsH): 
                AR_complete[8,countHMI] = lonS_HMI[t]
                AR_complete[9,countHMI] = latS_HMI[t]
                AR_complete[10,countHMI] = intS_HMI[t]
                AR_complete[11,countHMI] = frmS_HMI[t]
                countHMI += 1
    
        
    AR_lat0 =  AR_complete[1,:]
    AR_lat = AR_lat0[AR_lat0 != 0]
    AR_int0 =  AR_complete[2,:]
    AR_int = AR_int0[AR_lat0 != 0]
    #AR_int_norm = np.sum(AR_int)
    #AR_int = AR_int / AR_int_norm
    AR_lon0 =  AR_complete[0,:]
    AR_lon = AR_lon0[AR_lat0 != 0]
    AR_frm0 = AR_complete[3,:]
    AR_frm = AR_frm0[AR_lat0 != 0] 
    
    if AR_complete[6,0] != 0:
        NOAA_lat0 =  AR_complete[5,:]
        NOAA_lat = NOAA_lat0[NOAA_lat0 != 0] 
        NOAA_lon0 =  AR_complete[4,:]
        NOAA_lon = NOAA_lon0[NOAA_lat0 != 0]    
        NOAA_area0 = AR_complete[6,:]
        NOAA_area = NOAA_area0[NOAA_lat0 != 0]  
        NOAA_area /= np.max(NOAA_area)
        NOAA_frm0 = AR_complete[7,:]
        NOAA_frm = NOAA_frm0[NOAA_lat0 != 0] 
        
        intNOAA = np.zeros((int(np.max(NOAA_frm)-np.min(NOAA_frm))+1))
        frmNOAA = [np.min(NOAA_frm)+u for u in range((int(np.max(NOAA_frm)-np.min(NOAA_frm))+1))]
    
        for r in range(len(NOAA_frm)):
            indFrm = int(NOAA_frm[r]-np.min(NOAA_frm))
            #intF[indFrm-int(np.min(AR_frm))] += AR_int[r]
            intNOAA[indFrm] += NOAA_area[r]
    
    AR_frm *= 2
    
    intF = np.zeros((int(np.max(AR_frm)-np.min(AR_frm))+1))
    frmF = [np.min(AR_frm)+u for u in range((int(np.max(AR_frm)-np.min(AR_frm))+1))]

    for r in range(len(AR_frm)):
        indFrm = int(AR_frm[r]-np.min(AR_frm))
        #intF[indFrm-int(np.min(AR_frm))] += AR_int[r]
        intF[indFrm] += AR_int[r]
        
    
    
    """    
    intHMI = np.zeros((int(np.max(HMI_frm)-np.min(HMI_frm))+1))
    frmHMI = np.array([np.min(HMI_frm)+u for u in range((int(np.max(HMI_frm)-np.min(HMI_frm))+1))])

    for r in range(len(HMI_frm)):
        indFrm = int(HMI_frm[r]-np.min(HMI_frm))
        #intF[indFrm-int(np.min(AR_frm))] += AR_int[r]
        intHMI[indFrm] += HMI_int[r]
    """    
    intF /= np.max(intF)   
    
    frmF = np.array(frmF)
    frmF /= 2.   
    
    if AR_complete[6,0] != 0:
        fMax = np.max([np.max(NOAA_frm), np.max(frmF)])
        fMin = np.min([np.min(NOAA_frm), np.min(frmF)])
    else:
        fMax = np.max(frmF)
        fMin = np.min(frmF)
    
    ax2.clear()
    plt.draw()
    ax2.set_title('Intensity Comparison', y = 1.01, fontsize=font_size)
    ax2.plot(frmF, intF, color='blue', linestyle='solid', linewidth=2., label='EUV')
    
    if AR_complete[6,0] != 0:
        ax2.plot(frmNOAA, intNOAA, color='black', linestyle='dashed', linewidth=2., label='NOAA')
    
    #plt.plot(NOAA_frm, NOAA_area, color='purple', label='NOAA')
       
    #plt.plot(frmHMI, intHMI, color='red', linestyle='solid', linewidth=0.7)
    #plt.plot(frmHMI, intHMI, color='red', label='HMI', linestyle='dashed', linewidth=2.) 
    ax2.set_xlim(fMin-20,fMax+20)
    ax2.set_ylim(0,1.1)
    ax2.set_ylabel('Normalized Intensity / Area / Strength', fontsize=font_size)
    ax2.set_xlabel('Days', fontsize=font_size)
    legend = ax2.legend(fontsize=font_size)
    for label in legend.get_lines():
                label.set_linewidth(3.0)  # the legend line width
    return x1,y1,x2,y2


plt.rcParams["font.family"] = "Times New Roman"
global font_size
font_size = 23

global toggle 
toggle = 0

int_thresh = 0

## EUV
abs_dates = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/EUV_Absolute_Dates_%ithresh_revised.npy' % int_thresh)
abs_ARs = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/EUV_Absolute_ARs_%ithresh_revised.npy' % int_thresh)

global latN_tot
global latS_tot
global lonN_tot
global lonS_tot
global intN_tot
global intS_tot
global frmN_tot
global frmS_tot
latN_tot = []
latS_tot = []
lonN_tot = []
lonS_tot = []
intN_tot = []
intS_tot = []
frmN_tot = []
frmS_tot = []

for i in range(abs_ARs.shape[0]):   
    longitude = abs_ARs[i,0,:]
    latitude = abs_ARs[i,1,:]
    intensity = abs_ARs[i,2,:]
    frames = abs_ARs[i,3,:]
    longitude = longitude[intensity != 0]
    latitude = latitude[intensity != 0]
    frames = frames[intensity != 0]
    intensity = intensity[intensity != 0]
    lonN = longitude[latitude > 0]
    latN = latitude[latitude > 0]
    intN = intensity[latitude > 0]
    frmN = frames[latitude > 0]
    lonS = longitude[latitude < 0]
    latS = latitude[latitude < 0]
    intS = intensity[latitude < 0]
    frmS = frames[latitude < 0]
    
    latN_tot = np.append(latN_tot, latN)
    latS_tot = np.append(latS_tot, latS)
    lonN_tot = np.append(lonN_tot, lonN)
    lonS_tot = np.append(lonS_tot, lonS)
    intN_tot = np.append(intN_tot, intN)
    intS_tot = np.append(intS_tot, intS)
    frmN_tot = np.append(frmN_tot, frmN)
    frmS_tot = np.append(frmS_tot, frmS)

global frm_ticks    
frm_indices = ['2010/05/19', '2011/01/01', '2012/01/01', '2013/01/01', '2014/01/01', '2014/05/31']
frm_ticks = [0, 227, 592, 958, 1323, 1473]


## NOAA
abs_dates = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/NOAA_Absolute_Dates.npy')
abs_ARs = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/NOAA_Absolute_ARs.npy')

longitude = abs_ARs[0]
latitude = abs_ARs[1]
intensity = abs_ARs[2]
frames = abs_ARs[3]

global latN_NOAA
global latS_NOAA
global lonN_NOAA
global lonS_NOAA
global intN_NOAA
global intS_NOAA
global frmN_NOAA
global frmS_NOAA    
lonN_NOAA = longitude[latitude > 0]
intN_NOAA = intensity[latitude > 0]
frmN_NOAA = frames[latitude > 0]
latN_NOAA = latitude[latitude > 0]
lonS_NOAA = longitude[latitude < 0]
intS_NOAA = intensity[latitude < 0]
frmS_NOAA = frames[latitude < 0]
latS_NOAA = latitude[latitude < 0]


## HMI
abs_ARs = np.load('C:/Users/Brendan/Desktop/AR_Tracking_MSU/data/HMI_Absolute_ARs.npy')

longitude = abs_ARs[0]
latitude = abs_ARs[1]
intensity = abs_ARs[2]
frames = abs_ARs[3]

global latN_HMI
global latS_HMI
global lonN_HMI
global lonS_HMI
global intN_HMI
global intS_HMI
global frmN_HMI
global frmS_HMI    
lonN_HMI = longitude[latitude > 0]
intN_HMI = intensity[latitude > 0]
frmN_HMI = frames[latitude > 0]
latN_HMI = latitude[latitude > 0]
lonS_HMI = longitude[latitude < 0]
intS_HMI = intensity[latitude < 0]
frmS_HMI = frames[latitude < 0]
latS_HMI = latitude[latitude < 0]




if 1:

    #date_title = '%i/%02i/%02i' % (int(date[0:4]),int(date[4:6]),int(date[6:8]))

    
    # create figure with heatmap and spectra side-by-side subplots
    fig1 = plt.figure(figsize=(20,12))
    ax1 = plt.gca()
    ax1 = plt.subplot2grid((11,1),(1, 0), colspan=1, rowspan=5)
    plt.suptitle('EUV vs NOAA vs HMI Seismic', y=0.97, fontsize=font_size)
    ax1.set_title('2010-2014', y = 1.02, fontsize=font_size)
    
    im, = ([ax1.scatter(frmN_tot, lonN_tot, color='black', picker=True)])
    im2 = ax1.scatter(frmN_NOAA, lonN_NOAA, color='red')
    ax1.set_xlim(frm_ticks[0]-20,frm_ticks[5]+20)
    ax1.set_ylim(-10,370)
    
    # drawtype is 'box' or 'line' or 'none'
    RS = RectangleSelector(ax1, line_select_callback, drawtype='box', useblit=True,
                       button=[1, 3],  # don't use middle button
                       minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    
    #fig1.canvas.mpl_connect('button_press_event', onclick)
    #plt.connect('key_press_event', RS)
    fig1.canvas.mpl_connect('key_press_event', RS)
 
 
    # set up spectra subplot
    ax2 = plt.subplot2grid((11,1),(6, 0), colspan=1, rowspan=5)    
    ax2.set_title('Intensity Comparison', y = 1.01, fontsize=font_size)
    plt.tight_layout()
    
    
    # add callbacks to each button - linking corresponding action
    callback = Index()
    
    # make toggle buttons to display each parameter's heatmap
    ax2010 = plt.axes([0.05, 0.9, 0.07, 0.08])
    ax2011 = plt.axes([0.13, 0.9, 0.07, 0.08])
    ax2012 = plt.axes([0.21, 0.9, 0.07, 0.08])
    ax2013 = plt.axes([0.29, 0.9, 0.07, 0.08])
    ax2014 = plt.axes([0.64, 0.9, 0.07, 0.08])
    axFull = plt.axes([0.72, 0.9, 0.07, 0.08])
    axNorth = plt.axes([0.80, 0.9, 0.07, 0.08])
    axSouth = plt.axes([0.88, 0.9, 0.07, 0.08])
    axSaveFig = plt.axes([0.955, 0.915, 0.04, 0.05])
    
    b2010 = Button(ax2010, '2010')
    b2010.on_clicked(callback.y2010)
    b2010.label.set_fontsize(font_size)
    b2011 = Button(ax2011, '2011')
    b2011.on_clicked(callback.y2011)
    b2011.label.set_fontsize(font_size)
    b2012 = Button(ax2012, '2012')
    b2012.on_clicked(callback.y2012)
    b2012.label.set_fontsize(font_size)
    b2013 = Button(ax2013, '2013')
    b2013.on_clicked(callback.y2013)
    b2013.label.set_fontsize(font_size)
    b2014 = Button(ax2014, '2014')
    b2014.on_clicked(callback.y2014)
    b2014.label.set_fontsize(font_size)
    bFull = Button(axFull, 'Full')
    bFull.on_clicked(callback.yFull)
    bFull.label.set_fontsize(font_size)
    bNorth = Button(axNorth, 'North')
    bNorth.on_clicked(callback.yNorth)
    bNorth.label.set_fontsize(font_size)
    bSouth = Button(axSouth, 'South')
    bSouth.on_clicked(callback.ySouth)
    bSouth.label.set_fontsize(font_size)
    bSaveFig = Button(axSaveFig, 'Save')
    bSaveFig.on_clicked(callback.saveFig)
    bSaveFig.label.set_fontsize(font_size)
    
plt.draw()
  


