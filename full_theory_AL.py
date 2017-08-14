# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:12:00 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
import scipy.signal

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

x = np.linspace(0,11,22)
x0 = np.copy(x)
y = np.sin(x0/11*np.pi)

seg = 33

for j in range(3):
    for i in range(seg):
        
        plt.figure(figsize=(20,10))
        ax1 = plt.subplot2grid((11,11),(0, 0), colspan=5, rowspan=5)
        ax2 = plt.subplot2grid((11,11),(0, 6), colspan=5, rowspan=5)
        ax3 = plt.subplot2grid((11,11),(6, 0), colspan=11, rowspan=5)
        
        patches = []
            
        x00 = 0.  # 2011=12, 2012=1.5
        rot_rate = 11./seg
        
        x0 = np.zeros((4))
        y0 = np.zeros((4))
        
        x0[0] = x00+(rot_rate*i)
        x0[1] = x00+(rot_rate*i)
        x0[2] = x00+(rot_rate*(i+1))
        x0[3] = x00+(rot_rate*(i+1))
        y0[0] = 1.1
        y0[1] = 0
        y0[2] = 0
        y0[3] = 1.1
               
        points = zip(x0,y0)
    
        polygon = Polygon(points, True)
        patches.append(polygon)
        
        p = PatchCollection(patches, color='red', alpha=0.35)  
    
        ax1.plot(x,y, color='black', linewidth=2.)
        ax1.add_collection(p)
        ax1.set_title('Solar Cycle Activity', y=1.01, fontsize=font_size)
        ax1.set_xlabel('Year', fontsize=font_size, labelpad=-3)
        ax1.set_ylabel('Activity', fontsize=font_size)
        ax1.set_xlim(-1,12)
        ax1.set_ylim(0,1.1)
        ax1.tick_params(axis='both',labelsize=15)
        
        lat = 25-((25./seg)*i)
        x2 = np.linspace(1,2,5)
        y2 = (lat-15)*x2*0.3 + 15
        y2 = np.flipud(y2)
        
        
        ax2.set_title('Slope - Differential Rotation', y=1.01, fontsize=font_size)
        ax2.vlines(1.45,0,30.)
        ax2.vlines(1.55,0,30.)
        ax2.scatter(1.5,np.median(y2),50.,color='red')
        ax2.hlines(15,0,np.pi,linestyle='dashed', linewidth=2.)
        ax2.plot(x2,y2, linestyle='solid', color='red', linewidth=2.)
        ax2.text(2.5,25,'Slower', fontsize=font_size-3, color='blue')
        ax2.text(2.5,7,'Faster', fontsize=font_size-3, color='red')
        ax2.set_xlim(0,np.pi)
        ax2.set_ylim(0,30.)
        ax2.set_xlabel('Active Longitude', fontsize=font_size, labelpad=9)
        ax2.set_ylabel('Latitude', fontsize=font_size)
        ax2.set_xticks([])
        ax2.set_yticklabels([0,5,10,15,20,25,30])
        ax2.tick_params(axis='both',labelsize=15)
        ax2.text(2.5, 16.5, r'15${^\circ}$ = 0${^\circ}$', fontsize=font_size-3)
        
        phase_ticks = [(12/5.)*k for k in range(6)]
        car_ticks = [(18./17.)*k for k in range(18)]
        phase_ticks2 = [(12/5.)*k for k in range(11)]
        car_ticks2 = [(18./17.)*k for k in range(18)]
        
        phase_ind = [0.2*k for k in range(6)]
        car_ind = [1+(3*k) for k in range(18)]
        phase_ind2 = [0.2*k for k in range(11)]
        car_ind2 = [1+(3*k) for k in range(18)]
        
        aspect = np.float(12*2)/np.float((18))
        
        #aspect_shift = (2./3.)/aspect
        aspect_shift = (2./3.)/aspect
        
        patches2 = []
            
        x00 = 0.  # 2011=12, 2012=1.5
        rot_rate = (11./seg)
        
        x0 = np.zeros((4))
        y0 = np.zeros((4))
        
        x0[0] = x00+(rot_rate*i)+(11*j)
        x0[1] = x00+(rot_rate*i)+(11*j)
        x0[2] = x00+(rot_rate*(i+1))+(11*j)
        x0[3] = x00+(rot_rate*(i+1))+(11*j)
        y0[0] = 12
        y0[1] = 0
        y0[2] = 0
        y0[3] = 12
               
        points = zip(x0,y0)
    
        polygon = Polygon(points, True)
        patches2.append(polygon)
        
        p2 = PatchCollection(patches2, color='red', alpha=0.35)  
        
        x2 = np.linspace(0,33,33)
        x20 = np.copy(x2)
        y2 = np.sin(x20/11*np.pi+(np.pi/2))**2 + 0.02*x2
        ax3.plot(x2,y2, color='black', linewidth=2.)
        
        ax3.set_title('Active Longitude Global Pattern', y=1.01, fontsize=font_size)
        ax3.set_xlabel('Year', fontsize=font_size)
        ax3.set_ylabel('Carrington Phase', fontsize=font_size)
        #ax3.imshow(np.flipud(AL_array), cmap='Greys', interpolation='none') # run 3x_car_rot_from_bands first
        #plt.yticks(phase_ticks, phase_ind)
        #plt.xticks(car_ticks, car_ind)
        #ax3.tick_params(axis='both',labelsize=15)
        #ax3.set_aspect(aspect_shift)
        ax3.add_collection(p2)
        #ax3.set_ylim(12.,0)
        
        #plt.savefig('C:/Users/Brendan/Desktop/full_theory/3_33/AL_Full_Theory_%i_%i.jpeg' % (j,i), bbox_inches='tight')
        plt.close()
