# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 13:58:55 2017

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import sunpy
from sunpy.map import Map
  
x = sunpy.map.Map('C:/Users/Brendan/Downloads/20130704_001615_304.fts')
#x = sunpy.map.Map('C:/Users/Brendan/Documents/Python Scripts/test_download2.fts')
x2 = x.data

date = '2013/07/04'


x_ticks = [300*i for i in range(13)]
y_ticks = [150*i for i in range(13)]

x_ind = [30*i for i in range(13)]
y_ind = [90-(15*i) for i in range(13)]


plt.rcParams["font.family"] = "Times New Roman"
font_size = 23

fig = plt.figure(figsize=(22,11))
ax = plt.gca()
#canvas = ax.figure.canvas
ax.set_title(r'304 $\AA$ 12-Hour Carrington Full-Surface Map' + '\n Date: %s' % date, y=1.01, fontweight='bold', fontsize=font_size)
ax.set_ylabel('Latitude [Deg]', fontsize=font_size)
ax.set_xlabel('Longitude [Deg]', fontsize=font_size)
#plt.tick_params(axis='both', labelsize=font_size, pad=7)
ax.imshow(x2, cmap='sdoaia304', vmin=0, vmax=1500)
plt.xticks(x_ticks,x_ind,fontsize=font_size)
plt.yticks(y_ticks,y_ind,fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/example_carrington_map_better.jpeg')
#plt.savefig('C:/Users/Brendan/Desktop/example_carrington_map.pdf', format='pdf')



    
