# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 15:14:30 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
s = readsav('fits_sample_strs_20161219v7.sav')

coord = s.STRS.coordinates  # rectangular box in pixels
cen_coord = s.STRS.centroid_cord  # centroid in degrees
n_regions = s.STRS.n_region

print coord[0][0]  # pixels?
print cen_coord[0][0]  # degrees?
print n_regions

all_xcoords = []
all_ycoords = []

for i in range(n_regions.size):
    num_reg = n_regions[i]
    for j in range(num_reg):
        all_xcoords = np.append(all_xcoords, cen_coord[i][j][1])
        all_ycoords = np.append(all_ycoords, cen_coord[i][j][0])
  
plt.figure()      
plt.hist(all_ycoords)

plt.figure()
plt.hist(all_xcoords)
    
