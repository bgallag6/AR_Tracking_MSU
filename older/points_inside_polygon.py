# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:12:49 2017

@author: Brendan
"""

from time import time
import numpy as np
import matplotlib.path as mplPath

#"""
vertices_l = [[0.,0.], [0.,10.], [10.,10.], [10.,0.]]

arr = np.zeros((25,25))

for i in range(25):
    for j in range(25):
        arr[i,j] = i+j
# random points set of points to test 
points = [(5.,5.),(3.,3.)]


# Matplotlib mplPath
path = mplPath.Path(vertices_l)

for i in range(25):
    for j in range(25):
        inside2 = [[i,j]]
        within = path.contains_points(inside2)
        #print inside2, within

within3 = []        
#"""
for c in range(25):
    b = zip([c for i in range(25)],[i for i in range(25)])
    within2 = path.contains_points(b)
    within3 = np.append(within3, within2)
    #print within2
    
within4 = np.reshape(within3, (25,25))
"""
poly = [190, 50, 500, 310]
bbPath = mplPath.Path(np.array([[poly[0], poly[1]],
                     [poly[1], poly[2]],
                     [poly[2], poly[3]],
                     [poly[3], poly[0]]]))

r = bbPath.contains_point((200, 100))
"""