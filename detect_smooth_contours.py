# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:28:04 2017

@author: Brendan
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.convolution import Box2DKernel, Box1DKernel, convolve
from astropy.modeling.models import Gaussian2D
from shapely import geometry
import matplotlib.path as mplPath


#box_2D_kernel = Box2DKernel(3)
box_1D_kernelX = Box1DKernel(8)
box_1D_kernelY = Box1DKernel(3)



frmN_tot = np.load('C:/Users/Brendan/Desktop/framesN_tot.npy')
xN_tot = np.load('C:/Users/Brendan/Desktop/xN_tot.npy')
intN_tot = np.load('C:/Users/Brendan/Desktop/intN_tot.npy')

font_size = 17

#"""  ### plot North / South Hemispheres scatter
fig = plt.figure(figsize=(22,10))
plt.suptitle(r'Nothern Hemisphere - Carrington Rotation Periods', y=0.97, fontweight='bold', fontsize=font_size)
ax1 = plt.subplot2grid((1,11),(0, 0), colspan=5, rowspan=1)
ax1 = plt.gca()    
ax1.set_ylabel('Longitude', fontsize=font_size)
ax1.set_xlabel('Frame', fontsize=font_size)
ax1.set_ylim(0,360)   
ax1.scatter(frmN_tot, xN_tot)  

ax2 = plt.subplot2grid((1,11),(0, 6), colspan=5, rowspan=1)
ax2 = plt.gca()
ax2.set_ylabel('Number of ARs', fontsize=font_size)
ax2.set_xlabel('Longitude', fontsize=font_size)
#ax2.set_ylim(0,bin_max)  
ax2.set_xlim(0,360)
ax2.hist(xN_tot) 
#plt.xticks(x_ticks)
#plt.savefig('C:/Users/Brendan/Desktop/Car_Rot_%i_%i_North.jpg' % ((c*rotations)+1, ((c+1)*rotations)), bbox_inches = 'tight')
#plt.close()


matrx = np.zeros((360,int(frmN_tot[-1])-int(frmN_tot[0])+1))

for i in range(len(frmN_tot)):
    #matrx[int(xN_tot[i]),int(frmN_tot[i])-int(frmN_tot[0])-1] = intN_tot[i]
    matrx[int(xN_tot[i]),int(frmN_tot[i])-int(frmN_tot[0])] = intN_tot[i]

    
matrx = np.flipud(matrx)
    
matrx[matrx < 1] = np.NaN
matrx0 = matrx
matrx = np.nan_to_num(matrx)
#matrx = np.flipud(matrx)
#matrx0 = np.flipud(matrx0)

smoothed_data_box = np.zeros_like(matrx)

for j in range(matrx.shape[0]):
    smoothed_data_box[j] = convolve(matrx[j], box_1D_kernelX)
    matrx[j] = convolve(matrx[j], box_1D_kernelX)

for i in range(matrx.shape[1]):
    smoothed_data_box[:,i] = convolve(matrx[:,i], box_1D_kernelY)
    matrx[:,i] = convolve(matrx[:,i], box_1D_kernelX)
    
#fig = plt.figure(figsize=(10,10))
#plt.imshow(matrx,vmin=0,vmax=1.)


delta = 1
x = np.arange(0,157, delta)
y = np.arange(0,360, delta)
X, Y = np.meshgrid(x, y)

#plt.figure()
#CS = plt.contour(X, Y, matrx)
#plt.clabel(CS, inline=1, fontsize=10)

#plt.figure()
#plt.imshow(smoothed_data_box)

plt.figure()
plt.imshow(matrx0)
CS = plt.contour(X, Y, smoothed_data_box, levels=[0])
plt.clabel(CS, inline=1, fontsize=10)


level0 = CS.levels[0]

c0 = CS.collections[0]

paths = c0.get_paths()

vertices_C = paths[0].vertices
vertices_l = vertices_C.tolist()

frm_arr = np.zeros((matrx.shape[0],matrx.shape[1]))
long_arr = np.zeros((matrx.shape[0],matrx.shape[1]))

for k1 in range(matrx.shape[0]):
    long_arr[k1] = [k1 for s in range(matrx.shape[1])]    
    
for k2 in range(matrx.shape[1]):
    frm_arr[:,k2] = [k2 for s in range(matrx.shape[0])] 
    
long_arr = np.flipud(long_arr)

#poly = geometry.Polygon(vertices_C)
#p_path = mpl.path.Path(poly)
#inside2 = p_path.contains_points(matrx)

ARs = np.zeros((len(paths),4,500))  # x, y, int, frame

#for i in range(len(paths)):
for i in range(1,2):
   #poly = paths[i].vertices
   path = mplPath.Path(paths[i].vertices)
   print path
   within = []
   for c in range(matrx.shape[0]):
       points = zip([c for i in range(matrx.shape[0])],[r for r in range(matrx.shape[1])])
       within_temp = path.contains_points(points)
       within = np.append(within, within_temp)
   within = np.reshape(within, (matrx.shape[0],matrx.shape[1]))
   frm = frm_arr[np.where(within == True)]
   lon = long_arr[np.where(within == True)]
   inten = matrx0[np.where(within == True)]
   #ARs[i,] = 

"""
##**##   
path = mplPath.Path(vertices_l)

for i in range(25):
    for j in range(25):
        inside2 = [[i,j]]
        within = path.contains_points(inside2)
        #print inside2, within

within3 = []        

for c in range(25):
    b = zip([c for i in range(25)],[i for i in range(25)])
    within2 = path.contains_points(b)
    within3 = np.append(within3, within2)
    #print within2
    
within4 = np.reshape(within3, (25,25))
##**## 
"""
"""  
# dont need
contours = np.zeros((len(paths), 5)) # x1,x2,y1,y2,num

for i in range(len(paths)):
    pathC = paths[i]
    x1 = np.min(pathC.vertices[:,1])
    x2 = np.max(pathC.vertices[:,1])
    y1 = np.min(pathC.vertices[:,0])
    y2 = np.max(pathC.vertices[:,0])
    area = (y2-y1)*(x2-x1)
    contours[i] = [x1,x2,y1,y2,area]

#contour_vertices0 = contour_vertices[contours[:,4] > 20]    
contours0 = contours[contours[:,4] > 20]
corners = np.zeros((2,2))
"""

#plt.figure(figsize=(15,15))
#plt.imshow(matrx)
#for k in range(contours0.shape[0]):
#    xy = contours0[k,0:4]
#    corners[0,:] = [xy[0],xy[2]]
#    corners[1,:] = [xy[1],xy[3]]
#    plt.plot(corners)



