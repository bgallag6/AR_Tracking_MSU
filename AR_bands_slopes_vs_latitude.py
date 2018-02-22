# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:58:32 2017

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

#fit_paramsS = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_slopes.npy')
#ARsS = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_lat.npy')
#fit_params = np.load('C:/Users/Brendan/Desktop/AR_slopes_S_1x_Rot.npy')
#ARs = np.load('C:/Users/Brendan/Desktop/AR_bands_S_1x_Rot.npy')
#fit_paramsS = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_slopes.npy')
#ARsS = np.load('C:/Users/Brendan/Desktop/MSU_Project/AR_bands_S_lat.npy')
fit_paramsS = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/South/AR_Absolute_slopes_S_3x_6x3ysmooth.npy')
ARsS = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/South/AR_Absolute_bands_S_3x_6x3ysmooth.npy')
#fit_paramsS = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/North/AR_Absolute_slopes_N_3x_6x3ysmooth.npy')
#ARsS = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/North/AR_Absolute_bands_N_3x_6x3ysmooth.npy')
#fit_paramsS = fit_params
#ARsS = AR_total

count = 0
#for i in range(500):
#for i in range(1000):
for i in range(3000):
    if ARsS[i,2,0] == 0.:
    #if ARsS[i,1,0] == 0.:  #NOAA
        count = i
        break
    
slopes = fit_paramsS[:,0]
intercept = fit_paramsS[:,1]
#slopes = slopes[slopes != 0]
slopes = slopes[intercept != 0]

med_lat = np.zeros((count))
avg_lat = np.zeros((count))
    
for c in range(count):
    lat_temp = ARsS[c,3,:]
    lat_temp = lat_temp[lat_temp != 0]
    med_lat[c] = np.median(lat_temp)
    avg_lat[c] = np.average(lat_temp)

m, b = np.polyfit(med_lat, slopes, 1)

#med_lat_sin2 = (np.sin(med_lat*np.pi/180))**2
med_lat_sin2 = np.sin(np.deg2rad(med_lat))**2
#med_lat_sin2 = np.rad2deg(np.sin(np.deg2rad(med_lat)))

slopes_days = slopes*2  # older
#slopes_days = slopes  # for absolute

m2, b2 = np.polyfit(med_lat_sin2, slopes_days, 1)

lat_full = np.array([c for c in range(35)])

r_val = pearsonr(slopes, med_lat)[0]

print r_val

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
plt.figure(figsize=(15,10))
plt.title('Band Slopes: Southern Hemisphere (Per 3x Carrington Rotation)', y=1.01, fontsize=font_size)
#plt.scatter(med_lat,slopes)
plt.scatter(med_lat_sin2, slopes_days)
#plt.plot(med_lat, m*med_lat + b, 'r-')  
plt.plot(med_lat_sin2, m2*med_lat_sin2 + b, 'r-')   
#plt.xlabel('Latitude [Deg]',fontsize=font_size)
plt.xlabel(r'$Sin^2$(Latitude)',fontsize=font_size)
plt.ylabel('Slope',fontsize=font_size)
plt.xticks(fontsize=font_size-3)
plt.yticks(fontsize=font_size-3)
#plt.xlim(0,40)
#plt.xlim(-35,0)
#plt.ylim(-1.,1.)
plt.xlim(0,0.3)
#plt.ylim(-1.5,1.5)
plt.ylim(-2.5,2.5)
#plt.text(-10,0.67,'r-value = %0.2f' % np.abs(r_val), fontsize=font_size)
plt.text(0.20,0.75,'y = %0.2fx + %0.2f' % (m2, b2), fontsize=font_size)
#plt.text(25,0.67,'r-value = %0.2f' % np.abs(r_val), fontsize=font_size)
#plt.text(30,0.47,'equation = %0.2fx + %0.2f' % (m, b), fontsize=font_size)
#plt.text(27,0.65,'y = %0.2fx + %0.2f' % (m2, b2), fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/Band_Slopes_South_equation_NOAA.jpeg', bbox_inches = 'tight')
#plt.savefig('C:/Users/Brendan/Desktop/Band_Slopes_North_equation_NOAA_1977.jpeg', bbox_inches = 'tight')


"""
fit_paramsN = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/North/AR_Absolute_slopes_N_3x_6x3ysmooth.npy')
ARsN = np.load('C:/Users/Brendan/Desktop/Inbox/AL/absolute_24thresh_revised/North/AR_Absolute_bands_N_3x_6x3ysmooth.npy')

count = 0
for i in range(3000):
    if ARsN[i,0,0] == 0.:
    if ARsN[i,2,0] == 0.:
    #if ARsS[i,1,0] == 0.:  #NOAA
        count = i
        break
    
slopes = fit_paramsN[:,0]
intercept = fit_paramsN[:,1]
#slopes = slopes[slopes != 0]
slopes = slopes[intercept != 0]

    
#slopes = fit_paramsN[:,0]
#slopes = slopes[slopes != 0]


med_lat = np.zeros((count))
avg_lat = np.zeros((count))
    
for c in range(count):
    lat_temp = ARsN[c,3,:]
    lat_temp = lat_temp[lat_temp != 0]
    med_lat[c] = np.median(lat_temp)
    avg_lat[c] = np.average(lat_temp)

m, b = np.polyfit(med_lat, slopes, 1)

#med_lat_sin2 = (np.sin(med_lat*np.pi/180))**2
med_lat_sin2 = np.sin(np.deg2rad(med_lat))**2
#med_lat_sin2 = np.rad2deg(np.sin(np.deg2rad(med_lat)))

slopes_days = slopes*2
#slopes_days = slopes

m2, b2 = np.polyfit(med_lat_sin2, slopes_days, 1)

resids = np.zeros((len(med_lat_sin2)))
for i in range(len(med_lat_sin2)):
    resids[i] = np.abs(slopes_days[i] - (m2*med_lat_sin2[i] + b))

#std_dev = np.std()

r_val = pearsonr(slopes, med_lat)[0]

print r_val

plt.rcParams["font.family"] = "Times New Roman"
font_size = 23
    
plt.figure(figsize=(15,10))
plt.title('Band Slopes: Northern Hemisphere (Per 3x Carrington Rotation)', y=1.01, fontsize=font_size)
#plt.scatter(med_lat,slopes)
plt.scatter(med_lat_sin2, slopes_days)
#plt.plot(med_lat, m*med_lat + b, 'r-')  
plt.plot(med_lat_sin2, m2*med_lat_sin2 + b, 'r-')   
#plt.xlabel('Latitude [Deg]',fontsize=font_size)
plt.xlabel(r'$Sin^2$(Latitude)',fontsize=font_size)
plt.ylabel('Slope',fontsize=font_size)
plt.xticks(fontsize=font_size-3)
plt.yticks(fontsize=font_size-3)
#plt.xlim(0,40)
plt.xlim(0,0.3)
plt.ylim(-1.5,1.5)
#plt.text(-10,0.67,'r-value = %0.2f' % np.abs(r_val), fontsize=font_size)
plt.text(0.20,0.75,'y = %0.2fx + %0.2f' % (m2, b2), fontsize=font_size)
#plt.text(25,0.67,'r-value = %0.2f' % np.abs(r_val), fontsize=font_size)
#plt.text(30,0.47,'equation = %0.2fx + %0.2f' % (m, b), fontsize=font_size)
#plt.text(27,0.65,'y = %0.2fx + %0.2f' % (m2, b2), fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/Band_Slopes_South_equation.jpeg', bbox_inches = 'tight')
#plt.savefig('C:/Users/Brendan/Desktop/Band_Slopes_North_equation.jpeg', bbox_inches = 'tight')
"""