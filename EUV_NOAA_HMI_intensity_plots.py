# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:58:16 2017

@author: Brendan
"""
import matplotlib.pyplot as plt
import numpy as np

#for h in range(11):
#for h in range(len(frm_boundsN_H)):
for h in range(1):
    
    AR_lat0 =  AR_complete[h,1,:]
    AR_lat = AR_lat0[AR_lat0 != 0]
    AR_int0 =  AR_complete[h,2,:]
    AR_int = AR_int0[AR_lat0 != 0]
    #AR_int_norm = np.sum(AR_int)
    #AR_int = AR_int / AR_int_norm
    AR_lon0 =  AR_complete[h,0,:]
    AR_lon = AR_lon0[AR_lat0 != 0]
    AR_frm0 = AR_complete[h,3,:]
    AR_frm = AR_frm0[AR_lat0 != 0] 

    NOAA_lat0 =  AR_complete[h,5,:]
    NOAA_lat = NOAA_lat0[NOAA_lat0 != 0] 
    NOAA_lon0 =  AR_complete[h,4,:]
    NOAA_lon = NOAA_lon0[NOAA_lat0 != 0]    
    NOAA_area0 = AR_complete[h,6,:]
    NOAA_area = NOAA_area0[NOAA_lat0 != 0]
    #NOAA_area_norm = np.sum(NOAA_area)
    #NOAA_area = NOAA_area / NOAA_area_norm  
    NOAA_area /= np.max(NOAA_area)
    NOAA_frm0 = AR_complete[h,7,:]
    NOAA_frm = NOAA_frm0[NOAA_lat0 != 0] 
    
    HMI_lat0 =  AR_complete[h,9,:]
    HMI_lat = HMI_lat0[HMI_lat0 != 0]  
    HMI_frm0 = AR_complete[h,11,:]
    HMI_frm = HMI_frm0[HMI_lat0 != 0]    
    HMI_lon0 =  AR_complete[h,8,:]
    HMI_lon = HMI_lon0[HMI_lat0 != 0]   
    HMI_int0 =  AR_complete[h,10,:]
    HMI_int = HMI_int0[HMI_lat0 != 0]
    #HMI_int_norm = np.sum(HMI_int)
    #HMI_int = HMI_int / HMI_int_norm
    HMI_int /= np.max(HMI_int)
    
    #AR_frm *= 2
    #NOAA_frm *= 2
    #HMI_frm *= 2.
    
    intF = np.zeros((int(np.max(AR_frm)-np.min(AR_frm))+1))
    frmF = [np.min(AR_frm)+u for u in range((int(np.max(AR_frm)-np.min(AR_frm))+1))]

    for r in range(len(AR_frm)):
        indFrm = int(AR_frm[r]-np.min(AR_frm))
        #intF[indFrm-int(np.min(AR_frm))] += AR_int[r]
        intF[indFrm] += AR_int[r]
        
    intNOAA = np.zeros((int(np.max(NOAA_frm)-np.min(NOAA_frm))+1))
    frmNOAA = [np.min(NOAA_frm)+u for u in range((int(np.max(NOAA_frm)-np.min(NOAA_frm))+1))]

    for r in range(len(NOAA_frm)):
        indFrm = int(NOAA_frm[r]-np.min(NOAA_frm))
        #intF[indFrm-int(np.min(AR_frm))] += AR_int[r]
        intNOAA[indFrm] += NOAA_area[r]
    
    """    
    intHMI = np.zeros((int(np.max(HMI_frm)-np.min(HMI_frm))+1))
    frmHMI = np.array([np.min(HMI_frm)+u for u in range((int(np.max(HMI_frm)-np.min(HMI_frm))+1))])

    for r in range(len(HMI_frm)):
        indFrm = int(HMI_frm[r]-np.min(HMI_frm))
        #intF[indFrm-int(np.min(AR_frm))] += AR_int[r]
        intHMI[indFrm] += HMI_int[r]
    """    
    intF /= np.max(intF)   
    #intNOAA /= np.max(intNOAA)  
    #intHMI /= np.max(intHMI)
    HMI_int /= np.max(HMI_int)
        
    #HMI_frm /= 2. 
    #frmHMI /= 2.     
   
    rMax = np.max([np.max(NOAA_area), np.max(AR_int), np.max(HMI_int)])
    fMax = np.max([np.max(NOAA_frm), np.max(AR_frm), np.max(HMI_frm)])
    fMin = np.min([np.min(NOAA_frm), np.min(AR_frm), np.min(HMI_frm)])
    
    """
    fig = plt.figure(figsize=(22,10))
    #plt.suptitle('Active Region Evolution: EUV vs NOAA vs HMI Seismic' + '\n %i: Northern Hemisphere' % (2011), y=0.97, fontsize=font_size)
    plt.suptitle('Active Region Evolution: EUV vs NOAA vs HMI Seismic' + '\n %i: Southern Hemisphere' % (2011), y=0.97, fontsize=font_size)
    ax1 = plt.gca()
    ax1 = plt.subplot2grid((2,11),(0, 0), colspan=5, rowspan=1)
    plt.scatter(AR_frm, AR_lon, color='blue', label='EUV')
    plt.scatter(NOAA_frm, NOAA_lon, color='purple', label='NOAA')
    plt.scatter(HMI_frm, HMI_lon, color='red', label='HMI')
    #plt.scatter(AR_frm, AR_lat, color='blue', label='EUV')
    #plt.scatter(NOAA_frm, NOAA_lat, color='purple', label='NOAA')
    #plt.scatter(HMI_frm, HMI_lat, color='red', label='HMI')
    #plt.xlim(0,180)
    plt.xlim(0,360)
    plt.ylim(0,360)
    #plt.ylim(0,45)
    legend = plt.legend(fontsize=font_size)
    
    ax3 = plt.gca()
    ax3 = plt.subplot2grid((2,11),(1, 0), colspan=5, rowspan=1)
    #plt.scatter(AR_frm, AR_lon, color='blue', label='EUV')
    #plt.scatter(NOAA_frm, NOAA_lon, color='purple', label='NOAA')
    #plt.scatter(HMI_frm, HMI_lon, color='red', label='HMI')
    plt.scatter(AR_frm, AR_lat, color='blue', label='EUV')
    plt.scatter(NOAA_frm, NOAA_lat, color='purple', label='NOAA')
    plt.scatter(HMI_frm, HMI_lat, color='red', label='HMI')
    #plt.xlim(0,180)
    plt.xlim(0,360)
    #plt.ylim(0,360)
    #plt.ylim(0,45)
    plt.ylim(-45,0)
    legend = plt.legend(fontsize=font_size)    
    
    
    ax2 = plt.gca()
    ax2 = plt.subplot2grid((2,11),(0, 6), colspan=5, rowspan=2)
    #plt.plot(AR_frm, AR_int, color='blue', label='EUV')
    plt.plot(frmF, intF, color='blue', label='EUV')
    #plt.plot(NOAA_frm, NOAA_area, color='purple', label='NOAA')
    plt.plot(frmNOAA, intNOAA, color='purple', label='NOAA')
    plt.plot(HMI_frm, HMI_int, color='red', label='HMI')
    plt.xlim(fMin*0.9,fMax*1.1)
    #plt.ylim(0,rMax*1.1)
    plt.ylim(0,1.1)
    plt.ylabel('Normalized Intensity / Area / Strength', fontsize=font_size)
    plt.xlabel('Days', fontsize=font_size)
    legend = plt.legend(fontsize=font_size)
    for label in legend.get_lines():
                label.set_linewidth(3.0)  # the legend line width
    #plt.savefig('C:/Users/Brendan/Desktop/AR_NOAA_HMI_%i_South_%i_Rev.jpeg' % (2012, h), bbox_inches='tight')
       
       
    fig = plt.figure(figsize=(22,10))
    #plt.suptitle('Active Region Evolution: EUV vs NOAA vs HMI Seismic' + '\n %i: Northern Hemisphere' % (2011), y=0.97, fontsize=font_size)
    plt.suptitle('Active Region Evolution: EUV vs NOAA vs HMI Seismic' + '\n %i: Southern Hemisphere' % (2012), y=0.97, fontsize=font_size)         
    ax2 = plt.gca()
    ax2 = plt.subplot2grid((1,1),(0, 0), colspan=1, rowspan=1)
    #plt.plot(AR_frm, AR_int, color='blue', label='EUV')
    plt.plot(frmF, intF, color='blue', label='EUV', linestyle='solid', linewidth=2.)
    #plt.plot(NOAA_frm, NOAA_area, color='purple', label='NOAA')
    plt.plot(frmNOAA, intNOAA, color='black', linestyle='dashed', linewidth=2., label='NOAA')
    plt.plot(HMI_frm, HMI_int, color='red', linestyle='solid', linewidth=0.7)
    plt.plot(HMI_frm, HMI_int, color='red', label='HMI', linestyle='dashed', linewidth=2.)    
    plt.xlim(fMin*0.9,fMax*1.1)
    #plt.ylim(0,rMax*1.1)
    plt.ylim(0,1.1)
    plt.ylabel('Normalized Intensity / Area / Strength', fontsize=font_size)
    plt.xlabel('Days', fontsize=font_size)
    legend = plt.legend(fontsize=font_size)
    for label in legend.get_lines():
                label.set_linewidth(3.0)  # the legend line width
    #plt.savefig('C:/Users/Brendan/Desktop/AR_NOAA_HMI_%i_South_%i_Rev2.jpeg' % (2012, h), bbox_inches='tight')
    #plt.savefig('C:/Users/Brendan/Desktop/AR_NOAA_HMI_%i_North_%i_Rev2.jpeg' % (2011, h), bbox_inches='tight')
    """
                
    #"""   
    fig = plt.figure(figsize=(22,25))
    plt.suptitle('Active Region Evolution: EUV vs NOAA vs HMI Seismic' + '\n %i: Northern Hemisphere' % (2011), y=0.97, fontsize=font_size)
    #plt.suptitle('Active Region Evolution: EUV vs NOAA vs HMI Seismic' + '\n %i: Southern Hemisphere' % (2012), y=0.97, fontsize=font_size)         
    ax1 = plt.gca()
    ax1 = plt.subplot2grid((2,1),(0, 0), colspan=1, rowspan=1)
    #plt.plot(AR_frm, AR_int, color='blue', label='EUV')
    plt.plot(frmF, intF, color='blue', label='EUV', linestyle='solid', linewidth=2.)
    #plt.plot(NOAA_frm, NOAA_area, color='purple', label='NOAA')
    plt.plot(frmNOAA, intNOAA, color='black', linestyle='dashed', linewidth=2., label='NOAA')
    plt.plot(HMI_frm, HMI_int, color='red', linestyle='solid', linewidth=0.7)
    plt.plot(HMI_frm, HMI_int, color='red', label='HMI', linestyle='dashed', linewidth=2.)    
    #plt.plot(frmHMI, intHMI, color='red', linestyle='solid', linewidth=0.7)
    #plt.plot(frmHMI, intHMI, color='red', label='HMI', linestyle='dashed', linewidth=2.) 
    plt.xlim(fMin*0.9,fMax*1.05)
    #plt.ylim(0,rMax*1.1)
    plt.ylim(0,1.1)
    plt.ylabel('Normalized Intensity / Area / Strength', fontsize=font_size)
    plt.xlabel('Days', fontsize=font_size)
    legend = plt.legend(fontsize=font_size)
    for label in legend.get_lines():
                label.set_linewidth(3.0)  # the legend line width
                
    ax2 = plt.subplot2grid((2,1),(1, 0), colspan=1, rowspan=1)
    plt.scatter(AR_frm, AR_lat, color='blue', label='EUV')
    plt.scatter(NOAA_frm, NOAA_lat, color='black', label='NOAA')
    plt.scatter(HMI_frm, HMI_lat, color='red', label='HMI')
    #plt.xlim(0,180)
    #plt.xlim(np.min(AR_frm)-(np.max(AR_frm)*0.1),np.max(AR_frm)+(np.max(AR_frm)*0.1))
    #plt.ylim(np.min(AR_lon)-(np.max(AR_lon)*0.1),np.max(AR_lon)+(np.max(AR_lon)*0.1))
    plt.xlim(np.min(AR_frm)*0.9,np.max(AR_frm)*1.05)
    plt.ylim(np.min(AR_lon)-(np.max(AR_lon)*0.15),np.max(AR_lon)+(np.max(AR_lon)*0.15))
    plt.ylim(0,90)
    plt.xlabel('Days', fontsize=font_size)
    plt.ylabel('Longitude', fontsize=font_size)
    #plt.ylim(0,360)
    #plt.ylim(0,45)
    #plt.ylim(-45,0)
    legend = plt.legend(fontsize=font_size)
    
    #plt.savefig('C:/Users/Brendan/Desktop/AR_NOAA_HMI_%i_South_%i_Rev2.jpeg' % (2012, h), bbox_inches='tight')
    #plt.savefig('C:/Users/Brendan/Desktop/AR_NOAA_HMI_%i_North_%i_RevB.jpeg' % (2011, h), bbox_inches='tight')
    #"""