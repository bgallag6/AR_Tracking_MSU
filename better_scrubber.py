# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 16:14:50 2017

@author: Brendan
"""

"""
###########################################
### scrubs website for links ##############
### and downloads according to pattern  ###
###########################################
"""

import numpy as np
import urllib2
import urllib

arr_need = []
arr_rename = []

url="http://jsoc.stanford.edu/SUM85/D965615904/S00000/"
page=urllib2.urlopen(url)
data=page.read().split("<td><A HREF=")
tag=".fits"
endtag="</tr>"
for item in data:
    if ".fits" in item:
        
        """        
        try:
            ind = item.index(tag)
            item=item[ind+len(tag):]
            end=item.index(endtag)
        except: pass
        else:
            if item[0:3] == '"http':
                arr_need = np.append(arr_need,item[1:105])
        """
        
        if item[1:5] == 'http':
                filename = item[50:105]
                arr_need = np.append(arr_need,filename)
                k = 2
                fyear = filename[14+k:18+k]
                fmonth = filename[19+k:21+k]
                fday = filename[22+k:24+k]
                fhour = filename[25+k:27+k]
                fmin = filename[27+k:29+k]
                fsec = filename[29+k:31+k]
                wavelength = filename[33+k:37+k]
                new_name = 'aia_lev1_%sa_%s_%s_%st%s_%s_%s_24z_image_lev1.fits' % (wavelength, fyear, fmonth, fday, fhour, fmin, fsec) 
                arr_rename = np.append(arr_rename, new_name)
                
#for fn in arr_need: 
for i in range(len(arr_need)):
    urllib.urlretrieve("http://jsoc.stanford.edu/SUM85/D965615904/S00000/%s" % arr_need[i], "S:/FITS/20140724/%s" % arr_rename[i])