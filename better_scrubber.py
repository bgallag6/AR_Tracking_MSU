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

url="https://solarmuse.jpl.nasa.gov/data/euvisdo_maps_carrington_12hr/304fits/"
page=urllib2.urlopen(url)
data=page.read().split("<td>")
tag=".fts'>"
endtag="</a>"
for item in data:
    if ".fts'>" in item:
        try:
            ind = item.index(tag)
            item=item[ind+len(tag):]
            end=item.index(endtag)
        except: pass
        else:
            if item[0:3] == '201':
                arr_need = np.append(arr_need,item[:end])
                
#for fn in arr_need: 
for i in range(10):
    urllib.urlretrieve("https://solarmuse.jpl.nasa.gov/data/euvisdo_maps_carrington_12hr/304fits/%s" % arr_need[i], "/disk/data/bgallag/fits/%s" % arr_need[i])