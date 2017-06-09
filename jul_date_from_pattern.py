# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 13:34:41 2017

@author: Brendan
"""

import numpy as np
import urllib2
import urllib
import matplotlib.pyplot as plt
from astropy.time import Time
import jdcal
import datetime
#"""
arr_need = []

url="https://solarmuse.jpl.nasa.gov/data/euvisdo_maps_carrington_12hr/304fits/"
page=urllib2.urlopen(url)
data=page.read().split("<td>")
tag=".fts'>"
endtag="</a>"

fmt = '%Y%m%d'

for item in data:
    if ".fts'>" in item:
        try:
            ind = item.index(tag)
            item=item[ind+len(tag):]
            end=item.index(endtag)
        except: pass
        else:
            if item[0:3] == '201':
                dt = datetime.datetime.strptime(item[0:8],fmt)
                #dt = date1.timetuple()
                #print date1.year
                jul_date = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
                arr_need = np.append(arr_need,int(jul_date))

#np.save('C:/Users/Brendan/Desktop/image_jul_dates.npy', arr_need)