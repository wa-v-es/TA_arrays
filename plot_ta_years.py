import matplotlib.pyplot as plt
import obspy
from obspy import read, Stream, UTCDateTime,read_events
from obspy.core.event import Origin
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from obspy.clients.fdsn import Client
import numpy as np
import datetime
import os
import sys


#######################################


client = Client("IRIS")
# Client.get_stations(starttime=None, endtime=None, startbefore=None, startafter=None,
# endbefore=None, endafter=None, network=None, station=None, location=None,
# channel=None, minlatitude=None, maxlatitude=None, minlongitude=None,
# maxlongitude=None, latitude=None, longitude=None, minradius=None,
# maxradius=None, level=None, includerestricted=None, includeavailability=None,
#  updatedafter=None, matchtimeseries=None, filename=None, format=None, **kwargs)
time_start_a=UTCDateTime('2011-03-01')
time_start_b=UTCDateTime('2012-01-01')

time_end_b=UTCDateTime('2014-01-01')
time_end_a=UTCDateTime('2013-01-01')


inventory = client.get_stations(network="TA",station='*',maxlatitude=50,startafter=time_start_a,startbefore=time_start_b,
endbefore=time_end_b,endafter=time_end_a, minlongitude=-95,maxlongitude=-84)#,level='stations')startbefore=time_start_b,
# inventory.plot()
start_times=[]
end_times=[]
for network in inventory:
    for station in network:
        start_times.append(station.start_date)
        end_times.append(station.end_date)
        print(f"Station: {station.code}, Start: {station.start_date}, End: {station.end_date}")
        print('#days active:{:.1f}\n'.format((station.end_date-station.start_date)/86400))
###

inventory.plot(projection='local', resolution='l', marker='^',fillstyle='none',
size=4, label=False, color='teal')#,outfile='TA_2012_active.png')
# plt.savefig('TA_2012_active.png', dpi=300,bbox_inches='tight', pad_inches=0.1)
######
