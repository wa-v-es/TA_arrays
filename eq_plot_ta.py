import matplotlib.pyplot as plt
import numpy as np
import obspy
from obspy import read, Stream, UTCDateTime,read_events
from obspy.core.event import Origin
from obspy.clients.fdsn import Client
import os
import cartopy.crs as ccrs
import sys

import circle as cir_robin
from importlib import reload
reload(cir_robin)
import requests
from obspy.taup.taup_geo import calc_dist,calc_dist_azi


boundaries = requests.get("https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json").json()

####
station='ILAR'
sta_lat= 39
sta_long=-89

client = Client("IRIS")

# eq_map = Basemap(projection='robin', resolution = 'i', area_thresh = 1000.0,
#               lat_0=sta_lat, lon_0=sta_long)
# eq_map.drawcoastlines()
# eq_map.drawcountries()
# eq_map.fillcontinents(color = 'LightGoldenrodYellow')
# eq_map.drawmapboundary()
# eq_map.drawmeridians(np.arange(0, 360, 60))
# eq_map.drawparallels(np.arange(-90, 90, 30))
# plt.show()

###########
# catfile = 'events_ak_2024_6.7.xml'
# catfile = 'events_ZS_2019_6.7.xml'
catfile_NW = 'events_TA_2012_6.3.xml'

### get eq data
#earthquakes
starttime= UTCDateTime('2012-01-01T00:00:01')
endtime= UTCDateTime('2013-01-01T00:00:01')

# starttime= UTCDateTime('2017-10-01T00:00:01') # for ZS
# endtime= UTCDateTime('2019-12-31T00:00:01') # for ZS

if not os.path.exists(catfile_NW):
    catalog = client.get_events(starttime=starttime, endtime=endtime,minmagnitude=6.3,\
    latitude=sta_lat,longitude=sta_long, minradius=65, maxradius=120,mindepth=10)
    catalog.write(catfile_NW, 'QUAKEML')

catalog_NW=read_events(catfile_NW)
# catalog_NW.plot()
lats, longs = [], []
mags = []
azi=[]
dists=[]
#
for event in catalog_NW:
    lats.append(event.origins[0].latitude)
    longs.append(event.origins[0].longitude)
    mags.append(event.magnitudes[0].mag)
    dist=calc_dist(event.origins[0].latitude,event.origins[0].longitude,sta_lat,sta_long,6400,0)
    dists.append(dist)

print('{} event in catalog'.format(len(catalog_NW)))
#
###
fig, ax=plt.subplots(figsize=(9,6))
plt.axis('off')

# ax = plt.axes(projection=ccrs.Mollweide(central_longitude=sta_long))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=sta_long))
# ax = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=sta_long,central_latitude=sta_lat))
ax.set_global()
ax.stock_img()
ax.coastlines(color='black', linewidth=.55)
ax.plot(sta_long, sta_lat, color='indigo', marker='^', markersize=7, transform=ccrs.Geodetic())
# ax.plot(sta_long_, sta_lat_, color='indigo', marker='^', markersize=7, transform=ccrs.Geodetic())

min_marker_size = 1
for i in range(len(lats)): #plot eqs
    # x,y = eq_map(lon, lat)
    msize = mags[i] * min_marker_size
    # marker_string = get_marker_color(mag)
    ax.plot(longs[i], lats[i],color='darkred',marker='o',markersize=msize,alpha=.4,transform=ccrs.Geodetic())


X,Y=cir_robin.equi(sta_long, sta_lat, 7150)
X1,Y1=cir_robin.equi(sta_long, sta_lat, 13200)

plt.plot(X,Y,transform=ccrs.Geodetic(),lw=.7,alpha=.6,linestyle='--',c='maroon')
plt.plot(X1,Y1,transform=ccrs.Geodetic(),lw=.7,alpha=.6,linestyle='--',c='maroon')

# X,Y=cir_robin.equi(sta_long_, sta_lat_, 7215)
# X1,Y1=cir_robin.equi(sta_long_, sta_lat_, 11100)
#
# plt.plot(X,Y,transform=ccrs.Geodetic(),lw=.7,alpha=.6,linestyle='--',c='navy')
# plt.plot(X1,Y1,transform=ccrs.Geodetic(),lw=.7,alpha=.6,linestyle='--',c='navy')

#plot gcp
for event in catalog_NW:
    ori= event.preferred_origin()
    plt.plot([sta_long, ori.longitude],[sta_lat, ori.latitude],  transform=ccrs.Geodetic(),color='black',lw=1.35,linestyle='dotted',alpha=.95)

# for event in catalog_SE:
#     ori= event.preferred_origin()
#     plt.plot([sta_long_, ori.longitude],[sta_lat_, ori.latitude],  transform=ccrs.Geodetic(),color='black',lw=1,linestyle='dotted',alpha=.65)

# Plot boundaries.
for f in boundaries["features"]:
    c = np.array(f["geometry"]["coordinates"])
    lng, lat = c[:, 0], c[:, 1]
    x, y = lng, lat
    mask = np.bitwise_or(np.abs(x) > 1e15, np.abs(y) > 1e15)
    x = np.ma.array(x)
    y = np.ma.array(y)
    x.mask = mask
    y.mask = mask
    plt.plot(x, y, color="Navy", lw=.35,transform=ccrs.Geodetic())

# ax.text(sta_long+5, sta_lat-3, 'AK',fontsize=8,fontfamily='serif', color='indigo',transform=ccrs.Geodetic())
# ax.text(sta_long_+5, sta_lat_-3, 'AK_SE',fontsize=8,fontfamily='serif', color='indigo',transform=ccrs.Geodetic())

# ax.text(-145, -8, '65°',fontsize=10,fontfamily='serif', color='maroon',transform=ccrs.Geodetic())
# ax.text(-145, -42, '100°',fontsize=10,fontfamily='serif', color='maroon',transform=ccrs.Geodetic())
# ax.axis('off')
# ax.set_axis_off()
# ax.set_frame_on(False)

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

# plt.show()

# sys.exit()
plt.savefig('eq_TA_2012_6.3_robin.jpg',dpi=400,bbox_inches='tight', pad_inches=0,transparent=True)
###
