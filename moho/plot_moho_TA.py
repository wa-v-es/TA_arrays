###
import numpy as np
# from numpy import load
import obspy
import miller_alaskamoho_srl2018 as alaskamoho
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from obspy.clients.fdsn import Client
from obspy import read, Stream, UTCDateTime,read_events
import rasterio
import os.path as path
import stripy as stripy
import sys
from matplotlib.colors import ListedColormap

#######


client = Client("IRIS")
time_start_a=UTCDateTime('2011-03-01')
time_start_b=UTCDateTime('2012-01-01')

time_end_b=UTCDateTime('2014-01-01')
time_end_a=UTCDateTime('2013-01-01')

inventory = client.get_stations(network="TA",station='*',maxlatitude=50,startafter=time_start_a,
                    startbefore=time_start_b,endbefore=time_end_b,endafter=time_end_a,
                     minlongitude=-95,maxlongitude=-84)#,level='stations')startbefore=time_start_b,

# inventory.plot()
start_times=[]
end_times=[]
for network in inventory:
    for station in network:
        start_times.append(station.start_date)
        end_times.append(station.end_date)
        # print(f"Station: {station.code}, Start: {station.start_date}, End: {station.end_date}")
        # print('#days active:{:.1f}\n'.format((station.end_date-station.start_date)/86400))
###
# inventory.plot(projection='local', resolution='l', marker='^',fillstyle='none',
# size=4, label=False, color='teal')#,outfile='TA_2012_active.png')
# plt.show()

lats = []
lons = []
for network in inventory:
    for station in network.stations:
        lats.append(station.latitude)
        lons.append(station.longitude)

# sys.exit()

cmap = plt.cm.RdYlBu
# Transparent colours
###
colA = cmap(np.arange(cmap.N))
colA[:,-1] = 0.25 + 0.5 * np.linspace(-1.0, 1.0, cmap.N)**2.0
#adjusts the opacity based on the quadratic curve, setting values between 0.25 and 0.75.
##
# Create new colormap
cmapA = ListedColormap(colA)
# cmapA = cmap
proj = ccrs.Stereographic(central_longitude=-90, central_latitude=90, true_scale_latitude=37)
# plt.clf()
plt.ion()
fig = plt.figure(figsize=(15, 8), facecolor=None)
ax1 = plt.subplot(1, 1, 1, projection=proj)

plt.rcParams.update({'font.size': 14})

# ax1.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)

ax1.set_extent([-95,-80,29,48.5], crs=ccrs.PlateCarree())

# grat = cartopy.feature.NaturalEarthFeature(category="physical", scale="10m", name="graticules_5")
# ax1.add_feature(grat, linewidth=0.5,linestyle="--",edgecolor="#000000",facecolor="None", zorder=2)

ax1.coastlines(resolution="10m",color="#111111", linewidth=0.5, zorder=99)
ax1.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')

ax1.add_feature(cartopy.feature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='gray')
ax1.add_feature(cfeature.OCEAN.with_scale('10m'),alpha=0.1,facecolor='xkcd:azure')
# ax1.add_feature(cfeature.LAKES.with_scale('10m'),alpha=0.3,facecolor='xkcd:dusty blue')

for i,lat in enumerate(lats):
    if lat > 34:
        ax1.plot(lons[i], lats[i], marker='^',markersize=8, linestyle='None',linewidth=.6, markerfacecolor='none', markeredgecolor='maroon', transform=ccrs.PlateCarree())

### plot Moho tiff
moho_file='GeophysicsMoho_USCanada/USCanada_Moho.tif'

with rasterio.open(moho_file) as src:
    image = src.read(1)
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
image_clean = np.ma.masked_where(image < -1e+30, image)
img=ax1.imshow(image_clean, origin='upper', extent=extent,
transform=ccrs.PlateCarree(), cmap=cmapA, vmin=25,vmax=50)

gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=.8, color='gray', alpha=0.5, linestyle='--',rotate_labels=False,
        x_inline=False, y_inline=False)


gl.xlocator = mticker.FixedLocator([-95,-90,-85,-80])
gl.ylocator = mticker.FixedLocator([30,35,40,45])

# gl.xlines = True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

# sys.exit()


cbar=plt.colorbar(img,orientation='horizontal',location='top',ax=ax1,shrink=0.2, extend='max', drawedges=False, pad=0.02 )
cbar.set_label("Moho depth (km)",fontsize=15)
# fig.savefig('Moho_TA.png', dpi=400,bbox_inches='tight', pad_inches=0.1)
plt.show()
sys.exit()
###
# bit to save tif as txt file (.2 degree)
lon_min, lon_max = -99, -70
lat_min, lat_max = 29, 55

step = 0.2
with rasterio.open(moho_file) as src:
    band = src.read(1)
    transform = src.transform

# Mask invalid values
band = np.ma.masked_where(band < -1e30, band)

# Coordinate grid
lons = np.arange(lon_min, lon_max + step, step)
lats = np.arange(lat_min, lat_max + step, step)

out_data = []

for lat in lats:
    for lon in lons:
        # convert world coordinates → raster row/col
        col, row = ~transform * (lon, lat)
        row, col = int(row), int(col)

        # check inside raster
        if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
            value = band[row, col]
            if value is np.ma.masked:
                continue
            out_data.append([lat, lon, float(value)])

# Save as text file
np.savetxt("moho_region_TA.txt", out_data, fmt="%.1f  %.1f  %.1f")
##
#
