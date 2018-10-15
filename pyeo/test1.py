'''

adapted from http://www.net-analysis.com/blog/cartopylayout.html

'''

import pandas as pd
import sys
import os
import subprocess
import datetime
import platform
import datetime
import math
import matplotlib.pyplot as plt
# import seaborn as sb
import cartopy
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.io.img_tiles import StamenTerrain
from cartopy.io.img_tiles import GoogleTiles
from owslib.wmts import WebMapTileService
from matplotlib.path import Path
import matplotlib.patheffects as PathEffects
from matplotlib import patheffects
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np

def blank_axes(ax):
    """
    blank_axes:  blank the extraneous spines and tick marks for an axes

    Input:
    ax:  a matplotlib Axes object

    Output: None
    """

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off', \
                   bottom='off', top='off', left='off', right='off')


# end blank_axes

#######################################################
# MAIN
#######################################################

fig = plt.figure(figsize=(10, 12))

# ------------------------------- Surrounding frame ------------------------------
# set up frame full height, full width of figure, this must be called first

left = -0.01
bottom = -0.01
width = 1.02
height = 1.02
'''
left = 0.01
bottom = 0.01
width = 0.98
height = 0.98
'''
rect = [left, bottom, width, height]
ax3 = plt.axes(rect)

# turn on the spines we want, ie just the surrounding frame
blank_axes(ax3)
ax3.spines['right'].set_visible(True)
ax3.spines['top'].set_visible(True)
ax3.spines['bottom'].set_visible(True)
ax3.spines['left'].set_visible(True)

ax3.text(0.03, 0.03, '© University of Leicester, 2018. ' +
         'Map generated at ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         fontsize=8)

# ---------------------------------  Main Map -------------------------------------
#
# set up main map almost full height (allow room for title), right 80% of figure

'''
left = 0.2
bottom = 0.1
width = 0.7
height = 0.8
'''
left = 0.25
bottom = 0.01
width = 0.65
height = 0.98
rect = [left, bottom, width, height]

ax = plt.axes(rect, projection=ccrs.PlateCarree(), )
ax.set_extent((150, 155, -30, -23))

ax.coastlines(resolution='10m', zorder=2)

LAND_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m',
                                               edgecolor='face',
                                               facecolor=cartopy.feature.COLORS['land'])
RIVERS_10m = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
                                                 edgecolor=cartopy.feature.COLORS['water'],
                                                 facecolor='none')
BORDERS2_10m = cartopy.feature.NaturalEarthFeature('cultural', 'admin_1_states_provinces',
                                                   '10m', edgecolor='black', facecolor='none')
ax.add_feature(LAND_10m)
ax.add_feature(RIVERS_10m)
ax.add_feature(BORDERS2_10m, edgecolor='grey')
ax.stock_img()
# stock image is good enough for example, but OCEAN_10m could be used, but very slow
#       ax.add_feature(OCEAN_10m)

ax.gridlines(draw_labels=True, xlocs=[150, 152, 154, 155])

lon0, lon1, lat0, lat1 = ax.get_extent()

# bar offset is how far from bottom left corner scale bar is (x,y) and how far up is scale bar text
bar_offset = [0.05, 0.05, 0.07]
bar_lon0 = lon0 + (lon1 - lon0) * bar_offset[0]
bar_lat0 = lat0 + (lat1 - lat0) * bar_offset[1]

text_lon0 = bar_lon0
text_lat0 = lat0 + (lat1 - lat0) * bar_offset[2]
length = bar_tickmark = 20000  # metres
bars = bar_ticks = 5
bar_alpha = 0.3
zorder = 9

# Get the limits of the axis in map coordinates
x0, x1, y0, y1 = ax.get_extent(ccrs.PlateCarree())

# specified scalebar location
# TODO need to transform sbx and sby into metres or length into lon

sbx = bar_lon0
sby = bar_lat0

# Get the thickness of the scalebar in map units
thickness = (y1 - y0) / 50

# Generate the x coordinate for the ends of the scalebar
bar_xs = [sbx, sbx + length / bars]

# Generate the y coordinate for the ends of the scalebar
bar_ys = [sby, sby + thickness]

# Plot the scalebar chunks
barcol = 'white'
for i in range(0, bars):
    # plot the chunk
    rect = patches.Rectangle((bar_xs[0], bar_ys[0]), bar_xs[1] - bar_xs[0], bar_ys[1] - bar_ys[0],
                             linewidth=1, edgecolor='black', facecolor=barcol, zorder=zorder)
    ax.add_patch(rect)

    #        ax.plot(bar_xs, bar_ys, transform=tifproj, color=barcol, linewidth=linewidth, zorder=zorder)

    # alternate the colour
    if barcol == 'white':
        barcol = 'black'
    else:
        barcol = 'white'
    # Generate the x,y coordinates for the number
    bar_xt = sbx + i * length / bars
    bar_yt = sby + thickness

    # Plot the scalebar label for that chunk
    ax.text(bar_xt, bar_yt, str(round(i * length / bars)), transform=ccrs.PlateCarree(),
            horizontalalignment='center', verticalalignment='bottom',
            color='black', zorder=zorder)
    # work out the position of the next chunk of the bar
    bar_xs[0] = bar_xs[1]
    bar_xs[1] = bar_xs[1] + length / bars
# Generate the x coordinate for the last number
bar_xt = sbx + length
# Plot the last scalebar label
t = ax.text(bar_xt, bar_yt, str(round(length)) + ' km', transform=ccrs.PlateCarree(),
            horizontalalignment='center', verticalalignment='bottom',
            color='black', zorder=zorder)

'''

for i in range(bar_ticks):
    #  90 degrees = direction of horizontal scale bar
    end_lat, end_lon = displace(bar_lat0, bar_lon0, 90, bar_tickmark)
    # capstyle must be set so line segments end square
    # TODO make transform match ax projection
    ax.plot([bar_lon0, end_lon], [bar_lat0, end_lat], color=bar_color[i % 2], linewidth=20,
            transform=ccrs.PlateCarree(), solid_capstyle='butt', alpha=bar_alpha)
    # start of next bar is end of last bar
    bar_lon0 = end_lon
    bar_lat0 = end_lat
# end for

# highlight text with white background
buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
# Plot the scalebar label
units = 'km'
# TODO make transform match ax projection
t0 = ax.text(text_lon0, text_lat0, str(bar_ticks * bar_tickmark / 1000) + ' ' + units, transform=ccrs.PlateCarree(),
             horizontalalignment='left', verticalalignment='bottom',
             path_effects=buffer, zorder=2)
'''

# ---------------------------------Overview Location Map ------------------------
#
# set up index map 20% height, left 16% of figure
left = 0.03
bottom = 0
width = 0.16
height = 0.2
rect = [left, bottom, width, height]

ax2 = plt.axes(rect, projection=ccrs.PlateCarree(), )
ax2.set_extent((110, 160, -45, -10))
#  ax2.set_global()  will show the whole world as context

ax2.coastlines(resolution='110m', zorder=2)
ax2.add_feature(cfeature.LAND)
ax2.add_feature(cfeature.OCEAN)

ax2.gridlines()

lon0, lon1, lat0, lat1 = ax.get_extent()
box_x = [lon0, lon1, lon1, lon0, lon0]
box_y = [lat0, lat0, lat1, lat1, lat0]

plt.plot(box_x, box_y, color='red', transform=ccrs.Geodetic())

# -------------------------------- Title -----------------------------
# set up map title top 4% of figure, right 80% of figure

left = 0.2
bottom = 0.95
width = 0.8
height = 0.04
rect = [left, bottom, width, height]
ax6 = plt.axes(rect)
ax6.text(0.5, 0.0, 'Multi-Axes Map Example', ha='center', fontsize=20)
blank_axes(ax6)

# ---------------------------------North Arrow  ----------------------------
#
left = 0.03
bottom = 0.2
width = 0.16
height = 0.2
rect = [left, bottom, width, height]
rect = [left, bottom, width, height]
ax4 = plt.axes(rect)

# need a font that support enough Unicode to draw up arrow. need space after Unicode to allow wide char to be drawm?
ax4.text(0.5, 0.0, u'\u25B2 \nN ', ha='center', fontsize=30, family='Arial', rotation=0)
blank_axes(ax4)

# ------------------------------------  Legend -------------------------------------
# legends can be quite long, so set near top of map (0.4 - bottom + 0.5 height = 0.9 - near top)
left = 0.04
bottom = 0.4
width = 0.15
height = 0.5
rect =[left, bottom, width, height]
ax5 = plt.axes(rect)
blank_axes(ax5)

# create an array of color patches and associated names for drawing in a legend
# colors are the predefined colors for cartopy features (only for example, Cartopy names are unusual)
colors = sorted(cartopy.feature.COLORS.keys())

# handles is a list of patch handles
handles =[]
# names is the list of corresponding labels to appear in the legend
names =[]

# for each cartopy defined color, draw a patch, append handle to list, and append color name to names list
for c in colors:
    patch = patches.Patch(color=cfeature.COLORS[c], label=c)
handles.append(patch)
names.append(c)
# end for

# do some example lines with colors
river = mlines.Line2D([], [], color=cfeature.COLORS['water'], marker='',
                      markersize=15, label='river')
coast = mlines.Line2D([], [], color='black', marker='',
                      markersize=15, label='coast')
bdy = mlines.Line2D([], [], color='grey', marker='',
                    markersize=15, label='state boundary')
handles.append(river)
handles.append(coast)
handles.append(bdy)
names.append('river')
names.append('coast')
names.append('state boundary')

# create legend
ax5.legend(handles, names)
ax5.set_title('Legend', loc='left')

plt.show()

wd = '/home/heiko/linuxpy/test/'  # working directory on Linux Virtual Box
plt.savefig(wd+'test1output.jpg')
plt.close()
