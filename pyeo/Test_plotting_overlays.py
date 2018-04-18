# -*- coding: utf-8 -*-
"""
Created on 17 April 2018

@author: Heiko Balzter
"""

#############################################################################
# test script using a single plot with overlaid axes
#############################################################################

# When you start the IPython Kernel, type in:
#   %matplotlib
# This will launch a graphical user interface (GUI) loop

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Bbox
import matplotlib.path as mpath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
import numpy as np
import cartopy
from cartopy.feature import ShapelyFeature, BORDERS
import cartopy.crs as ccrs
import os, sys
from osgeo import gdal, gdalnumeric, ogr, osr
from skimage import io
gdal.UseExceptions()
io.use_plugin('matplotlib')

##############################################
# MAIN
##############################################

# go to directory
wd = '/home/heiko/linuxpy/test/'  # working directory on Linux Virtual Box
os.chdir(wd)

# define a map projection
tifproj = ccrs.OSGB()
# EPSG 27700 is the British National Grid

# make the figure and the axes objects
fig = plt.figure()

'''
ax0 = fig.add_subplot(1, 1, 1, projection=tifproj)

# do not draw the bounding box around the scale bar area. This seems to be the only way to make this work.
#   there is a bug in Cartopy that always draws the box.
ax0.outline_patch.set_visible(False)
'''

# the corners below cover roughly the British Isles
left1 = 49000
right1 = 688000
bottom1 = 14000
top1 = 1232000

# and these cover a margin below the main map
left2 = left1
right2 = right1
bottom2 = bottom1
top2 = bottom1 + 0.2 * (top1 - bottom1)

# now we overlay a new axes object on to the plot
# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.05, 0.3, 0.9, 0.6]
ax1 = fig.add_axes([left, bottom, width, height], projection=tifproj)
ax1.set_adjustable('box-forced')

# add coastlines etc.
ax1.coastlines(resolution='10m', color='navy', linewidth=1)
ax1.gridlines()
ax1.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax1.add_feature(cartopy.feature.RIVERS)
BORDERS.scale = '10m'
ax1.add_feature(BORDERS, color='red')

# axes ticks
xticks = np.arange(left1, right1, 100000)
yticks = np.arange(bottom1, top1, 100000)
ax1.set_xticks(xticks, crs=tifproj)
ax1.set_yticks(yticks, crs=tifproj)
# stagger x gridline / tick labels
#labels = ax1.set_xticklabels(xticks)
#for i, label in enumerate(labels):
#    label.set_y(label.get_position()[1] - (i % 2) * 0.1)
# rotate the font orientation of the axis tick labels
plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')

# plot a line on the map
ax1.plot([left1 + 30000, left1 + 130000], [(top1 + bottom1) / 2, (top1 + bottom1) / 2],
         color='blue', linewidth=2, marker='.', zorder=90, transform=tifproj)

# mark a known place to help us geo-locate ourselves
ax1.plot((left1+right1)/2, (top1+bottom1)/2, 'bo', markersize=7, transform=tifproj)

# define axes extents in map coordinates, extent1 is the map
extent1 = (left1, right1, bottom1, top1)
# extent2 covers the area below the map for scalebar annotation
extent2 = (left2, right2, bottom2, top2)

# set axes extent
ax1.set_extent(extent1, crs=tifproj)


# add scale bar

'''
# following https://matplotlib.org/2.0.2/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes
# we need to set axes_class=plt.Axes, else it attempts to create
# a GeoAxes as colorbar

divider = make_axes_locatable(ax1)
ax_cb = divider.new_vertical(size="10%", pad=0.1, axes_class=plt.Axes)
'''

# add a second overlaid axes object
left, bottom, width, height = [0.05, 0.0, 0.9, 0.3]
ax = fig.add_axes([left, bottom, width, height], projection=tifproj, sharex=ax1, sharey=None)
#ax.set_adjustable('box-forced')

# define pars
bars=4
length=400 # in km
location=(0.2, 0.5)
linewidth=5
col='black'
zorder=20


# def draw_scale_bar(ax, tifproj, bars=4, length=None, location=(0.1, 0.8), linewidth=5, col='black', zorder=20):

"""
Plot a nice scale bar with 4 subdivisions on an axis linked to the map scale.

ax is the axes to draw the scalebar on.
tifproj is the map projection
bars is the number of subdivisions of the bar (black and white chunks)
length is the length of the scalebar in km.
location is left side of the scalebar in axis coordinates.
(ie. 0 is the left side of the plot)
linewidth is the thickness of the scalebar.
color is the color of the scale bar and the text

modified from
https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot/35705477#35705477

"""
# Get the limits of the axis in map coordinates
#x0, x1, y0, y1 = ax.get_extent(tifproj)
x0, x1, y0, y1 = extent2

# specified relative scalebar location in coordinates in metres
sbx = x0 + (x1 - x0) * location[0]
sby = y0 + (y1 - y0) * location[1]

# Get the thickness of the scalebar in map units
thickness = (y1 - y0) / 20

# Calculate a scale bar length if none has been given
if not length:
    length = (x1 - x0) / 1000 / bars  # in km
    ndim = int(np.floor(np.log10(length)))  # number of digits in number
    length = round(length, -ndim)  # round to 1sf

    # Returns numbers starting with the list
    def scale_number(x):
        if str(x)[0] in ['1', '2', '5']:
            return int(x)
        else:
            return scale_number(x - 10 ** ndim)

    length = scale_number(length)

# Generate the x coordinate for the ends of the scalebar
bar_xs = [sbx, sbx + length * 1000 / bars]

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
        barcol = col
    else:
        barcol = 'white'
    # Generate the x,y coordinates for the number
    bar_xt = sbx + i * length * 1000 / bars
    bar_yt = sby + thickness

    # Plot the scalebar label for that chunk
    ax.text(bar_xt, bar_yt, str(round(i * length / bars)), transform=tifproj,
            horizontalalignment='center', verticalalignment='bottom',
            color=col, zorder=zorder)
    # work out the position of the next chunk of the bar
    bar_xs[0] = bar_xs[1]
    bar_xs[1] = bar_xs[1] + length * 1000 / bars
# Generate the x coordinate for the last number
bar_xt = sbx + length * 1000
# Plot the last scalebar label
t = ax.text(bar_xt, bar_yt, str(round(length)) + ' km', transform=tifproj,
            horizontalalignment='center', verticalalignment='bottom',
            color=col, zorder=zorder)

# same length segment on axes 2 for comparison of scale
ax.plot([left2 + 30000, left2 + 130000], [(top2 + bottom2) / 2, (top2 + bottom2) / 2],
         color='blue', linewidth=2, marker='.', zorder=90, transform=tifproj)


'''
# and below ax1
ax1.set_ylim(bottom=bottom2-10000)
ax1.plot([left2 + 30000, left2 + 130000], [bottom2, bottom2],
         color='blue', linewidth=2, marker='.', zorder=90, transform=tifproj)
'''

# set axes extent
ax.set_extent(extent2, crs=tifproj)

# do not draw the bounding box around the scale bar area. This seems to be the only way to make this work.
#   there is a bug in Cartopy that always draws the box.
ax.outline_patch.set_visible(False)
# remove the facecolor of the geoAxes
ax.background_patch.set_visible(False)

'''
The adjustment of the position of the axes objects below can be used to 'shrink' the length and height of the scale bar.
A hack is needed to make the length of the scale bar match on ax the map scale on ax1.

The code below returns map coordinates, so it is no use. We need figure coordinates.

'''

'''
# plot a line of the same length as the scale bar onto the map on ax1 but put it behind the map to make it invisible
l = ax1.plot([left1, left1 + length * 1000], [(top1 + bottom1) / 2, (top1 + bottom1) / 2],
         color='blue', linewidth=1, marker='', zorder=1, visible=False, transform=tifproj)
path1 = l[0]._path # extract the path of the line from the plot object
l = ax.plot([left2, left2 + length * 1000], [(top2 + bottom2) / 2, (top2 + bottom2) / 2],
         color='blue', linewidth=1, marker='', zorder=1, visible=False, transform=tifproj)
path2 = l[0]._path # extract the path of the line from the plot object
'''
# TODO Try:
#ax1.outline_patch.get_window_extent()
#ax.outline_patch.get_window_extent()

#ax1.patch.get_window_extent()
#ax.patch.get_window_extent()

# get axes positions in figure canvas coordinates
#bbax1 = ax1.patch.get_window_extent()
#bbax = ax.patch.get_window_extent()
#l1 = bbax1.x1 - bbax1.x0
#l2 = bbax.x1 - bbax.x0

# You can use the ax.transData instance to transform from your data to your display coordinate system,
#   either a single point or a sequence of points as shown below:
#print( ax1.transData.transform((left1, bottom1)),
#    ax1.transData.transform((right1, bottom1)),
#    ax.transData.transform((left2, bottom2)),
#    ax.transData.transform((right2, bottom2)))

fig.canvas.show()
renderer = plt.gca().get_renderer_cache()
bbax1 = ax1.get_lines()[0].get_window_extent(renderer)
bbax =  ax.get_lines()[0].get_window_extent(renderer)
l1 = bbax1.x1 - bbax1.x0
l2 = bbax.x1 - bbax.x0

# define the bounding box of axes 2
'''
I need to get these parameters right !!!!
'''
x0 = 0.5 - l1 / l2 * 0.5
x1 = 0.5 + l1 / l2 * 0.5
y0 = 0.05
y1 = 0.25
bbax = Bbox(np.array([[x0, y0], [x1, y1]]))
# set axes 2 position to match the x position and width
ax.set_position(bbax)

#fig.tight_layout()
fig.show()
fig.savefig('Map_with_subplots.jpg')

'''
#########################################################
# Hack to get text size in renderer coordinates
#########################################################
import matplotlib.pyplot as plt

xx=[1,2,3]
yy=[2,3,4]
dy=[0.1,0.2,0.05]

fig=plt.figure()
figname = "out.png"
ax=fig.add_subplot(111)

ax.errorbar(xx,yy,dy,fmt='ro-',ms=6,elinewidth=4)

# start of hack to get renderer
fig.savefig(figname)
renderer = plt.gca().get_renderer_cache()
# end of hack

txt = ax.text(xx[1], yy[1],r'$S=0$',fontsize=16)
tbox = txt.get_window_extent(renderer)
dbox = tbox.transformed(ax.transData.inverted())
text_width = dbox.x1-dbox.x0
text_height = dbox.y1-dbox.y0
x = xx[1] - text_height
y = yy[1] - text_width/2
txt.set_position((x,y))

ax.set_xlim([0.,3.4])
ax.set_ylim([0.,4.4])

fig.savefig(figname)
'''




'''
##################
#
# this does not work either:
fig = plt.figure()
ax1 = plt.subplot2grid((4,3), (0,0), colspan=3, rowspan=2)
ax2 = plt.subplot2grid((4,3), (2,0), colspan=3, sharex=ax1)
ax3 = plt.subplot2grid((4,3), (3,0), colspan=3, sharex=ax1)
'''
