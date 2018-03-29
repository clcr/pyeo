# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:11:00 2018

@author: Heiko Balzter
"""

#############################################################################
# read all Sentinel-2 band geotiffs in a directory and a shape file
#   and make RGB quicklook maps at different scales
# written for Python 3.6.4
#############################################################################

# TODO for John IMPORTANT:
# When you start the IPython Kernel, type in:
#   %matplotlib
# This will launch a graphical user interface (GUI) loop

########################
# TODO plot the scale bar below the map outside of its boundaries
# TODO add a north arrow
# TODO separate geotiff conversion and 10 m resampling into 2 functions
# TODO plot multiple scenes onto the same map by providing a list of scene IDs to map_it instead
# TODO   of rgbdata and running readsen2rgb from within map_it
# TODO directory management: save outputs to a different subdirectory outside raw scene directory structure
########################

import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature, BORDERS
import cartopy
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.io.img_tiles import StamenTerrain
from cartopy.io.img_tiles import GoogleTiles
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patheffects as PathEffects
from matplotlib import patheffects
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os, sys
from os import listdir
from os.path import isfile, isdir, join
from osgeo import gdal, gdalnumeric, ogr, osr
from skimage import io
import subprocess
gdal.UseExceptions()
io.use_plugin('matplotlib')
import pandas as pd
import subprocess
import datetime
import platform
import datetime
import math
import matplotlib.pyplot as plt
from owslib.wmts import WebMapTileService


# The pyplot interface provides 4 commands that are useful for interactive control.
# plt.isinteractive() returns the interactive setting True|False
# plt.ion() turns interactive mode on
# plt.ioff() turns interactive mode off
# plt.draw() forces a figure redraw

#############################################################################
# OPTIONS
#############################################################################
# wd = '/scratch/clcr/shared/py/' # working directory on Linux HPC
wd = '/home/heiko/linuxpy/mexico/'  # working directory on Linux Virtual Box
datadir = wd + 'data/'  # directory of Sentinel L1C data files in .SAFE format
shapefile = 'Sitios_Poly.shp' # the shapefile resides in wd
bands = [5, 4, 3]  # band selection for RGB


#############################################################################
# FUNCTION DECLARATIONS
#############################################################################

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

# define functions to read/write floating point numbers from/to a text file
def read_floats(filename):
    with open(filename) as f:
        return [float(x) for x in f]
    f.close()

def write_floats(data, filename):
    file = open(filename, 'w')
    for item in data:
        file.write("%f\n" % item)
    file.close()

def get_gridlines(x0, x1, y0, y1, nticks):
    '''
    make neat gridline labels for map projections
        x0, x1 = minimum and maximum x positions in map projection coordinates
        y0, y1 = minimum and maximum y positions in map projection coordinates
        nticks = number of ticks / gridlines in x direction
        returns two numpy arrays with x and y tick positions
    '''
    # make sure gridline positions have min 2 digits
    ndigits = len(str(abs(x0)).split('.')[0])  # number of digits before the decimal point
    xx0 = x0
    xfactor = 1  # how many time do we need to multiply by 10
    while ndigits < 2:
        xx0 = xx0 * 10
        xfactor = xfactor * 10
        ndigits = len(str(abs(xx0)).split('.')[0])  # number of digits before the decimal point
        if xfactor > 100000:
            print('\nError in XFactor while loop!')
            break
    x0 = round(x0 * xfactor, 0) / xfactor
    x1 = round(x1 * xfactor, 0) / xfactor
    y0 = round(y0 * xfactor, 0) / xfactor
    y1 = round(y1 * xfactor, 0) / xfactor
    # make sure gridline positions have max 3 digits
    ndigits = len(str(abs(x0)).split('.')[0])  # number of digits before the decimal point
    xx0 = x0
    xfactor = 1  # how many time do we need to divide by 10
    while ndigits > 3:
        xx0 = xx0 / 10
        xfactor = xfactor * 10
        ndigits = len(str(abs(xx0)).split('.')[0])  # number of digits before the decimal point
        if xfactor > 100000:
            print('\nError in XFactor while loop!')
            break
    x0 = round(x0 / xfactor, 0) * xfactor
    x1 = round(x1 / xfactor, 0) * xfactor
    y0 = round(y0 / xfactor, 0) * xfactor
    y1 = round(y1 / xfactor, 0) * xfactor
    # carry on
    dx = (x1 - x0) / nticks
    dy = (y1 - y0) / nticks
    xticks = np.arange(x0, x1 + dx, dx)
    yticks = np.arange(y0, y1 + dy, dy)
    return xticks, yticks


# plot a scale bar with 4 subdivisions on the left side of the map
def scale_bar_left(ax, bars=4, length=None, location=(0.1, 0.05), linewidth=3, col='black'):
    """
    ax is the axes to draw the scalebar on.
    bars is the number of subdivisions of the bar (black and white chunks)
    length is the length of the scalebar in km.
    location is left side of the scalebar in axis coordinates.
    (ie. 0 is the left side of the plot)
    linewidth is the thickness of the scalebar.
    color is the color of the scale bar and the text
    """
    # Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    # Make tmc aligned to the left of the map,
    # vertically at scale bar location
    sbllx = llx0 + (llx1 - llx0) * location[0]
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000  # in km
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
    # Plot the scalebar chunks
    barcol = 'white'
    for i in range(0, bars):
        # plot the chunk
        ax.plot(bar_xs, [sby, sby], transform=tmc, color=barcol, linewidth=linewidth)
        # alternate the colour
        if barcol == 'white':
            barcol = col
        else:
            barcol = 'white'
        # Generate the x coordinate for the number
        bar_xt = sbx + i * length * 1000 / bars
        # Plot the scalebar label for that chunk
        ax.text(bar_xt, sby, str(round(i * length / bars)), transform=tmc,
                horizontalalignment='center', verticalalignment='bottom',
                color=col)
        # work out the position of the next chunk of the bar
        bar_xs[0] = bar_xs[1]
        bar_xs[1] = bar_xs[1] + length * 1000 / bars
    # Generate the x coordinate for the last number
    bar_xt = sbx + length * 1000
    # Plot the last scalebar label
    ax.text(bar_xt, sby, str(round(length)), transform=tmc,
            horizontalalignment='center', verticalalignment='bottom',
            color=col)
    # Plot the unit label below the bar
    bar_xt = sbx + length * 1000 / 2
    bar_yt = y0 + (y1 - y0) * (location[1] / 4)
    ax.text(bar_xt, bar_yt, 'km', transform=tmc, horizontalalignment='center',
            verticalalignment='bottom', color=col)


# function to convert coordinates
def convertXY(xy_source, inproj, outproj):
    shape = xy_source[0, :, :].shape
    size = xy_source[0, :, :].size
    # the ct object takes and returns pairs of x,y, not 2d grids
    # so the the grid needs to be reshaped (flattened) and back.
    ct = osr.CoordinateTransformation(inproj, outproj)
    xy_target = np.array(ct.TransformPoints(xy_source.reshape(2, size).T))
    xx = xy_target[:, 0].reshape(shape)
    yy = xy_target[:, 1].reshape(shape)
    return xx, yy


# This function will convert the rasterized clipper shapefile to a mask for use within GDAL.
def imageToArray(i):
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a = gdalnumeric.fromstring(i.tostring(), 'b')
    a.shape = i.im.size[1], i.im.size[0]
    return a


def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)


def transformxy(s_srs, t_srs, xcoord, ycoord):
    """
    Transforms a point coordinate x,y from a source reference system (s_srs)
    to a target reference system (t_srs)
    """
    geom = ogr.Geometry(ogr.wkbPoint)
    geom.SetPoint_2D(0, xcoord, ycoord)
    geom.AssignSpatialReference(s_srs)
    geom.TransformTo(t_srs)
    return geom.GetPoint_2D()


def projectshape(inshp, outshp, t_srs):
    """
    Reprojects an ESRI shapefile from its source reference system
    to a target reference system (e.g. t_srs = 4326)
    filenames must include the full directory paths
    requires:
        from osgeo import ogr, osr
        import os
    """

    driver = ogr.GetDriverByName('ESRI Shapefile')  # get shapefile driver
    infile = driver.Open(inshp, 0)
    if infile is None:
        print('Could not open ' + inshp)
        sys.exit(1)  # exit with an error code
    inLayer = infile.GetLayer()  # get input layer
    inSpatialRef = inLayer.GetSpatialRef()  # get source spatial reference system
    # or input SpatialReference manually here
    #   inSpatialRef = osr.SpatialReference()
    #   inSpatialRef.ImportFromEPSG(2927)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(t_srs)
    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # create the output layer
    if os.path.exists(outshp):
        driver.DeleteDataSource(outshp)
    outDataSet = driver.CreateDataSource(outshp)
    outLayer = outDataSet.CreateLayer("basemap_" + str(t_srs), geom_type=ogr.wkbMultiPolygon)
    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)
    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()
    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = inLayer.GetNextFeature()
    # Save and close the shapefiles
    inDataSet = None
    outDataSet = None
    # Try to open the output file to check it worked
    outfile = driver.Open(outshp, 0)
    if outfile is None:
        print('Failed to create ' + outshp)
        sys.exit(1)  # exit with an error code
    else:
        print('Reprojection of shapefile seems to have worked.')
    return None


def OpenArray(array, prototype_ds=None, xoff=0, yoff=0):
    #  this is basically an overloaded version of the gdal_array.OpenArray passing in xoff, yoff explicitly
    #  so we can pass these params off to CopyDatasetInfo
    ds = gdal.Open(gdalnumeric.GetArrayFilename(array))

    if ds is not None and prototype_ds is not None:
        if type(prototype_ds).__name__ == 'str':
            prototype_ds = gdal.Open(prototype_ds)
        if prototype_ds is not None:
            gdalnumeric.CopyDatasetInfo(prototype_ds, ds, xoff=xoff, yoff=yoff)
    return ds


def histogram(a, bins=range(0, 256)):
    """
    Histogram function for multi-dimensional array.
    a = array
    bins = range of numbers to match
    """
    fa = a.flat
    n = gdalnumeric.searchsorted(gdalnumeric.sort(fa), bins)
    n = gdalnumeric.concatenate([n, [len(fa)]])
    hist = n[1:] - n[:-1]
    return hist


def stretch(im, nbins=256, nozero=True):
    """
    Performs a histogram stretch on an ndarray image.
    """
    # modified from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # ignore zeroes
    if nozero:
        im2 = im[np.not_equal(im, 0)]
    else:
        im2 = im
    # get image histogram
    image_histogram, bins = np.histogram(im2.flatten(), nbins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(im.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(im.shape), cdf


def read_sen2_rgb(rgbfiles, enhance=True):
    '''
    reads in 3 separate geotiff files as R G and B channels
    rgbfiles: list of three filenames including directory structure
    enhance = True: applies histogram stretching (optional)
    returns a data frame scaled to unsigned 8 bit integer values
    '''
    # make array of 8-bit unsigned integers to be memory efficient
    # open the first file with GDAL to get dimensions
    ds = gdal.Open(rgbfiles[0])
    data = ds.ReadAsArray()
    rgbdata = np.zeros([len(bands), data.shape[0], data.shape[1]], \
                       dtype=np.uint8)

    for i, thisfile in enumerate(rgbfiles):
        print('Reading data from ' + thisfile)

        # open the file with GDAL
        ds = gdal.Open(thisfile)
        data = ds.ReadAsArray()

        # only process single-band files, these have not got 3 bands
        if data.shape[0] > 3:
            # histogram stretching and keeping the values in
            #   the RGB data array as 8 bit unsigned integers
            rgbdata[i, :, :] = np.uint8(stretch(data)[0])

        ds = None
    return rgbdata


def map_it(rgbdata, tifproj, mapextent, shapefile, plotfile='map.jpg',
           plottitle='', figsizex=10, figsizey=10):
    '''
    standard map making function that saves a jpeg file of the output
    and visualises it on screen
    rgbdata = numpy array of the red, green and blue channels, made by read_sen2rgb
    tifproj = map projection of the tiff files from which the rgbdata originate
    mapextent = extent of the map in map coordinates
    shapefile = shapefile name to be plotted on top of the map
    shpproj = map projection of the shapefile
    plotfile = output filename for the map plot
    plottitle = text to be written above the map
    figsizex = width of the figure in inches
    figsizey = height of the figure in inches
    '''
    # get shapefile projection from the file
    # get driver to read a shapefile and open it
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shapefile, 0)
    if dataSource is None:
        print('Could not open ' + shapefile)
        sys.exit(1)  # exit with an error code
    # get the layer from the shapefile
    layer = dataSource.GetLayer()
    # get the projection information and convert to wkt
    projsr = layer.GetSpatialRef()
    projwkt = projsr.ExportToWkt()
    projosr = osr.SpatialReference()
    projosr.ImportFromWkt(projwkt)
    # convert wkt projection to Cartopy projection
    projcs = projosr.GetAuthorityCode('PROJCS')
    shapeproj = ccrs.epsg(projcs)

    # make the figure and the axes
    subplot_kw = dict(projection=tifproj)
    fig, ax = plt.subplots(figsize=(figsizex, figsizey),
                           subplot_kw=subplot_kw)

    # set a margin around the data
    ax.set_xmargin(0.05)
    ax.set_ymargin(0.10)

    # add a background image for rendering
    ax.stock_img()

    # show the data from the geotiff RGB image
    img = ax.imshow(rgbdata[:3, :, :].transpose((1, 2, 0)),
                    extent=extent, origin='upper')

    # read shapefile and plot it onto the tiff image map
    shape_feature = ShapelyFeature(Reader(shapefile).geometries(),
                                   crs=shapeproj, edgecolor='yellow',
                                   facecolor='none')
    ax.add_feature(shape_feature)

    # add a title
    plt.title(plottitle)

    # set map extent
    ax.set_extent(mapextent, tifproj)

    # add coastlines
    ax.coastlines(resolution='10m', color='navy', linewidth=1)

    # add lakes and rivers
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)

    # add borders
    BORDERS.scale = '10m'
    ax.add_feature(BORDERS, color='red')

    # format the gridline positions nicely
    xticks, yticks = get_gridlines(mapextent[0], mapextent[1],
                                   mapextent[2], mapextent[3],
                                   nticks=10)

    # add gridlines
    gl = ax.gridlines(crs=tifproj, xlocs=xticks, ylocs=yticks,
                      linestyle='--', color='grey', alpha=1, linewidth=1)

    # add ticks
    ax.set_xticks(xticks, crs=tifproj)
    ax.set_yticks(yticks, crs=tifproj)

    # stagger x gridline / tick labels
    labels = ax.set_xticklabels(xticks)
    for i, label in enumerate(labels):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)

    # add scale bar
    scale_bar_left(ax, bars=4, length=40, col='dimgrey')

    # show the map
    plt.show()

    # save it to a file
    fig.savefig(plotfile)




#############################################################################
# MAIN
#############################################################################

# go to working directory
os.chdir(wd)

###################################################
# make a 'plots' directory (if it does not exist yet) for map output files
###################################################
plotdir = wd + 'plots_' + shapefile.split(".")[0] + "/"
if not os.path.exists(plotdir):
    print("Creating directory: ", plotdir)
    os.mkdir(plotdir)

###################################################
# get names of all scenes
###################################################

# get list of all data subdirectories (one for each image)
allscenes = [f for f in listdir(datadir) if isdir(join(datadir, f))]
print('\nList of Sentinel-2 scenes:')
for scene in allscenes:
    print(scene)
print('\n')

###################################################
# resample all Sentinel-2 scenes in the data directory to 10 m
###################################################
for x in range(len(allscenes)):
    if allscenes[x].split(".")[1] == "SAFE":
        # open the file
        print('\n******************************')
        print("Reading scene", x + 1, ":", allscenes[x])
        print('******************************\n')

        # set working directory to the Sentinel scene subdirectory
        scenedir = datadir + allscenes[x] + "/"
        os.chdir(scenedir)

        ###################################################
        # get footprint of the scene from the metadatafile
        ###################################################
        # get the list of filenames ending in .xml, but exclude 'INSPIRE.xml'
        xmlfiles = [f for f in os.listdir(scenedir) if f.endswith('.xml') & (1 - f.startswith('INSPIRE'))]
        #print('Reading footprint from ' + xmlfiles[0])
        # use the first .xml file in the directory
        with open(xmlfiles[0]) as f:
            content = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        # find the footprint in the metadata
        footprint = [x for x in content if x.startswith('<EXT_POS_LIST>')]
        # the first element of the returned list is a string
        #   so extract the string and split it
        footprint = footprint[0].split(" ")
        #   and split off the metadata text
        footprint[0] = footprint[0].split(">")[1]
        #   and remove the metadata text at the end of the list
        footprint = footprint[:-1]
        # convert the string list to floats
        footprint = [float(s) for s in footprint]
        # list slicing to separate lon and lat coordinates: list[start:stop:step]
        footprinty = footprint[0::2]  # latitudes
        footprintx = footprint[1::2]  # longitudes
        #print(footprint)

        # set working directory to the Granule subdirectory
        os.chdir(datadir + allscenes[x] + "/" + "GRANULE" + "/")
        sdir = listdir()[0]  # only one subdirectory expected in this directory

        # set working directory to the image data subdirectory
        imgdir = datadir + allscenes[x] + "/" + "GRANULE" + "/" + sdir + "/" + "IMG_DATA" + "/"
        os.chdir(imgdir)

        ###################################################
        # get the list of filenames for all bands in .jp2 format
        ###################################################
        sbands = sorted([f for f in os.listdir(imgdir) if f.endswith('.jp2')])
        print('Bands:')
        for band in sbands:
            print(band)
        nbands = len(sbands)  # get the number of bands in the image
        print('\n')

        ###################################################
        # load all bands to get row and column numbers, and resample to 10 m
        ###################################################
        ncolmax = nrowmax = 0
        obands = sbands  # filenames of output tiff files, all at 10 m resolution

        # in the scene directory, make a 'tiff' subdirectory for 10 m Geotiffs
        tiffdir = scenedir + 'tiff/'
        if not os.path.exists(tiffdir):
            print("Creating directory: ", tiffdir)
            os.mkdir(tiffdir)

        ###################################################
        # process all the bands to 10 m resolution
        ###################################################

        # enumerate produces a counter and the contents of the band list
        for i, iband in enumerate(sbands):

            # open a band
            bandx = gdal.Open(iband, gdal.GA_Update)

            # get image dimensions
            ncols = bandx.RasterXSize
            nrows = bandx.RasterYSize

            # get raster georeferencing information
            geotrans = bandx.GetGeoTransform()
            ulx = geotrans[0]  # Upper Left corner coordinate in x
            uly = geotrans[3]  # Upper Left corner coordinate in y
            pixelWidth = geotrans[1]  # pixel spacing in map units in x
            pixelHeight = geotrans[5]  # (negative) pixel spacing in y
            print("Band %s has %6d columns, %6d rows and a %d m resolution." \
                  % (iband, ncols, nrows, pixelWidth))
            # scale factor for resampling to 10 m pixel resolution
            sf = abs(int(pixelWidth / 10))
            # determining the maximum number of columns and rows at 10 m
            ncolmax = max(ncols * sf, ncolmax)
            nrowmax = max(nrows * sf, nrowmax)

            # resample the 20 m and 40 m images to 10 m and convert to Geotiff
            if pixelWidth != 999:  # can be removed, is redundant as all images will be converted to GeoTiff
                print('  Resampling %s image from %d m to 10 m resolution and converting to Geotiff' \
                      % (iband, pixelWidth))
                # define the zoom factor in %
                zf = str(pixelWidth * 10) + '%'
                # define an output file name
                obands[i] = iband[:-4] + '_10m.tif'
                # assemble command line code
                res_cmd = ['gdal_translate', '-outsize', zf, zf, '-of', 'GTiff',
                           iband, tiffdir + obands[i]]
            # save geotiff file at 10 m resolution
            subprocess.call(res_cmd)

            #close GDAL file
            bandx = None

        print("Output number of columns = %6d\nOutput number of rows = %6d." \
              % (ncolmax, nrowmax))

        print("\n")
        print("Resampling to 10 m resolution and conversion to Geotiff completed.")
        print("\n")

        ###################################################
        # Make RGB maps from three Geotiff files
        ###################################################

        print('\n******************************')
        print('Making maps from Geotiff RGB files')
        print('******************************\n')

        # get names of all 10 m resolution geotiff files
        os.chdir(tiffdir)
        allfiles = sorted([f for f in os.listdir(tiffdir) if f.endswith('.tif')])
        nfiles = len(allfiles)
        print('\nProcessing %d Geotiff files:' % nfiles)
        for thisfile in allfiles:
            print(thisfile)
        print('\n\n')

        ###################################################
        # read and plot the selected RGB bands / geotiffs onto a map
        ###################################################

        # identify the filenames of the geotiff files for RGB map display
        rgbfiles = []
        for i in bands:
            rgbfiles.append(allfiles[i - 1])
        for thisfile in rgbfiles:
            print(thisfile)
        print('\n\n')

        # open the first tiff file with GDAL to get file dimensions
        thisfile = allfiles[0]
        ds = gdal.Open(thisfile)
        data = ds.ReadAsArray()

        # get the projection information and convert to wkt
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        inproj = osr.SpatialReference()
        inproj.ImportFromWkt(proj)
        print(inproj)

        # convert wkt projection to Cartopy projection
        projcs = inproj.GetAuthorityCode('PROJCS')
        projection = ccrs.epsg(projcs)
        print(projection)

        # get the extent of the image
        extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
                  gt[3] + ds.RasterYSize * gt[5], gt[3])

        # read in the three geotiff files, one for each band
        rgbdata = read_sen2_rgb(rgbfiles)

        # close the GDAL file
        ds = None

        #######################################
        # make a plot of the tiff file in the image projection
        #######################################
        plotfile = allscenes[x].split('.')[0] + '_map1.jpg'
        title = allscenes[x].split('.')[0]
        mapextent = extent
        map_it(rgbdata, projection, mapextent, wd + shapefile,
               plotdir + plotfile,
               plottitle=title,
               figsizex=10, figsizey=10)

        # zoom out
        zf = 10
        plotfile = allscenes[x].split('.')[0] + '_map2.jpg'
        title = allscenes[x].split('.')[0]
        # need to unpack the tuple 'extent' and create a new tuple 'mapextent'
        mapextent = (extent[0] - (extent[1] - extent[0]) * zf,
                     extent[1] + (extent[1] - extent[0]) * zf,
                     extent[2] - (extent[3] - extent[2]) * zf,
                     extent[3] + (extent[3] - extent[2]) * zf)
        map_it(rgbdata, projection, mapextent, wd + shapefile,
               plotdir + plotfile,
               plottitle=title,
               figsizex=10, figsizey=10)

        # zoom in to the upper right corner
        zf = 1 / 16
        plotfile = allscenes[x].split('.')[0] + '_map3.jpg'
        title = allscenes[x].split('.')[0]
        # need to unpack the tuple 'extent' and create a new tuple 'mapextent'
        mapextent = (extent[0] - (extent[1] - extent[0]) * zf,
                     extent[1] + (extent[1] - extent[0]),
                     extent[2] - (extent[3] - extent[2]) * zf,
                     extent[3] + (extent[3] - extent[2]))
        map_it(rgbdata, projection, mapextent, wd + shapefile,
               plotdir + plotfile,
               plottitle=title,
               figsizex=10, figsizey=10)


#################################################
# TODO testing from here on... under development
#################################################

# source: http://www.net-analysis.com/blog/cartopylayout.html

'''
Axes are your friend(s)

Excuses in advance to Matplotlib experts who read this, as I have probably got some of the key concepts askew (but that's all part of the learning). The main concept behind my solution is the concept that a figure can contain a number of Axes objects that co-exist. Each Axes object can be commanded to turn on or off the 'decorations' that surround the plotting content.

So the first thing is to define a function that will clear away all the decorations, leaving us with the equivalent of a blank space on the paper.

'''

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
        ax.tick_params(labelbottom='off', labeltop='off', labelleft='off', labelright='off' ,\
                        bottom='off', top='off', left='off', right='off' )
    #end blank_axes

'''

The term spine is a little misleading (you or I have only one spine, but an Axes has four: top, bottom, left, right). On a spine there are tick marks, and labels for those tick marks. We clear them all away.
Draw the outer frame

'''

fig = plt.figure(figsize=(10, 12))

# ------------------------------- Surrounding frame ------------------------------
# set up frame full height, full width of figure, this must be called first

left = -0.05
bottom = -0.05
width = 1.1
height = 1.05
rect = [left, bottom, width, height]
ax3 = plt.axes(rect)

# turn on the spines we want, ie just the surrounding frame
blank_axes(ax3)
ax3.spines['right'].set_visible(True)
ax3.spines['top'].set_visible(True)
ax3.spines['bottom'].set_visible(True)
ax3.spines['left'].set_visible(True)

ax3.text(0.01, 0.01, '© Don Cameron, 2017: net-analysis.com. ' +
         'Map generated at ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' from ' + theNotebook,
         fontsize=8)

'''
This outer frame is a little big larger than the usual square of side one, so the surround box doesn't touch any interior component. We set the enclosing spines visible, but supress tick marks, etc. Note that I have included a Copyright text, and some information to support reproducibility (the classic small print at the bottom of the document). I get the Jupyter Notebook name by some JavaScript hackery I got from Stack Exchange, but this could be a Python Module name, or anything you like.
Draw the spatial data
'''

# ---------------------------------  Main Map -------------------------------------
#
# set up main map almost full height (allow room for title), right 80% of figure

left = 0.2
bottom = 0
width = 0.8
height = 0.90
rect = [left, bottom, width, height]

ax = plt.axes(rect, projection=ccrs.PlateCarree(), )
ax.set_extent((150, 155, -30, -23))

ax.coastlines(resolution='10m', zorder=2)

# land polygons, including major islands, use cartopy default color
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

'''
I have allocated about 80% of the width of the map to the Cartopy spatial representation. The height is set to 90%, to allow a title at the top.

Because I want a high quality map, I have defined features using the Natural Earth 1:10Million scale data, as follows:


Note that by default, Cartopy draws its tick labels outside the main map, so we have to adjust the surrounding Axes objects so they don't overlap them.
Scale Bar

Now come the hacky-est part of this post. Basemap has a nifty method to draw scale bars (drawmapscale()), but so far as I know, Cartopy has no such method. Maybe this is because Cartopy is oriented towards those who want to display quantitive or qualitative data in a spatial context, and they don't expect people to use their maps to measure distance. However, in my view, a scale bar is part of the spatial data that should be displayed. Just how big are those Texas county borders that we keep seeing in examples, color-coded for poverty, or some such sociological attribute?

The double-sad part is that I know on no easy way to draw a scale bar in one Axes object, that will exactly match the distance measurements on the main map. Maybe if I define a second Cartopy Axes, with lat/lon range adjusted for the different sizes of the Axes objects ... but that is a matter for more investigation.

What all this means is that I can't follow the original blog post that prompted this post, which suggested the scale bar as a separate object, off to the side. I am going to have to write it on the main map. Not too bad a compromise.
Define the scale bar parameters
'''

lon0, lon1, lat0, lat1 = ax.get_extent()

# bar offset is how far from bottom left corner scale bar is (x,y) and how far up is scale bar text
bar_offset = [0.05, 0.05, 0.07]
bar_lon0 = lon0 + (lon1 - lon0) * bar_offset[0]
bar_lat0 = lat0 + (lat1 - lat0) * bar_offset[1]

text_lon0 = bar_lon0
text_lat0 = lat0 + (lat1 - lat0) * bar_offset[2]
bar_tickmark = 20000  # metres
bar_ticks = 5
bar_alpha = 0.3

bar_color = ['black', 'red']

'''
My scale bar will be 5% up, 5% right from the lower left corner; made up of 5 sub-bars each 20 km long, alternating red and black, and quite transparent (in case they hide something important; alpha=0.3). The text will be up 7%, above the bar.
Draw the Scale Bar
'''

# draw a scale bar that is a set of colored line segments (bar_ticks of these), bar_tickmarks long
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

'''
I have a helper function displace() that takes:

    a lat/lon (in degrees - no love for radians here!)

    a direction (degrees)

    a distance in metres

and returns the lat/lon at the end of that distance in that direction. I draw a set of lines (NOT bars), but as a result, I have to set the appearence of the line segment end as 'butt', or else the line will be drawn with rounded ends that go past the point I want them to stop or start at.
Draw the scale bar text
'''

# highlight text with white background
buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
# Plot the scalebar label
units = 'km'
# TODO make transform match ax projection
t0 = ax.text(text_lon0, text_lat0, str(bar_ticks * bar_tickmark / 1000) + ' ' + units, transform=ccrs.PlateCarree(),
             horizontalalignment='left', verticalalignment='bottom',
             path_effects=buffer, zorder=2)
'''

We put a white background behind the black scale bar text.
Locating Map
'''

# ---------------------------------Locating Map ------------------------
#
# set up index map 20% height, left 16% of figure
left = 0
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

'''
We reserve a small area at the bottom left for a locating map. In my example, I have drawn Australia (well, some far flung islands like Christmas Island may be missing). A set_global() call will force the whole globe to be shown, but that is zoomed too far back for my case. I add coastlines, and shade the land and oceans with the default Cartopy resolution and colors. I draw gridlines, and then draw a red box that represents the main map bounds.
The Title
'''

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

'''
The title is centered above the map. I could use some ornate font, but just went with the Cartopy defaults. Note that no Cartopy projection systems are used, just the 0.0-1.0 default Axes object units. We turn off all spines, etc.
The North Arrow
'''

# ---------------------------------North Arrow  ----------------------------
#
left = 0
bottom = 0.2
width = 0.16
height = 0.2
rect = [left, bottom, width, height]
rect = [left, bottom, width, height]
ax4 = plt.axes(rect)

# need a font that support enough Unicode to draw up arrow. need space after Unicode to allow wide char to be drawm?
ax4.text(0.5, 0.0, u'\u25B2 \nN ', ha='center', fontsize=30, family='Arial', rotation=0)
blank_axes(ax4)

'''
The North arrow lives above the locating map. The only gotchas of note here are:

    You need to specify a font family that support the Unicode character used (the default font doesn't, in my case). I am uncertain as to where Matplotlib / Cartopy gets its fonts from, so there may be scope for a more florid North arrow.

    In my case, the Unicode needed a following space before the training newline character, to be drawn properly. Why, I don't know.

In my case, my map is NOT rotated, but the text( rotation= ) parameter would allow for a rotated arrow, if needed.
the Legend

The legend is bit hacky as well. A simple call to the Matplotlib 'make me a legend' usually fails because under the covers, the necessary calls haven't been made to link a drawing object (like a line with linestyle and linecolor), to a name. So we have to do all that.

The trick is to create some drawing entities (patches and lines), get the handles to these, associate them with names, and then create a legend. It's not all bad, it does give you the chance to control the legend in fine detail. Spoiler alert, GeoPandas legends can be quite painful.
Create the Axes Object
'''

# ------------------------------------  Legend -------------------------------------

# legends can be quite long, so set near top of map (0.4 - bottom + 0.5 height = 0.9 - near top)
left = 0
bottom = 0.4
width = 0.16
height = 0.5
rect = [left, bottom, width, height]
rect = [left, bottom, width, height]
ax5 = plt.axes(rect)
blank_axes(ax5)

# Create Area Legend Entries

# create an array of color patches and associated names for drawing in a legend
# colors are the predefined colors for cartopy features (only for example, Cartopy names are unusual)
colors = sorted(cartopy.feature.COLORS.keys())

# handles is a list of patch handles
handles = []
# names is the list of corresponding labels to appear in the legend
names = []

# for each cartopy defined color, draw a patch, append handle to list, and append color name to names list
for c in colors:
    patch = mpatches.Patch(color=cfeature.COLORS[c], label=c)
    handles.append(patch)
names.append(c)

'''
Note that these Patches never appear in our diagram. Now Cartopy color names are not perfect, but it's only an example. If at some time Cartopy expend their color range, the code above should still work.
Create Line Legend Entries
'''

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

'''
Note that here we have turned off all markers, and have invented our own labels to appear in the legend.
Create the Legend, and Display All!
'''

# create legend
ax5.legend(handles, names)
ax5.set_title('Legend', loc='left')

plt.show()

