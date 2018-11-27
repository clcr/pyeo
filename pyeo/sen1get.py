# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:26:46 2017

@author: Heiko Balzter
"""

###########################################
# search for and download Sentinel-1 scenes
# written for Python 3.6.4 on Ubuntu 16
###########################################

#import numpy as np
#import pylab
#import cartopy.io.shapereader as shpreader
from collections import OrderedDict
from osgeo import ogr
import os
from os import listdir
from os.path import isfile, join
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
import json
#import geojson as gj
#from geojson import Polygon

# OPTIONS
ndown = 2 # number of scenes to be downloaded (in order of least cloud cover)
wd = '/home/heiko/linuxpy/s1/' # working directory on Virtualbox
#wd = '/home/heiko/linuxpy/spacepark/' # working directory on Virtualbox
#wd = '/scratch/clcr/shared/py/' # working directory on ALICE
#shapefile = 'Sitios_Poly.shp' # ESRI Shapefile of the study area
shapefile = 'spacepark.shp' # ESRI Shapefile of the study area
datefrom = '20180801' # start date for imagery search
dateto   = '20180831' # end date for imagery search
#clouds = '[0 TO 20]'  # range of acceptable cloud cover % for imagery search
credentials = '/home/heiko/linuxpy/sencredentials.txt'  # contains two lines of text with username and password
                                                        # for the Sentinel Data Hub

##############################################################################
# MAIN
##############################################################################

###############################################
# load user credentials for Sentinel Data Hub at ESA
###############################################
# set working direcory
os.chdir(wd)

# read two lines of text with username and password
with open(credentials) as f:
    lines = f.readlines()
username = lines[0].strip()
password = lines[1].strip()
f.close()
api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')

###############################################
# load search area shapefile
###############################################
# get driver to read a shapefile
driver = ogr.GetDriverByName('ESRI Shapefile')

# open it
dataSource = driver.Open(shapefile, 0)
if dataSource is None:
    print('Could not open ' + shapefile)
    sys.exit(1) #exit with an error code

# get the layer from the shapefile
layer = dataSource.GetLayer()

# get the number of features in the layer
nfeat = layer.GetFeatureCount()
print('Feature count: ', nfeat)

# get the extent and projection of the layer
extent = layer.GetExtent()
print('Extent:', extent)
print('UL:', extent[0], extent[3])
print('LR:', extent[1], extent[2])
projlyr = layer.GetSpatialRef()
print('Projection information:', projlyr)

# get the first feature from the layer and print its geometry
feat = layer.GetFeature(0)
geom = feat.GetGeometryRef()
print('Geometry of feature 1:', geom)

###############################################
# convert the shapefile to geojson
###############################################
gjfile = shapefile.split(".")[0]+".geojson"
com = "ogr2ogr -f GeoJSON -t_srs crs:84 " + gjfile + " " + shapefile
flag = os.system(com)
if flag == 0:
    print('Shapefile converted to Geojson format: ' + gjfile)
else:
    print('Error converting shaoefile to Geojson')

# convert the geojson to wkt for the API search
footprint = geojson_to_wkt(read_geojson(gjfile))

# old code to open a geojson file directly
# with open(geojsonfile) as f:
#     polydata = gj.load(f)

###############################################
# search the ESA Sentinel data hub
###############################################

# set query parameters
query_kwargs = {
        'area': footprint,
        'platformname': 'Sentinel-1',
        'producttype': 'GRD',
#        orbitdirection='ASCENDING'),
        'date': (datefrom, dateto),
#        'processinglevel': 'Level-1C',
#        'cloudcoverpercentage': clouds
        }

# search the Sentinel data hub API
products = api.query(**query_kwargs)

# convert list of products to Pandas DataFrame
products_df = api.to_dataframe(products)
print('Search resulted in '+str(products_df.shape[0])+' satellite images with '+
      str(products_df.shape[1])+' attributes.')

# sort the search results
products_df_sorted = products_df.sort_values(['ingestiondate'], ascending=[True])
#print(products_df_sorted)
outfile = 'searchresults_full.csv'
products_df_sorted.to_csv(outfile)

# limit to first Ndown products sorted by lowest cloud cover and earliest acquisition date
products_df_n = products_df_sorted.head(ndown)
outfile = 'searchresults4download.csv'
products_df_n.to_csv(outfile)

# get the footprints of the selected scenes
s1footprints = products_df_n.footprint
outfile = 'searchresultsfootprints.csv'
s1footprints.to_csv(outfile)

###############################################
# download the selected scenes
###############################################
# make a 'data' directory (if it does not exist yet) to where the images will be downloaded
datadir = wd+"data/"
if not os.path.exists(datadir):
    print("Creating directory: ", datadir)
    os.mkdir(datadir)

# change to the 'data' directory
os.chdir(datadir)

# download sorted and reduced products in order
api.download_all(products_df_n['uuid'])

# save the footprints of the scenes marked for download together with their
#   metadata in a Geojson file
# first, run a new query to get the metadata for the selected scenes
products_n = OrderedDict()
for uuid in products_df_n['uuid']:
    kw = query_kwargs.copy()
    kw['uuid'] = uuid
    pp = api.query(**kw)
    products_n.update(pp)

# then, write the footprints and metadata to a geojson file
os.chdir(wd) # change to the working directory
outfile = 'footprints.geojson'
with open(outfile, 'w') as f:
    json.dump(api.to_geojson(products_n), f)

###############################################
# unzip the downloaded files
###############################################
# get list of all zip files in the data directory
os.chdir(datadir) # change to the data directory
allfiles = [f for f in listdir(datadir) if isfile(join(datadir, f))]
# unzip all files
for x in range(len(allfiles)):
    if allfiles[x].split(".")[1] == "zip":
        print("Unzipping file ", x+1, ": ", allfiles[x])
        os.system("unzip "+allfiles[x])
        # remove zip file after extraction
        os.remove(allfiles[x])
