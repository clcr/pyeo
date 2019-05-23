# This model was developed by Prof. Dr. Lilik Budi Prasetyo, Dr. Yudi Setiyawan, Desi Suyamto, Sahid Hudjimartsu. Faculty of Forestry, Bogor Agricultural University
# References: Hudjimartsu, S., Prasetyo, L., Setiawan, Y. and Suyamto, D., 2017, November.
# Https://github.com/Forests2020-Indonesia/Topographic-Correction
# Illumination Modelling for Topographic Correction of Landsat 8 and Sentinel-2A Imageries.
# In European Modelling Symposium (EMS), 2017 (pp. 95-99). IEEE.

# The model is updated by Dr. Yaqing Gou, John F Roberts and Polyanna Bispo from UoL under Forest 2020 project
# Main updates are:
# (1) rewrite to python 3
# (2) rewrite to apply to Sentinel 2

#TODO: current version the code import numexpr
#TODO: current code works for 20m so far
#TODO: recheck the meta data imported MTD_MSIL2A.xml, if it's generated from sen2cor during the process. If true,
#      and if we are give up sen2cor need to find alternative for it (line 82)

from datetime import datetime, date
import numpy as np
from osgeo import gdal, osr
from scipy.stats import linregress
import numexpr
import pdb
import xml.etree.ElementTree as ET
import s2_functions
from os import system
import sys, os, glob, subprocess, zipfile, shutil, math, numexpr
from pyproj import Proj, transform


def jp2toTif(injp2):
    outtif = injp2[:-4]+'.tif'
    os.system('gdal_translate -of GTiff ' + injp2 +' ' + outtif)

def extractFile (input):
    zipFile = glob.glob(input + "/*.zip")
    filename = zipFile[0].split("\\")
    folderName = zipped_safe_path + "\\" + filename[-1][:-4]
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    else:
        shutil.rmtree(folderName)
        os.makedirs(folderName)
    zip = zipfile.ZipFile(zipFile[0])
    zip.extractall(folderName)
    return folderName

def resample(Input, Output):
    for o in Input:
        file = o.split("\\")
        out_path = Output+ "/" +file[-1].split("_")[-1]
        system('gdalwarp -overwrite -s_srs EPSG:{proj} -r near -ts {N1} {N2} -of GTiff {o} {outputR}'.format(proj=projection, N1=resolution.RasterXSize, N2=resolution.RasterXSize, o=o, outputR=out_path))

# TODO: rewrite to use Datetime lib
def year_date(data_sensing_start):
    year_file=data_sensing_start.split("T")[0]
    date_file=data_sensing_start.split("T")[1]
    date_file2= date_file [:-1]
    all= year_file+" "+date_file2
    parsing = datetime.strptime(all, '%Y-%m-%d %H:%M:%S.%f')
    return parsing

def hour(dt):
    h=dt.hour+7
    return h
def second(dt):
    s= float(dt.microsecond)/1000000+dt.second
    return s
def leap(dt):
    if (dt.year % 4) == 0:
        if (dt.year % 100) == 0:
            if (dt.year % 400) == 0:
               a = int(366)
            else:
                a = int(365)
        else:
            a= int(366)
    else:
        a= int(365)
    return a

def cos(x):
    cos= np.cos(np.deg2rad(x))
    return cos
def sin(x):
    sin=np.sin(np.deg2rad(x))
    return sin

def day(dt):
    day_date= date(dt.year, dt.month, dt.day)
    sum_of_day=int(day_date.strftime('%j'))
    return sum_of_day

############################
# set input
############################

safe_folder = "/media/ubuntu/storage/Indonesia/s2/L2/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031450.SAFE"
m20_res_folder =  safe_folder + '/GRANULE/L2A_T48MXU_A011755_20170922T031450/IMG_DATA/R20m/'


############################
# import and process image
############################
# 1. unzip if needed
#safe_folder = extractFile(zipped_safe_path)

# 2. selecting image data for projection and resolution
outname = []
for root, dirs, files in os.walk((os.path.normpath(m20_res_folder)), topdown=False):
        for name in files:
            if name.endswith('.jp2'):
                outname.append(os.path.join(root, name))
resolution = gdal.Open(outname[5])
proj = osr.SpatialReference(wkt=resolution.GetProjection())
projection =proj.GetAttrValue('AUTHORITY', 1)

# 3. Resample for same resolution (to what?)  <-- not working?
resample(outname, safe_folder)

# 4. searching metadata from xml file for calculating Solar Position
filexml = glob.glob(safe_folder + '/MTD_MSIL2A.xml') #is it generated from sen2cor?
print('find matching meta data: ' + filexml)
tree = ET.parse(filexml[0])
root = tree.getroot()
data_sensing_start= tree.findtext("//DATATAKE_SENSING_START")

# 5. import raster data to be corrected
raster_list=glob.glob(m20_res_folder + '/*.jp2')
read=[]
for i in raster_list:
    band=gdal.Open(i)
    read.append(band.GetRasterBand(1).ReadAsArray().astype(float)/10000.0)    #Double-check this
filename=[]
for a in [os.path.basename(x) for x in glob.glob(m20_res_folder + '/*.jp2')]:
    p=os.path.splitext(a)[0]
    y = p.split("_")
    filename.append(y[-2])

# 6. Builds a dictionary of filename:data from that file
s2_image_dict= dict(zip(filename, read))


###################################################
# Preprocess & generate slope and aspect from DEM
####################################################
# mosaic into one regional DEM
dem_path = '/media/ubuntu/storage/Indonesia/SRTM'
dem_region = '/media/ubuntu/storage/Indonesia/SRTM/Indonesia_merge.tif'
s2_functions.mosaicTiff_text(input_dir=dem_path,outname = merge_dem,search_suffix = '.tif', filetype = 'Int16')

#re-project to UTM, otherwise, can also use different scale number 111120 (if elevation is in meters)
merge_dem_prj= dem_region[:-4] + '_prj.tif'
os.system('gdalwarp ' + merge_dem + ' ' + merge_dem_prj + ' -t_srs "EPSG:32748" -tr 20 20 -overwrite') #UTM48s

##calculate slope and aspect
slope_out = merge_dem_prj[:-4]+'_slop.tif'
os.system('gdaldem slope '+merge_dem_prj + ' ' + slope_out + ' -s 1.') #can also use different scale number 111120 (if elevation is in meters)

aspect_out = merge_dem_prj[:-4]+'_aspect.tif'
os.system('gdaldem aspect '+merge_dem_prj + ' ' + aspect_out)

#cutline into the extent of the s2 image
cut_img = raster_list[2]
cut_img_outline = cut_img[:-4]+'.shp'
os.system('gdaltindex '+cut_img_outline + ' '+cut_img)

slope_out_clip = slope_out[:-4] + '_clip.tif'
aspect_out_clip = aspect_out[:-4] +'_clip.tif'

os.system('gdalwarp -cutline '+  cut_img_outline + ' -crop_to_cutline -overwrite ' +slope_out + ' '+ slope_out_clip)
os.system('gdalwarp -cutline '+  cut_img_outline + ' -crop_to_cutline -overwrite ' +aspect_out + ' '+ aspect_out_clip)

###################################################
# Caculate solar Position
####################################################

pathDem = r"/media/ubuntu/storage/Indonesia/SRTM/testDEM"
raster_list_dem=glob.glob(pathDem+ '/*.tif')
readDem=[]
print(raster_list_dem)
for i in raster_list_dem:
    band2=gdal.Open(i)
    readDem.append(band2.GetRasterBand(1).ReadAsArray().astype(float))


dt=year_date(data_sensing_start)

print ("Calculating Solar Position...")
gamma=((2 * math.pi) / leap(dt)) * ((day(dt) - 1) + (((hour(dt)+dt.minute/60+second(dt)/3600) - 12) / 24) )# degree

#sun declination angle
decl=0.006918 - 0.399912 * cos(gamma) + 0.070257 * sin(gamma) - 0.006758 * cos (2 * gamma)\
     + 0.000907 * sin (2 * gamma) - 0.002697 * cos (3 * gamma) + 0.00148 * sin (3 * gamma) #radians
decl_deg= (360 / (2 * math.pi)) * decl

#lat long value
# get columns and rows of your image from gdalinfo
xoff, a, b, yoff, d, e = band.GetGeoTransform()
def pixel2coord(x, y):
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)
rows=read[0].shape[0]
colms=read[0].shape[1]
coordinate=[]
for row in  range(0,rows):
  for col in  range(0,colms):
      coordinate.append(pixel2coord(col,row))
coor_2=np.array(coordinate, dtype=float)
long=coor_2[:,0]
lat=coor_2[:,1]

# convert UTM to GCS 84 ( this step must be done because the calculation of the Sun's position is in degrees, not meters)
inProj = Proj(init='epsg:'+ band.GetProjection()[-8:-3])
outProj = Proj(init='epsg:4326')
longG, latG = transform(inProj, outProj, long, lat)
long_n=longG.reshape(rows,colms)
lat_n=latG.reshape(rows,colms)

#eqtime
eqtime = 229.18 * (0.000075 + 0.001868 * cos(gamma) - 0.032077 * sin(gamma) - 0.014615 * cos(2 * gamma) - 0.040849 * sin(2 * gamma))  # minutes
timeoff= eqtime - 4 * long_n + 60 * 7 #minutes
tst=hour() * 60 + dt.minute + second() / 60 + timeoff #minutes
ha=(tst /4)-180 #degree

#sun zenith angle
zenit1 =sin(lat_n)* sin(decl_deg) + cos (lat_n)* cos(decl_deg) * cos(ha)
zenit2=np.arccos(zenit1) #radians
zenit_angle= np.rad2deg(zenit2)

#sun azimuth angle
theta1= -1 * ((sin(lat_n)) * cos(zenit_angle)- sin(decl_deg)/(cos (lat_n) * sin (zenit_angle)))
theta2=np.arccos(theta1) #radians
theta3=np.rad2deg(theta2)#degree
azimuth_angle=180 - theta3 #degrees

ASPECT = readDem[0]
SLOPE = readDem[1]

# IC calculation
delta=azimuth_angle - ASPECT
IC=(cos(zenit_angle)* cos (SLOPE)) + (sin(zenit_angle) * sin (SLOPE) * cos(delta))#radians

# sample
Nir = s2_image_dict['B8A']   # Needs to be B8A for 20m resultion, B08 for 10m
Red = s2_image_dict['B04']
NDVI=numexpr.evaluate("(Nir - Red) / (Nir + Red)")
sample_ndvi= numexpr.evaluate("(NDVI >0.5) & (SLOPE >= 18)")
area_true= sample_ndvi.nonzero() #outputnya index row n col
a_true=area_true[0]
b_true=area_true[1]

#correction
cos_zenith= cos(zenit_angle)

#auto
#def IC_all(my_dict):
temp={}
IC_final={}
for y in s2_image_dict:
    val2=s2_image_dict[y]
    temp[y]=val2[a_true,b_true].ravel()
    IC_true=IC[a_true,b_true].ravel()
    slope=linregress(IC_true, temp[y])
    IC_final[y]= s2_image_dict[y] - (slope[0] * (IC - cos_zenith))
print ("Exporting to GeoTIFF...")
#export auto
for item in IC_final:
    geo = band.GetGeoTransform()
    proj = band.GetProjection()
    shape = s2_image_dict['B8A'].shape # Needs to be B8A for 20m resultion, B08 for 10m
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(safe_folder + "\\" + item + "corr.TIF", shape[1], shape[0], 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geo)
    dst_ds.SetProjection(proj)
    ds=dst_ds.GetRasterBand(1)
    ds.SetNoDataValue(-9999)
    ds.WriteArray(IC_final[item])
    dst_ds.FlushCache()
    dst_ds = None  # save, close""


###################################################
# merge bands into one image and compare with the un-corrected image
####################################################

outtif = os.path.join(safe_folder,'all.tif')
s2_functions.mergeTiff_text(input_dir=safe_folder,outname=outtif,search_suffix = 'corr.TIF', driver = 'gdal_merge', filetype = 'Float32')

outtif = os.path.join(safe_folder,'orgianl_all.tif')
s2_functions.mergeTiff_text(input_dir=m20_res_folder,outname=outtif,search_suffix = '.tif', driver = 'gdal_merge', filetype = 'Float32')




