# This model was developed by Prof. Dr. Lilik Budi Prasetyo, Dr. Yudi Setiyawan, Desi Suyamto, Sahid Hudjimartsu. Faculty of Forestry, Bogor Agricultural University
# References: Hudjimartsu, S., Prasetyo, L., Setiawan, Y. and Suyamto, D., 2017, November.
# Https://github.com/Forests2020-Indonesia/Topographic-Correction
# Illumination Modelling for Topographic Correction of Landsat 8 and Sentinel-2A Imageries.
# In European Modelling Symposium (EMS), 2017 (pp. 95-99). IEEE.

# The model is updated by Dr. Yaqing Gou, UoL under Forest 2020
# The main update is (1) to fit for python 3
# (2) to fit for Sentinel 2

from datetime import datetime, date
import numpy as np
import glob, math, os
from osgeo import gdal
from scipy.stats import linregress
#from dict import dict
import numexpr
import pdb
import xml.etree.ElementTree as ET


def jp2toTif(injp2):
    outtif = injp2[:-4]+'.tif'
    os.system('gdal_translate -of GTiff ' + injp2 +' ' + outtif)

s2_img_path = '/media/ubuntu/storage/Indonesia/s2/L2/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031450.SAFE'

import s2_functions
##test on 20m
jp2_list = s2_functions.search_files_fulldir(input_path=s2_img_path,search_key = '20m.jp2',search_type = 'end')

# for jp2 in jp2_list:
#     print(jp2)
#     jp2toTif(jp2)

print('Load Metadata')
metadata = s2_functions.search_files_fulldir(input_path=s2_img_path,search_key='MTD_MSIL2A.xml',search_type = 'end')[0] # come back for this, if it's L1C data will be MTD_MSIL1C.xml
print(metadata)

# import xmltodict
# with open(metadata) as fd:
#     data = xmltodict.parse(fd.read())
tree = ET.parse(metadata)
data = tree.findall("//SOLAR_IRRADIANCE") # // is for loop through everything

## for element in data:
##     print(element.text) # all the parameters we need
##     print(element.attrib) # a dictionary for band Id and unit

# #Load data raster
print("Loading Data Raster...")
# #Load data raster
#raster_list=glob.glob(path+ '*.TIF')
raster_list = s2_functions.search_files_fulldir(input_path=s2_img_path,search_key = '20m.tif',search_type = 'end')
dataRaster=[]
filename = []
for i in raster_list:
    band=gdal.Open(i)
    dataRaster.append(band.GetRasterBand(1).ReadAsArray().astype(float))

    p=os.path.basename(i)[:-4]
    filename.append(p)
my_dict= dict(zip(filename, dataRaster))

####Calculate slop and aspect from DEM
# mosaic big DEM
dem_path = '/media/ubuntu/storage/Indonesia/SRTM'
merge_dem = '/media/ubuntu/storage/Indonesia/SRTM/Indonesia_merge.tif'
#s2_functions.mosaicTiff_text(input_dir=dem_path,outname = merge_dem,search_suffix = '.tif', filetype = 'Int16')

#re-project to UTM, otherwise, can also use different scale number 111120 (if elevation is in meters)
merge_dem_prj= merge_dem[:-4] + '_prj.tif'
os.system('gdalwarp ' + merge_dem + ' ' + merge_dem_prj + ' -t_srs "EPSG:32748" -tr 20 20 -overwrite') #UTM48s

##calculate slope and aspect
slope_out = merge_dem_prj[:-4]+'_slop.tif'
os.system('gdaldem slope '+merge_dem_prj + ' ' + slope_out + ' -s 1.') #can also use different scale number 111120 (if elevation is in meters)

aspect_out = merge_dem_prj[:-4]+'_aspect.tif'
os.system('gdaldem aspect '+merge_dem_prj + ' ' + aspect_out)

#cutline
cut_img = raster_list[2]

slope_out_clip = slope_out[:-4] + '_clip.tif'
aspect_out_clip = aspect_out[:-4] +'_clip.tif'

cut_img_outline = cut_img[:-4]+'.shp'
#os.system('gdaltindex '+cut_img_outline + ' '+cut_img)

os.system('gdalwarp -cutline '+  cut_img_outline + ' -crop_to_cutline -overwrite ' +slope_out + ' '+ slope_out_clip)
os.system('gdalwarp -cutline '+  cut_img_outline + ' -crop_to_cutline -overwrite ' +aspect_out + ' '+ aspect_out_clip)

# #Load data raster aspect & slope
slope_tif = slope_out_clip
aspect_tif = aspect_out_clip

g_slope = gdal.Open(slope_tif)
slope = g_slope.GetRasterBand(1).ReadAsArray()
g_aspect = gdal.Open(aspect_tif)
aspect = g_aspect.GetRasterBand(1).ReadAsArray()
# pathname='Folder name' # folder consist of aspect and slope data
# raster_list_dem=glob.glob(pathname +'/*.TIF')
# dataTopo=[]
# for d in raster_list_dem:
#     band2=gdal.Open(d)
#     dataTopo.append(band2.GetRasterBand(1).ReadAsArray())
#
def year_date():
    year_file=data['DATE_ACQUIRED']
    date_file=data['SCENE_CENTER_TIME']
    date_file2= date_file [1:16]
    all= year_file+" "+date_file2
    parsing = datetime.strptime(all, '%Y-%m-%d %H:%M:%S.%f')
    return parsing

#dt=year_date()

sensing_start = data['n1:Level-2A_User_Product']['n1:General_Info']['L2A_Product_Info']['PRODUCT_START_TIME'][:-5] # not sure what .026Z after second means
sensing_end = data['n1:Level-2A_User_Product']['n1:General_Info']['L2A_Product_Info']['PRODUCT_STOP_TIME'][:-5]
time_format = '%Y-%m-%dT%H:%M:%S'

start = datetime.strptime(sensing_start, time_format)
end = datetime.strptime(sensing_end, time_format)
#mid_time = start + (start+end)/2

dt = start
#
# # UTC based on zoning area
# # This sample uses + 7, because in the western region of Indonesia
def hour():
    h=dt.hour+7
    return h
def second():
    s= float(dt.microsecond)/1000000+dt.second
    return s
def leap():
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
    return  cos
def sin(x):
    sin=np.sin(np.deg2rad(x))
    return sin
def day():
    day_date= date(dt.year, dt.month, dt.day)
    sum_of_day=int(day_date.strftime('%j'))
    return sum_of_day

print("Calculating Solar Position...")
gamma=((2 * math.pi) / leap()) * ((day() - 1) + (((hour()+dt.minute/60+second()/3600) - 12) / 24) )# degree


#sun declination angle
decl=0.006918 - 0.399912 * cos(gamma) + 0.070257 * sin(gamma) - 0.006758 * cos (2 * gamma)\
     + 0.000907 * sin (2 * gamma) - 0.002697 * cos (3 * gamma) + 0.00148 * sin (3 * gamma) #radians
decl_deg= (360 / (2 * math.pi)) * decl

#lat long value
# get columns and rows of your image from gdalinfo
xoff, a, b, yoff, d, e = g_slope.GetGeoTransform()
def pixel2coord(x, y):
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)
rows=dataRaster[0].shape[0]
colms=dataRaster[0].shape[1]
coordinate=[]
for row in  range(0,rows):
  for col in  range(0,colms):
      coordinate.append(pixel2coord(col,row))
coor_2=np.array(coordinate, dtype=float)
long=coor_2[:,0]
lat=coor_2[:,1]
long_n=long.reshape(rows,colms)
lat_n=lat.reshape(rows,colms)

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

# IC calculation
dataTopo=[]
dataTopo.append(aspect)
dataTopo.append(slope)

pdb.set_trace()
delta=azimuth_angle - dataTopo[0]
IC=(cos(zenit_angle)* cos (dataTopo[1])) + (sin(zenit_angle) * sin (dataTopo[1]) * cos(delta))#radians

print("Calculating Reflectances...")

pdb.set_trace()

############################################
#i stopped here because i can't get the equation right
###########################################
#Reflectance
reflectance_band1=(float(data['REFLECTANCE_MULT_BAND_1'])*my_dict[filename[0][:-2]+'B1']+float(data['REFLECTANCE_ADD_BAND_1']))/cos(zenit_angle)
reflectance_band2=(float(data['REFLECTANCE_MULT_BAND_2'])*my_dict[filename[0][:-2]+'B2']+float(data['REFLECTANCE_ADD_BAND_2']))/cos(zenit_angle)
reflectance_band3=(float(data['REFLECTANCE_MULT_BAND_3'])*my_dict[filename[0][:-2]+'B3']+float(data['REFLECTANCE_ADD_BAND_3']))/cos(zenit_angle)
reflectance_band4=(float(data['REFLECTANCE_MULT_BAND_4'])*my_dict[filename[0][:-2]+'B4']+float(data['REFLECTANCE_ADD_BAND_4']))/cos(zenit_angle)
reflectance_band5=(float(data['REFLECTANCE_MULT_BAND_5'])*my_dict[filename[0][:-2]+'B5']+float(data['REFLECTANCE_ADD_BAND_5']))/cos(zenit_angle)
reflectance_band6=(float(data['REFLECTANCE_MULT_BAND_6'])*my_dict[filename[0][:-2]+'B6']+float(data['REFLECTANCE_ADD_BAND_6']))/cos(zenit_angle)
reflectance_band7=(float(data['REFLECTANCE_MULT_BAND_7'])*my_dict[filename[0][:-2]+'B7']+float(data['REFLECTANCE_ADD_BAND_7']))/cos(zenit_angle)
reflectance_band9=(float(data['REFLECTANCE_MULT_BAND_9'])*my_dict[filename[0][:-2]+'B9']+float(data['REFLECTANCE_ADD_BAND_9']))/cos(zenit_angle)
reflectance_f= {filename[0][:-2]+'B1':reflectance_band1, filename[0][:-2]+'B2':reflectance_band2,filename[0][:-2]+'B3':reflectance_band3, filename[0][:-2]+'B4':reflectance_band4, filename[0][:-2]+'B5':reflectance_band5, filename[0][:-2]+'B6':reflectance_band6, filename[0][:-2]+'B7':reflectance_band7, filename[0][:-2]+'B9':reflectance_band9}



# Training sample to avoid the cloud
NDVI=numexpr.evaluate("(reflectance_band5 - reflectance_band4) / (reflectance_band5 + reflectance_band4)")
sampleArea= numexpr.evaluate("(NDVI >0.5) & (dataTopo[1] >= 18)")
area_true= sampleArea.nonzero()
a_true=area_true[0]
b_true=area_true[1]

#Topographic correction using Illumination condition and rotation model
temp={}
IC_final={}
for y in reflectance_f:
        val2=reflectance_f[y]
        temp[y]=val2[a_true,b_true].ravel()
        IC_true=IC[a_true,b_true].ravel()
        slope=linregress(IC_true, temp[y])
        IC_final[y]=reflectance_f[y]-(slope[0]*(IC-cos(zenit_angle)))
print("Exporting to GeoTIFF...")
#export auto
for item in IC_final:
    geo = band.GetGeoTransform()
    proj = band.GetProjection()
    shape = my_dict[filename[0][:-2]+'B1'].shape
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create("Folder Output" + "topo.TIF", shape[1], shape[0], 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geo)
    dst_ds.SetProjection(proj)
    ds=dst_ds.GetRasterBand(1)
    ds.SetNoDataValue(0)
    ds.WriteArray(IC_final[item])
    dst_ds.FlushCache()
    dst_ds = None  # save, close"""

