"""
Cirrus correction from Kustiyo
Needs: S2 L1 RGB and Cirrus mask


"""


from osgeo import gdal
import numpy as np
from pyeo import raster_manipulation as ras

def cirrus_correction(stacked_raster_path, out_path):
    stacked_raster = gdal.Open(stacked_raster_path)
    ras_array = stacked_raster.GetVirtualMemArray()
    r_band = ras_array[0]
    g_band = ras_array[1]
    b_band = ras_array[2]
    cirrus_band = ras_array[3]
    out_raster = ras.create_matching_dataset(stacked_raster, out_path, bands = 3)
    out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)

    for ii, band in enumerate([r_band, g_band, b_band]):
        out_array[ii, ...] = band - ((cirrus_band - 100)*12/(np.log(cirrus_band - 100) +1))

    out_array = None
    b_band = None
    g_band = None
    b_band = None
    cirrus_band = None
    out_array = None
    out_raster = None
    stacked_raster = None
