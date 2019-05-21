"""Functions for implementing terrain correction algorithm (credit to Wim Nursal, LAPAN"""

import gdal

def do_terrain_correction(in_safe_file, out_path, dem_path):
    """Corrects for shadow effects due to terrain features.
    Takes in a L2 SAFE file and a DEM, and produces a really boring image.
    Algorithm:
    -Generate slope and aspect from DEM using gdaldem
    -Calculate solar position from datatake sensing start and location of image
    -Calculate the correction factor for that image from the sun zenith angle, azimuth angle, DEM aspect and DEM slope
    -Build a mask of green areas using NDVI
    -Perform a linear regression based on that IC calculation and the contents of the L2 image to get ground slope(?)
    -Correct pixel p in original image with following: p_out = p_in - (ground_slope*(IC-cos(sun_zenith)))
    -Write to output
    NOTE TO SELF: Watch out; Wim has recast the bands to float. Beware off-by-one and rounding errors"""


def download_dem():
    """Downloads a DEM (probably JAXA) for the relevent area (maybe)"""

def get_dem_slope_and_angle(dem_path, slope_out_path, aspect_out_path):
    dem = gdal.Open(dem_path)
    gdal.DEMProcessing(slope_out_path, dem, "slope", slopeFormat="degree")
    gdal.DEMProcessing(aspect_out_path, dem, "aspect")





