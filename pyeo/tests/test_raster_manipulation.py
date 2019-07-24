import glob
import os
import shutil

import gdal
import osr

import pyeo.masks
import pyeo.raster_manipulation


def test_composite_images_with_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/composite_test.tif")
    except FileNotFoundError:
        pass
    test_data = [r"test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
                 r"test_data/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif"]
    out_file = r"test_outputs/composite_test.tif"
    pyeo.raster_manipulation.composite_images_with_mask(test_data, out_file, generate_date_image=True)
    image = gdal.Open("test_outputs/composite_test.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert image_array.max() > 10


def test_composite_across_projections():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/composite_test.tif")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(r"test_outputs/reprojected")
    except FileNotFoundError:
        pass
    os.mkdir(r"test_outputs/reprojected")
    epsg = 4326
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsg)
    projection = proj.ExportToWkt() # Refactor this terrible nonsense later
    test_data = [r"test_data/S2A_MSIL2A_20180703T073611_N0206_R092_T36MZE_20180703T094637.tif",
                 r"test_data/S2B_MSIL2A_20180728T073609_N0206_R092_T37MBV_20180728T114325.tif"]
    pyeo.raster_manipulation.reproject_image(test_data[0], r"test_outputs/reprojected/0.tif", projection)
    pyeo.raster_manipulation.reproject_image(test_data[1], r"test_outputs/reprojected/1.tif", projection)
    pyeo.raster_manipulation.reproject_image(pyeo.masks.get_mask_path(test_data[0]), r"test_outputs/reprojected/0.msk", projection)
    pyeo.raster_manipulation.reproject_image(pyeo.masks.get_mask_path(test_data[1]), r"test_outputs/reprojected/1.msk", projection)

    out_file = r"test_outputs/composite_test.tif"
    pyeo.raster_manipulation.composite_images_with_mask([
       r"test_outputs/reprojected/0.tif",
        r"test_outputs/reprojected/1.tif"],
        out_file)
    image = gdal.Open("test_outputs/composite_test.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert image_array.max() > 10


def test_composite_across_projections_meters():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/composite_test.tif")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(r"test_outputs/reprojected")
    except FileNotFoundError:
        pass
    os.mkdir(r"test_outputs/reprojected")
    epsg = 32736
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsg)
    projection = proj.ExportToWkt() # Refactor this terrible nonsense later

    test_data = [r"test_data/S2A_MSIL2A_20180703T073611_N0206_R092_T36MZE_20180703T094637.tif",
                 r"test_data/S2B_MSIL2A_20180728T073609_N0206_R092_T37MBV_20180728T114325.tif"]
    pyeo.raster_manipulation.reproject_image(test_data[0], r"test_outputs/reprojected/0.tif", projection)
    pyeo.raster_manipulation.reproject_image(test_data[1], r"test_outputs/reprojected/1.tif", projection)
    pyeo.raster_manipulation.reproject_image(pyeo.masks.get_mask_path(test_data[0]), r"test_outputs/reprojected/0.msk", projection)
    pyeo.raster_manipulation.reproject_image(pyeo.masks.get_mask_path(test_data[1]), r"test_outputs/reprojected/1.msk", projection)

    out_file = r"test_outputs/composite_test.tif"
    pyeo.raster_manipulation.composite_images_with_mask([
       r"test_outputs/reprojected/0.tif",
        r"test_outputs/reprojected/1.tif"],
        out_file)
    image = gdal.Open("test_outputs/composite_test.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert image_array.max() > 1


def test_reprojection():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/reprojection_test.tif")
    except FileNotFoundError:
        pass
    new_projection = r"""PROJCS["WGS 84 / UTM zone 36S",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",33],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32736"]]"""
    image = r"test_data/S2B_MSIL2A_20180728T073609_N0206_R092_T37MBV_20180728T114325.tif"
    out_file = r"test_outputs/reprojection_test.tif"
    pyeo.raster_manipulation.reproject_image(image, out_file, new_projection)
    result = gdal.Open(out_file)
    assert result
   # assert result.GetProjection() == new_projection
    result_array = result.GetVirtualMemArray()
    assert result_array.max() > 10


def test_buffered_composite():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/buffered_composite_test.tif")
    except FileNotFoundError:
        pass
    test_data = [r"test_data/buffered_masks/20180103T172709.tif",
                 r"test_data/buffered_masks/20180319T172021.tif",
                 r"test_data/buffered_masks/20180329T171921.tif"]
    out_file = r"test_outputs/buffered_composite_test.tif"
    pyeo.raster_manipulation.composite_images_with_mask(test_data, out_file)


def test_composite_off_by_one():
    """The images in test_outputs/off_by_one were producing off-by-one errors"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        shutil.rmtree("test_outputs/off_by_one_error")
    except FileNotFoundError:
        pass
    os.mkdir("test_outputs/off_by_one_error")
    pyeo.raster_manipulation.composite_directory("test_data/off_by_one", "test_outputs/off_by_one_error", generate_date_images=True)


def test_raster_sum():
    test_dir = "/media/ubuntu/data_archive/F2020/Kenya/outputs/classifications/mt_elgon"
    test_pattern = '*_T37MDR_*_rcl.tif'
    tile_id = 'T37MDR'
    fn = 'KEN_cherangany_' + tile_id + '_forestChange_sum2018.tif'
    out_fn = os.path.join(test_dir, fn)
    test_image_list = glob.glob(os.path.join(test_dir, test_pattern))
    pyeo.raster_manipulation.raster_sum(inRstList=test_image_list, outFn=out_fn)