import pytest
import tempfile
import gdal
import ogr
import osr
import os
import numpy as np

# Configuration code from https://docs.pytest.org/en/latest/example/simple.html


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     default=False, help="run slow tests")
    parser.addoption("--runweb", action="store_true",
                     default=False, help="run web-based tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        # --runslow not given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--runweb"):
        # --runnonlocal not given in cli; skip nonlocals
        skip_nonlocal = pytest.mark.skip(reason="need --runweb option to run")
        for item in items:
            if "webtest" in item.keywords:
                item.add_marker(skip_nonlocal)


# My own test environment, build around this class

class TestGeodataManager:
    """
    An object that provides methods to create and manipulate a temporary
    directory that contains geotiffs, shapefiles, ect
    All are created with the same projection, corner locations all
    default to 100,100

    Usage: with TestGeodataMangager() as tgm:
        tgm.create_temp_tiff(....)

    Attributes
    ----------
    path : str
        The path to the temporary folder
    images : list[str]
        A list of images in the temporary folder



    """

    def create_temp_tiff(self, name, content=np.ones([3, 3, 3]), geotransform=(10, 10, 0, 10, 0, -10)):
        """Creates a temporary geotiff in self.path
        """
        if len(content.shape) != 3:
            raise IndexError
        path = os.path.join(self.path, name)
        driver = gdal.GetDriverByName('GTiff')
        new_image = driver.Create(
            path,
            xsize=content.shape[1],
            ysize=content.shape[2],
            bands=content.shape[0],
            eType=gdal.GDT_Byte
        )
        new_image.SetGeoTransform(geotransform)
        for band in range(content.shape[0]):
            raster_band = new_image.GetRasterBand(band+1)
            raster_band.WriteArray(content[band, ...].T)
        new_image.SetProjection(self.srs.ExportToWkt())
        new_image.FlushCache()
        self.images.append(new_image)
        self.image_paths.append(path)

    def create_100x100_shp(self, name):
        """Cretes  a shapefile with a vector layer named "geometry" containing a 100mx100m square , top left corner
        being at wgs coords 10,10.
        This polygon has a field, 'class' with a value of 3. Left in for back-compatability"""
        # TODO Generalise this
        vector_file = os.path.join(self.temp_dir.name, name)
        shape_driver = ogr.GetDriverByName("ESRI Shapefile")  # Depreciated; replace at some point
        vector_data_source = shape_driver.CreateDataSource(vector_file)
        vector_layer = vector_data_source.CreateLayer("geometry", self.srs, geom_type=ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(10.0, 10.0)
        ring.AddPoint(10.0, 110.0)
        ring.AddPoint(110.0, 110.0)
        ring.AddPoint(110.0, 10.0)
        ring.AddPoint(10.0, 10.0)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        vector_feature_definition = vector_layer.GetLayerDefn()
        vector_feature = ogr.Feature(vector_feature_definition)
        vector_feature.SetGeometry(poly)
        vector_layer.CreateFeature(vector_feature)
        vector_layer.CreateField(ogr.FieldDefn("class", ogr.OFTInteger))
        feature = ogr.Feature(vector_layer.GetLayerDefn())
        feature.SetField("class", 3)

        vector_data_source.FlushCache()
        self.vectors.append(vector_data_source)  # Check this is the right thing to be saving here
        self.vector_paths.append(vector_file)


    def create_temp_shape(self, name, point_list):
        vector_file = os.path.join(self.temp_dir.name, name)
        shape_driver = ogr.GetDriverByName("ESRI Shapefile")  # Depreciated; replace at some point
        vector_data_source = shape_driver.CreateDataSource(vector_file)
        vector_layer = vector_data_source.CreateLayer("geometry", self.srs, geom_type=ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in point_list:
            ring.AddPoint(point[0], point[1])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        vector_feature_definition = vector_layer.GetLayerDefn()
        vector_feature = ogr.Feature(vector_feature_definition)
        vector_feature.SetGeometry(poly)
        vector_layer.CreateFeature(vector_feature)
        vector_layer.CreateField(ogr.FieldDefn("class", ogr.OFTInteger))
        feature = ogr.Feature(vector_layer.GetLayerDefn())
        feature.SetField("class", 3)

        vector_data_source.FlushCache()
        self.vectors.append(vector_data_source)  # Check this is the right thing to be saving here
        self.vector_paths.append(vector_file)


    def __init__(self, srs=4326):
        self.srs = osr.SpatialReference()
        self.srs.ImportFromEPSG(srs)

    def __enter__(self):
        """Creates a randomly named temp folder
        """
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name)
        self.path = temp_path
        self.temp_dir = temp_dir
        self.images = []
        self.image_paths = []
        self.vectors = []
        self.vector_paths = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.temp_dir.cleanup()


@pytest.fixture
def managed_geotiff_dir():
    """Holds context for the TestGeotiffManager class defined above"""
    with TestGeodataManager() as tgm:
        tgm.create_temp_tiff("temp.tif")
        yield tgm


@pytest.fixture
def managed_geotiff_shapefile_dir():
    """Creates a temp dir with a globally contiguous shapefile and geotiff"""
    with TestGeodataManager() as tgm:
        array = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1],
                           [1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1],
                           [1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1],
                           [1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1],
                           [1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]], dtype=np.byte)
        tgm.create_temp_tiff("temp.tif", np.transpose(array, (0, 2, 1)))
        tgm.create_100x100_shp("temp.shp")
        yield tgm


@pytest.fixture
def managed_multiple_wrong_geotransform_geotiff_dir():
    """Creates a temporary file with 4 contiguous single-band geotiffs that
     don't have an aligned geotransform"""
    with TestGeodataManager() as tgm:
        array = np.array([[[1, 2,
                            2, 4,
                            3, 5]]], dtype=np.byte)
        tgm.create_temp_tiff("band_1.tif", np.transpose(array, (0, 2, 1)))
        tgm.create_temp_tiff("band_2.tif", np.transpose(array+10, (0, 2, 1)))
        tgm.create_temp_tiff("band_3.tif", np.transpose(array+20, (0, 2, 1)))
        tgm.create_temp_tiff("band_4.tif", np.transpose(array+30, (0, 2, 1)))
        yield tgm


@pytest.fixture
def managed_multiple_geotiff_dir():
    """Creates a temporary file with 4 contiguous single-band 10m pixel geotiffs"""
    with TestGeodataManager() as tgm:
        array = np.array([[[1, 2,
                            2, 4,
                            3, 5]]], dtype=np.byte)
        gt = (10, 10, 0, 10, 0, -10)
        tgm.create_temp_tiff("band_1.tif", np.transpose(array, (0, 2, 1)), geotransform=gt)
        tgm.create_temp_tiff("band_2.tif", np.transpose(array+10, (0, 2, 1)), geotransform=gt)
        tgm.create_temp_tiff("band_3.tif", np.transpose(array+20, (0, 2, 1)), geotransform=gt)
        tgm.create_temp_tiff("band_4.tif", np.transpose(array+30, (0, 2, 1)), geotransform=gt)
        yield tgm


@pytest.fixture
def managed_noncontiguous_geotiff_dir():
    """Creates a temporary file with 2 non-contiguous geotiffs and
    a shapefile covering the contigous area"""
    # Test data is a five band 11x12 pixel geotiff
    test_data = np.array(range(660))
    test_data = test_data.reshape([5, 11, 12])
    with TestGeodataManager() as tgm:
        tgm.create_100x100_shp("aoi")
        tgm.create_temp_tiff(name="se_test", content=test_data, geotransform=(10, 10, 0, 10, 0, -10))
        tgm.create_temp_tiff(name="ne_test", content=test_data, geotransform=(0, 10, 0, 0, 0, -10))
        yield tgm


@pytest.fixture
def managed_ml_geotiff_dir():
    """Contains a 10x12 geotiff with 10m pixels called "training_data" with 8 bands for ml input
    Each pixel has the same value in each band
    Also contains a shapefile covering most of the image"""
    with TestGeodataManager() as tgm:
        test_data = np.empty((8, 12, 10))
        for y in range(12):
            for x in range(10):
                test_data[:, y, x] = x*12 + y
        tgm.create_temp_tiff(name="training_image", content=test_data)
        tgm.create_100x100_shp("training_shape")
        yield tgm


@pytest.fixture
def geotiff_dir():
    """

    Returns
    -------
    A pointer to a temporary folder that contains a 3-band geotiff
    of 3x3, with all values being 1.

    """
    tempDir = tempfile.TemporaryDirectory()
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)
    metadata = driver.GetMetadata()
    tempPath = os.path.join(tempDir.name)
    testDataset = driver.Create(os.path.join(tempDir.name, "tempTiff.tif"),
                                xsize=3, ysize=3, bands=3, eType=gdal.GDT_CFloat32)
    for i in range(3):
        testDataset.GetRasterBand(i+1).WriteArray(np.ones([3, 3]))
    testDataset = None
    yield tempPath
    tempDir.cleanup()