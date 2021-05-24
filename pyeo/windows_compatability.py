import os
import sys
import numpy as np
from osgeo import gdal
from osgeo.gdal_array import GDALTypeCodeToNumericTypeCode
import logging
from tempfile import NamedTemporaryFile
import logging

WINDOWS_PREFIX = "win"
IOS_PREFIX = "darwin"
log = logging.getLogger("pyeo")
# May rewrite to use numpy

class _WinHackVirtualMemArray(np.memmap):
    """
    A class that replaces gdal.Dataset.GetVirtualMemArray on Windows.
    Creates a numpy memmap and copies the 
    """
    def __new__(subtype, raster, eAccess = False):
        filepath = NamedTemporaryFile()
        obj = super(_WinHackVirtualMemArray, subtype).__new__(subtype,
                filepath,
                dtype = GDALTypeCodeToNumericTypeCode(raster.GetRasterBand(1).DataType),
                mode = "w+",
                shape = (raster.RasterCount, raster.RasterYSize, raster.RasterXSize)
                )
        obj.geotransform = raster.GetGeoTransform()
        obj.projection = raster.GetProjection()
        obj.out_path = raster.GetFileList()[0]
        obj.writeable = eAccess
        obj.raster = raster
        obj[...] = raster.ReadAsArray()
        obj.flush()
        log.debug("Memarray created at {}".format(filepath))
        return obj

    def __init__(self, raster, eAccess=False):
        shape = (raster.RasterCount, raster.RasterYSize, raster.RasterXSize)
        super().__init__()
        log.debug("Attributes attached to memarray:{}\n{}\n{}\n{}".format(
            self.geotransform,
            self.projection,
            self.out_path,
            self.writeable))

    def __array_finalize__(self, obj):
#        log.debug("Finalising array as {}".format(obj))
        if obj is not None:
            self.geotransform = getattr(obj, 'geotransform', None)
            self.projection = getattr(obj, 'projection', None)
            self.out_path = getattr(obj, 'out_path', None)
            self.raster = None
            self.writeable = 0

    def __del__(self):
        # If appropriate, we want the memmap to write on close
        log.debug("Preparing to remove {}".format(self))
        if self.writeable:
            #pdb.set_trace()
            for band_index, band in enumerate(self.__array__()[:, ...]):
                out_band = self.raster.GetRasterBand(band_index + 1)
                out_band.WriteArray(band)
                out_band.FlushCache()
                out_band = None
            self.raster.FlushCache()


if sys.platform.startswith(WINDOWS_PREFIX) or sys.platform.startswith(IOS_PREFIX):
    # WARNING. THIS IS A DARK ART AND SHOULD NOT BE REPLICATED
    # Monkeypatching outside of test environments is normally Very Bad,
    # and should only be attempted by those with special training or 
    # nothing to lose.
    log.warning("Windows or iOS detected; monkeypatching GetVirtualMemArray. Some functions may not respond as expected.")
    def WindowsVirtualMemArray(self, eAccess=None):
        return _WinHackVirtualMemArray(self, eAccess)
    gdal.Dataset.GetVirtualMemArray = WindowsVirtualMemArray
