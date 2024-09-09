"""
Functions for converting raster data to vectors, notably .shp but also .kml and 
non-geographic formats, .csv and .pkl.

Key functions
-------------

:py:func:`vectorise_from_band` This function uses GDAL to vectorise specific 
layers of the change report .geotiff.
"""

import csv
import datetime
import fiona
import geopandas as gpd
import glob
import json
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal, ogr, osr
import pandas as pd
from pathlib import Path
import re
import shutil
from skimage.morphology import closing, opening, square
import subprocess
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as et

import pyeo.coordinate_manipulation as coordinate_manipulation
import pyeo.filesystem_utilities as filesystem_utilities
from pyeo.filesystem_utilities import serial_date_to_string, move_and_rename_old_file
import pyeo.raster_manipulation as raster_manipulation

class ChangeEvent:
    """
    A class that contains all relevant data for a detected ChangeEvent.
    Credit: Ciaran Robb.

    Parameters
    ----------

    Attributes
    ----------
    geometry : ogr.wkbPolygon
        The geometry of the change event
    source_map_path : str
        The location of the class map
    source_imagery_path : str
        The location of the 8-layer raster used to generate the change map
    before_image : str
        The path to the before_image of this polygon
    after_image : str
        The path to the after image of this polygon
    before_image_url : str
        The URL of the uploaded before_image
    after_image_url : str
        The URL of the uploaded after_image
    area : int
        The area covered by this polygon
    region : str
        The region this polygon belongs to
    before_date : DateTime
        The date before which this was detected as forest
    after_date : DateTime
        The date after which this was detected as deforested
    client_id : str
        TO IMPLEMENT: The client ID that this change_event belongs to
    model_path : str
        TO IMPLEMENT, OPTIONAL: The model used to generate the change map. (this needs to be propoagated)
    id: str
        The ID of this change image.
    report_id: str
        The report_id of the report this image was generated for


    """

    def __init__(self, geometry, source_map_path, source_imagery_path, region, image_id, report_id=None):
        # Get these from function args and source map
        self.geometry = geometry
        self.source_map_path = os.path.abspath(source_map_path)
        self.source_imagery_path = os.path.abspath(source_imagery_path)
        self._extract_dates()
        self._load_projection()
        self.region = region
        self.area = geometry.GetArea()
        self.center = geometry.Centroid()
        self.id = str(image_id)
        self.report_id = report_id

        # These set after create_before_and_after_images
        self.before_image_path = None
        self.after_image_path = None
        self.geotransform = None

        # These set after upload. If image is not being uploaded, set to local storage.
        self.before_image_url = self.before_image_path
        self.after_image_url = self.after_image_path

    def _extract_dates(self):
        """Takes the source filepath and extracts the before_date and after_date dates from in, in that order."""
        # This will die if we use the temporary files we built.
        # TODO: Replace with get_change_detection_dates in pyeo.core
        date_regex = r"\d\d\d\d\d\d\d\dT\d\d\d\d\d\d"
        timestamps = re.findall(date_regex, self.source_map_path)
        date_times = [datetime.datetime.strptime(timestamp, r"%Y%m%dT%H%M%S") for timestamp in timestamps]
        timestamps.sort()
        self.before_date = date_times[0]
        self.after_date = date_times[1]

    def _load_projection(self):
        source_image = gdal.Open(self.source_map_path)
        self.projection = source_image.GetProjection()

    def create_before_and_after_images(self, figure_dir, image_size=4000):
        """Extracts two iamges with this polygon burned onto them"""
        with TemporaryDirectory() as td:
            self.before_image_path = os.path.join(figure_dir, self.id + "_before.png")
            self.after_image_path = os.path.join(figure_dir, self.id + "_after.png")
            # Reformat to file:///
            self.before_image_url = Path(self.before_image_path).as_uri()
            self.after_image_url = Path(self.after_image_path).as_uri()

            stacked_raster = gdal.Open(self.source_imagery_path)
            temp_before_path = os.path.join(td, self.id + "_before.tif")
            temp_after_path = os.path.join(td, self.id + "_after.tif")
            before_array, self.geotransform = get_polygon_subarray(self.geometry, stacked_raster, first_band=0,
                                                                   last_band=2, return_gt=True, output_size=image_size)
            after_array = get_polygon_subarray(self.geometry, stacked_raster, first_band=4, last_band=6,
                                               output_size=image_size)
            raster_manipulation.save_array_as_image(before_array, temp_before_path, self.geotransform, self.projection)
            raster_manipulation.save_array_as_image(after_array, temp_after_path, self.geotransform, self.projection)
            self._burn_polygon_outline_into_image(self.geometry, temp_before_path, [1, 2, 3], [0, 1500, 1500])
            self._burn_polygon_outline_into_image(self.geometry, temp_after_path, [1, 2, 3], [0, 1500, 1500])
            self._plot_image(temp_before_path, self.before_image_path, is_before=True)
            self._plot_image(temp_after_path, self.after_image_path, is_before=False)

    def _reformat_gdal_array_to_image(self, gdal_array, max_in=None, max_out=255):
        """Rearranges a [bands, y, x] gdal array with bgr band ordering into a [y, x, bands] image array with rgb ordering
        normalized to between 0 and max_out. If not provided, max_in is gdal_array.max()"""
        rgb_gdal_array = np.flip(gdal_array, 0)
        rgb_gdal_image = np.transpose(rgb_gdal_array, (1, 2, 0))
        if not max_in:
            max_in = rgb_gdal_image.max()
        rgb_gdal_image = (rgb_gdal_image * (max_out / max_in)).astype(np.uint8)
        return rgb_gdal_image

    def _plot_image(self, gdal_image_path, display_image_path, is_before):
        """Produces a display image with scalebar and title"""
        gdal_image = gdal.Open(gdal_image_path)
        gdal_array = gdal_image.GetVirtualMemArray().squeeze()
        display_array = self._reformat_gdal_array_to_image(gdal_array)
        extent = coordinate_manipulation.get_raster_bounds(gdal_image).GetEnvelope()
        font = {'family': 'monospace',
                'size': 4}
        matplotlib.rc('font', **font)
        plt.imshow(display_array.astype("uint8"), extent=extent)
        if is_before:
            plt.xlabel(self.before_date.strftime(r"%Y-%m-%d"))
        else:
            plt.xlabel(self.after_date.strftime(r"%Y-%m-%d"))
        #TODO: either remove or enable
        #scalebar = ScaleBar(111, 'km')
        #plt.gca().add_artist(scalebar)
        plt.grid()
        plt.tight_layout()
        plt.savefig(display_image_path, dpi=600, bbox_inches="tight")

    def _burn_polygon_outline_into_image(self, polygon, image_path, bands, values):
        """Burns polygons into image in-place."""
        with TemporaryDirectory() as td:
            self.log.info("Burning polygons into {}".format(image_path))
            image = gdal.Open(image_path, gdal.GA_Update)
            geo_path = os.path.join(td, "geo_path.shp")
            boundary = polygon.GetBoundary()
            multiline = boundary
            coordinate_manipulation.write_geometry(multiline, geo_path, srs_id=self.projection)
            gdal.Rasterize(
                image,
                geo_path,
                bands=bands,
                burnValues=values,
                allTouched=True,
            )
            return image

    '''
    # This bit is specific to Amazon web services
    # Problem here: before_date and after_date give issues as they upload to separate folders inside the bucket.
    # Is this a problem? Think about. Let's leave it for now.
    def upload_images_to_s3(self, bucket="forestsentinel-v2"):
        session = boto3.session.Session()  # Need individual session for thread safety
        s3 = session.resource("s3")
        for image_file_path, date in zip(
                [self.before_image_path, self.after_image_path],
                [self.before_date, self.after_date]):
            image_name = os.path.basename(image_file_path)
            s3_image_path = os.path.join(self.region, "images", date.strftime(r"%Y-%m-%d"), image_name)
            s3.meta.client.upload_file(image_file_path, bucket, s3_image_path)
            acl = s3.ObjectAcl(bucket, s3_image_path)
            acl.put(ACL="public-read")
            if image_file_path is self.before_image_path:
                self.before_image_url = r"https://s3.eu-central-1.amazonaws.com/{}/{}".format(bucket, s3_image_path)
            else:
                self.after_image_url = r"https://s3.eu-central-1.amazonaws.com/{}/{}".format(bucket, s3_image_path)

    def write_to_database(self, table_id="change_events"):
        properties = self.generate_property_dict()
        dynamodb = boto3.resource("dynamodb", region_name="eu-central-1")
        change_table = dynamodb.Table(table_id)
        change_table.put_item(Item=properties)
    '''

    def generate_kml_object(self):
        # Largely obsoleted by Report.ConvertToKml
        """Generates a KML Feature element for this change event"""
        poly_root = et.Element("Placemark")
        properties = self.generate_property_dict()
        description_element = et.Element("description")
        description_element.text = self.generate_html_description()
        name_element = et.Element("name")
        name_element.text = self.id
        geometry = self._transform_to_4326().ExportToKml

    def generate_json_feature(self):
        """Generates a feature in EPSG 4326"""
        properties = self.generate_property_dict()
        html_string = self.generate_html_description()
        properties.update(
            {"description": html_string,
             "tessellate": -1,
             "extrude": 0,
             "visibility": -1
             })
        geometry = json.loads(self._transform_to_4326().ExportToJson())
        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": geometry
        }
        return feature

    def _transform_to_4326(self):
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(self.projection)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)
        return transform_polygon(self.geometry, src_srs, dst_srs)

    def generate_html_description(self):
        property_dict = self.generate_property_dict()
        html = "<html>"
        for key, value in property_dict.items():
            html += ("{}: {} <br>".format(key, value))
        html += "Comparison: <br>"
        html += "<img src={} width=500><br>".format(self.before_image_url)
        html += "<img src={} width=500><br>".format(self.after_image_url)
        html += "</html>"
        return html

    def generate_property_dict(self):
        properties = {
            "before_image_url": self.before_image_url,
            "after_image_url": self.after_image_url,
            "area (m sq)": str(self.area),
            "event_id": self.id,
            "report_id": self.report_id,
            "before_date": self.before_date.strftime(r"%Y-%m-%d"),
            "after_date": self.after_date.strftime(r"%Y-%m-%d"),
            "centroid_lat": str(self.center.GetY()),
            "centroid_lon": str(self.center.GetX())
        }
        return properties

class ChangeReport:
    """A class containing a collection of ChangeEvents, all made on the same day. 
    Governs sending reports.
    The class contains functions for various processing steps.
    Modified from Ciaran Robb's version.

    Attributes
    ----------
    report_id : int
        The unique ID of this report
    json_url : string
        The location of json containing this report
    kml_url : string
        The location of the KML containing this report
    date_covered : DateTime
        The date range this report
    area_detected : float
        The total area of detected polygons inside the report in m2
    number_of_detections : int
        The number of detected polygons inside the report
    change_event_list : ChangeEvent[]
        The list of change events

    """

    def __init__(self, class_raster_path, imagery_path, class_of_interest, region, out_comparison_dir, date,
                 filter_map=None, filter_classes=None, max_changes=None, log_path=None):
        # Set from augments to constructor
        self.class_raster_path = class_raster_path
        self.imagery_path = imagery_path
        self.class_of_interest = int(class_of_interest)
        self.region = region
        self.filter_map = filter_map
        self.filter_classes = filter_classes
        self.max_changes = int(max_changes)
        self.out_comparison_dir = out_comparison_dir
        self.run_date = date
        if not log_path:
            self.log = filesystem_utilities.init_log("/dev/null")
        else:
            self.log = filesystem_utilities.init_log(log_path)

        # Set after create_change_event_list
        self.change_event_list = []
        self.number_of_detections = None
        self.area_detected = None
        self.start_date = None
        self.end_date = None
        self.max_changes_reached = False

        # set after create_before_and_after_images
        self.images_created = False

        # Key value; derived from region and run date.
        self.id = "{}-{}".format(self.region, self.run_date.strftime(r"%Y-%m-%d"))

        # Set after upload_report_to_backend
        self.json_url = None
        self.kml_url = None
        self.upload_done = False

    def create_change_event_list(self):
        self.log.debug("Creating change event list for {}".format(self.class_raster_path))
        with TemporaryDirectory() as td:
            if self.filter_map:
                self.log.debug("Filtering with classes {} from {}".format(self.filter_classes, self.filter_map))
                filtered_path = os.path.join(td, "filtered_path")
                class_raster_path = filter_by_class_map(self.filter_map, self.class_raster_path,
                                                        filtered_path, self.filter_classes)
            else:
                class_raster_path = self.class_raster_path
            shape_path = os.path.join(td, "temp_shape.shp")
            polygonize_classes(class_raster_path, shape_path, self.class_of_interest)
            polygon_list = extract_polygon_list(shape_path)
            polygon_list = sort_polygon_list_by_area(polygon_list, largest_first=True)
            self._load_polygons(polygon_list)
            self._calculate_stats()

    def _load_polygons(self, polygon_list):
        change_counter = 0
        for poly_id, polygon in enumerate(polygon_list):
            if polygon.GetBoundary().GetPointCount() <= 5:  # n points = n sides + 1 (to close the loop)
                self.log.info("Square change event detected, skipping...")
                continue
            self.change_event_list.append(ChangeEvent(  
                # This is not thread safe, so be careful if you parallelise it
                geometry=polygon,
                source_map_path=self.class_raster_path,
                source_imagery_path=self.imagery_path,
                region=self.region,
                image_id=str(self.id) + "_" + str(poly_id),
                report_id=self.id
            ))
            change_counter += 1
            self.log.debug("Change event created: {}".format(self.change_event_list[-1]))
            if self.max_changes:
                if change_counter >= self.max_changes:
                    self.log.debug("Max changes reached")
                    self.max_changes_reached = True
                    break

    def create_before_and_after_images(self, image_size=1000):
        if not self.change_event_list:
            self.log.warning("No change events generated; skipping image creation")
            return
        for i, event in enumerate(self.change_event_list):
            self.log.debug("Creating event {} of {}".format(i, len(self.change_event_list)))
            event.create_before_and_after_images(self.out_comparison_dir, image_size)
        self.images_created = True

    def upload_report_to_backend(self, event_table="change_events", report_table="reports"):
        """Creates and uploads all relevant parts of the report to the backend"""
        if not self.change_event_list:
            self.log.warning("No change events generated: skipping report upload")
            return
        if not self.images_created:
            self.log.warning("No images created: skipping report upload")
            return
        for event in self.change_event_list:
            event.upload_images_to_s3()
            event.write_to_database(event_table)
        self._upload_report()
        self.write_to_database(report_table)

    def create_local_report(self, json_file_path=None, kml_file_path=None):
        if self.change_event_list and (json_file_path or kml_file_path):
            with TemporaryDirectory() as td:
                report = self._generate_json_report()
                if not json_file_path:
                    json_file_path = os.path.join(td, "temp_json.geojson")
                with open(json_file_path, "w") as fp:
                    json.dump(report, fp)
                if kml_file_path:
                    convert_json_to_kml(json_file_path, kml_file_path)

    def _generate_json_report(self):
        """If a comparison is done, generate a geojson report"""
        if self.change_event_list:
            report = {
                "type": "FeatureCollection",
                "name": self.region,
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},  # hard coded for now
                "features": [change_event.generate_json_feature() for change_event in self.change_event_list]
            }
            return report

    '''
    def _upload_report(self, bucket="forestsentinel-v2", do_kml_copy=True):
        if self.change_event_list:
            with TemporaryDirectory() as td:
                s3 = boto3.resource('s3')
                json_file_path = os.path.join(td, "json")
                json_s3_path = "{}/jsons/{}.geojson".format(self.region, self.id)
                json_report = self._generate_json_report()
                with open(json_file_path, "w+") as fp:
                    json.dump(json_report, fp)
                s3.meta.client.upload_file(json_file_path, bucket, json_s3_path)
                acl = s3.ObjectAcl(bucket, json_s3_path)
                acl.put(ACL='public-read')
                self.json_url = r"https://s3.eu-central-1.amazonaws.com/{}/{}".format(bucket, json_s3_path)
                if do_kml_copy:
                    kml_file_path = os.path.join(td, "kml")
                    kml_s3_path = "{}/kmls/{}.kml".format(self.region, self.id)
                    convert_json_to_kml(json_file_path, kml_file_path)
                    s3.meta.client.upload_file(kml_file_path, bucket, kml_s3_path)
                    acl = s3.ObjectAcl(bucket, kml_s3_path)
                    acl.put(ACL='public-read')
                    self.kml_url = r"https://s3.eu-central-1.amazonaws.com/{}/{}".format(bucket, kml_s3_path)
    '''

    def _calculate_stats(self):
        self.area_detected = sum(change_event.area for change_event in self.change_event_list)
        self.number_of_detections = len(self.change_event_list)
        self.start_date = min(change_event.before_date for change_event in self.change_event_list)
        self.end_date = min(change_event.after_date for change_event in self.change_event_list)

    def generate_property_dict(self):
        properties = {
            "report_id": self.id,
            "region": self.region,
            "area_detected": str(self.area_detected),
            "number_of_detections": self.number_of_detections,
            "start_date": self.start_date.strftime(r"%Y-%m-%d"),
            "end_date": self.end_date.strftime(r"%Y-%m-%d"),
            "run_date": self.run_date.strftime(r"%Y-%m-%d"),
            "max_changes": str(self.max_changes),
            "max_changes_reached": self.max_changes_reached,
            "report_url": self.json_url,
            "kml_url": self.kml_url
        }
        return properties

    '''
    def write_to_database(self, table_id="reports"):  # Urgh, replicated code.
        properties = self.generate_property_dict()
        self.log.debug("Uploading following dict to talbe {}: {}".format(table_id, properties))
        dynamodb = boto3.resource("dynamodb", region_name='eu-central-1')
        change_table = dynamodb.Table(table_id)
        change_table.put_item(Item=properties)
    '''

    '''
    def send_app_notification(self):
        """Creates and sends a push notifcation to the FS app containing the url of the geoJSON of this changeEvnet"""
        if not self.images_created:
            self.log.error("No images created for this change report, aborting")
            return
        client = boto3.client('sns', region_name='eu-central-1')
        fs_app_arn = "arn:aws:sns:eu-central-1:154374898533:app/GCM/forest_sentinel_app"  # WARN: Hardcoded ARN string.
        endpoints = client.list_endpoints_by_platform_application(PlatformApplicationArn=fs_app_arn)
        for endpoint in endpoints["Endpoints"]:
            endpoint_arn = endpoint["EndpointArn"]
            try:
                client.publish(TargetArn=endpoint_arn, Message=self.json_url)
            except ClientError as e:
                if e.response["Error"]["Code"] == "EndpointDisabled":
                    log.warning("Endpoint error, clean backend of disabled endpoints")
                    pass
    '''

    def send_email_report(self, address_list):
        """Sends an email with a link to the kml file"""
        properties = self.generate_property_dict()
        message = "Report digest:\n"
        message += "Total area detected (sq km): {}\n".format(float(properties["area_detected"]) / 100)
        message += "Total number of events: {}\n".format(properties["number_of_detections"])
        message += "Start date: {}\n".format(properties["start_date"])
        message += "End date: {}\n".format(properties["end_date"])
        message += "Click below for to open the report in Google Earth:\n{}\n".format(properties["kml_url"])
        subject = "Automated deforestation report for {} between {} and {}".format(
            properties["region"], properties["start_date"], properties["end_date"])
        #TODO: write this function based on send_reports functionality
        send_email(sender, password, address_list, subject, message)



def get_polygon_subarray(polygon, raster, first_band=0, last_band=2, return_gt=False, output_size=4000):
    """Returns an explicit array of the values of subarray
    """
    with TemporaryDirectory() as td:
        temp_ras_path = os.path.join(td, "tmp.tif")
        centroid = polygon.Centroid()
        cen_x = centroid.GetX()
        cen_y = centroid.GetY()
        min_x = cen_x - output_size / 2
        max_x = cen_x + output_size / 2
        min_y = cen_y - output_size / 2
        max_y = cen_y + output_size / 2
        bounds = [min_x, min_y, max_x, max_y]
        window = gdal.Warp(temp_ras_path, raster,
                           outputBounds=bounds)
        window_array = window.GetVirtualMemArray().squeeze()  # Oh, watch out; passing array by ref here. Could get nasty. LATER:  It did!
        out_array = np.copy(window_array[first_band: last_band + 1, ...])
        if not return_gt:
            window_array = None
            window = None
            return out_array
        else:
            gt = window.GetGeoTransform()
            window_array = None
            window = None
            return out_array, gt


def transform_polygon(polygon, src_srs, dst_srs):
    transformer = osr.CoordinateTransformation(src_srs, dst_srs)
    old_points = polygon.GetBoundary().GetPoints()
    new_ring = ogr.Geometry(ogr.wkbLinearRing)
    for old_point in old_points:
        new_point = transformer.TransformPoint(*old_point)  # Using arg unpacking (* operator)
        new_ring.AddPoint(new_point[0], new_point[1])
    new_poly = ogr.Geometry(ogr.wkbPolygon)
    new_poly.AddGeometry(new_ring)
    new_poly.AssignSpatialReference(dst_srs)
    return new_poly

def filter_by_class_map(filter_map_path, class_map_path, out_map_path, classes_of_interest):
    """
    Filters class_map_path for pixels in filter_map_path containing only classes_of_interest
    Credit to Ciaran Robb.
    """
    #log.info("Filtering {} using classes{} from map {}".format(class_map_path, classes_of_interest, filter_map_path))
    with TemporaryDirectory() as td:
        filter_mask_path = os.path.join(td, "filter_mask_path")
        raster_manipulation.create_mask_from_class_map(filter_map_path, filter_mask_path, classes_of_interest,
                                                       out_resolution=10)

        #log.info("Mask created at {}, applying...".format(filter_mask_path))
        class_map = gdal.Open(class_map_path)
        class_array = class_map.GetVirtualMemArray().squeeze()
        filter_map = gdal.Open(filter_mask_path)
        filter_array = filter_map.GetVirtualMemArray().squeeze()
        out_map = raster_manipulation.create_matching_dataset(class_map, out_map_path)
        out_array = out_map.GetVirtualMemArray(eAccess=gdal.GA_Update).squeeze()
        in_bounds = coordinate_manipulation.get_raster_bounds(class_map)
        in_x_min, in_x_max, in_y_min, in_y_max = coordinate_manipulation.pixel_bounds_from_polygon(filter_map,
                                                                                                   in_bounds)
        filter_view = filter_array[in_y_min: in_y_max, in_x_min: in_x_max]
        filtered_array = raster_manipulation.apply_array_image_mask(class_array, filter_view)

        np.copyto(out_array, filtered_array)
        out_array = None
        out_map = None
        class_array = None
        class_map = None

    #log.info("Map filtered")
    return out_map_path


def convert_json_to_kml(in_path, out_path):
    cmd = ['ogr2ogr', '-f', 'kml',
           out_path, in_path]
    subprocess.run(cmd)

def isolate_class(input_image_path, mask_path, class_of_interest):
    """Creates a feature mask removing every pixel that's not of the class of interest. Might move this to Pyeo."""
    #log.info("Isolating class {} in image {}".format(class_of_interest, input_image_path))
    raster = gdal.Open(input_image_path)
    mask = raster_manipulation.create_matching_dataset(raster, mask_path, datatype=gdal.GDT_Int32)
    raster_array = raster.GetVirtualMemArray().squeeze()
    mask_array = mask.GetVirtualMemArray(eAccess=gdal.GF_Write)
    mask_array[:, :] = np.where(raster_array == class_of_interest, class_of_interest, 0)
    mask_array = None
    raster_array = None
    raster = None
    mask = None
    
    
def denoise_mask(mask_path, kernel_size):
    """Performs in-place binary erosion with a square kernel on mask. Might move to pyeo. BROKEN."""
    #log.info("Denoising mask {}".format(mask_path))
    mask = gdal.Open(mask_path, gdal.GA_Update)
    mask_array = mask.GetVirtualMemArray(eAccess=gdal.GF_Write)
    kernel = square(kernel_size) # from scikit_image.morphology (I think)
    cache = closing(image=mask_array, selem=kernel) # from scikit_image.morphology
    np.copyto(mask_array, cache)
    mask_array = None
    mask = None
    #log.info("Denoising complete")


def polygonize_classes(input_image_path, out_shape_path, class_of_interest, band=1, kernel_size=0):
    """Creates a shapefile with the polygons of the specified class"""
    with TemporaryDirectory() as td:
        #log.info("Preparing to extract class {} polygons from image {}".format(class_of_interest, input_image_path))
        mask_path = os.path.join(td, 'class_mask.tif')
        isolate_class(input_image_path, mask_path, class_of_interest)
        if kernel_size:
            denoise_mask(mask_path, kernel_size)
        mask = gdal.Open(mask_path)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(mask.GetProjection())
        mask_layer = mask.GetRasterBand(1)
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
        out_shapefile = shp_driver.CreateDataSource(out_shape_path)
        out_layer = out_shapefile.CreateLayer("deforest_polygons", srs, ogr.wkbPolygon)
        class_field = ogr.FieldDefn("class", ogr.OFTInteger)
        out_layer.CreateField(class_field)
        # Both src and mask are the mask; this is becuase the mask contains 0 (ignored) and the actual data values
        # for the region of interest. It's a bit of a hack, but it'll have to do.
        #log.info("Polygonizing {}...".format(input_image_path))
        output = gdal.Polygonize(
            srcBand=mask_layer,
            maskBand=None,
            outLayer=out_layer,
            iPixValField=0,
            callback=None)
        if output:
            #log.critical("Polygonize failure: {}".format(output))
            raise Exception("Polygonize failure {}".format(output))
        #log.info("Polygonization complete, saved at {}".format(out_shape_path))
        mask_layer = None
        mask = None
        out_layer = None
        out_shapefile = None
        return out_shape_path

def extract_polygon_list(shape_path, layer=0):
    """Extracts the geometry of every polygon in shapefile at shape_path"""
    #log.info("Extracting polygons from {}".format(shape_path))
    shapefile = ogr.Open(shape_path)
    layer = shapefile.GetLayer(layer)
    out_geom = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom.GetGeometryType() == ogr.wkbPolygon:
            out_geom.append(geom.Clone())
        geom = None
    return out_geom

def sort_polygon_list_by_area(polygons, largest_first=False):
    area_lambda = lambda polygon: polygon.GetArea()
    out = sorted(polygons, key=area_lambda, reverse=largest_first)
    return out



def band_naming(band: int,
                log: logging.Logger):
    """
    This function provides a variable name (string) based on the input integer.

    Parameters
    ----------
    band : int
        the band to interpet as a name.
        The integer format used here is starting from 1, not 0
    log : logging.Logger


    Returns
    ----------
    band_name : str

    """
    # note to self : + 1 from Python
    # fields may get shortened (laundered)

    if band == 1:
        band_name = "band1"
    elif band == 2:
        band_name = "band2"
    elif band == 3:
        band_name = "band3"
    elif band == 4:
        band_name = "band4"
    elif band == 5:
        band_name = "band5"
    elif band == 6:
        band_name = "band6"
    elif band == 7:
        band_name = "band7"
    elif band == 8:
        band_name = "band8"
    elif band == 9:
        band_name = "band9"
    elif band == 10:
        band_name = "band10"
    elif band == 11:
        band_name = "band11"
    elif band == 12:
        band_name = "band12"
    elif band == 13:
        band_name = "band13"
    elif band == 14:
        band_name = "band14"
    elif band == 15:
        band_name = "band15"
    elif band == 16:
        band_name = "band16"
    elif band == 17:
        band_name = "band17"
    elif band == 18:
        band_name = "band18"

    else:
        log.error(f"band was not an integer from 1 - 18, {band} was supplied, instead.")
        pass

    return band_name


def vectorise_from_band(
    change_report_path: str,
    band: int,
    log: logging.Logger):
    """
    This function takes the path of a change report raster and using a 
    band integer, vectorises a band layer.


    Parameters
    ----------

    change_report_path : str
        path to a change report raster
    band : int
        an integer from 1 - 18, indicating the desired band to vectorise.
        the integer corresponds to GDAL numbering, i.e. starting at 1
        instead of 0 as in Python.
    log : logging.Logger
        log variable

    Returns
    -------
    out_filename_destination : str
        the output path of the vectorised band

    """

    log.info(f"change_report_path  :  {change_report_path}")
    # let GDAL use Python to raise Exceptions, instead of printing to sys.stdout
    gdal.UseExceptions()

    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        # get Raster datasource
        src_ds = gdal.Open(change_report_path)
        log.info(f"Opening {change_report_path}")

        if src_ds is None:
            log.error(f"Unable to open {change_report_path}")

        log.info(f"Successfully opened {change_report_path}")

        try:
            src_band = src_ds.GetRasterBand(band)

            # get projection of report.tif, assign to the vector
            proj = osr.SpatialReference(src_ds.GetProjection())
        except RuntimeError as error:
            log.error(
                f"Could not open band {band}. Encountered the following error \n {error}"
            )

        # create output datasource
        dst_layername = band_naming(band, log=log)

        drv = ogr.GetDriverByName("ESRI Shapefile")
        # create the output file in a temporary directory
        out_filename = os.path.join(td, f"{change_report_path[:-4]}_{dst_layername}.shp")
        dst_ds = drv.CreateDataSource(out_filename)
        dst_layer = dst_ds.CreateLayer(dst_layername, srs=proj)

        # make number of Detections column from pixel values
        field = ogr.FieldDefn(dst_layername, ogr.OFTInteger)
        dst_layer.CreateField(field)
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(dst_layername)

        # polygonise the raster band
        log.info("Now vectorising the raster band")
        try:
            gdal.Polygonize(
                src_band,
                # src_band.GetMaskBand(),  # use .msk to only polygonise values > 0 
                #TODO: can't get gdal.Polygonize to respect .msk
                None,  # no mask
                dst_layer,
                dst_field,  # -1 for no field column
                [],
            )

        except RuntimeError as error:
            log.error(f"GDAL Polygonize failed: \n {error}")
        except Exception as error:
            log.error(f"GDAL Polygonize failed, error received : \n {error}")

        # close dst_ds and src_band
        dst_ds = None
        src_band = None
        out_filename_destination = f"{change_report_path[:-4]}_{dst_layername}.shp"
        # move the output file to the correct path
        shutil.move(out_filename, out_filename_destination)
        if dst_ds is None:
            log.info(f"Band {band} of {change_report_path} was successfully vectorised")
            log.info(f"Band {band} was written to {out_filename}")

    return out_filename_destination


def clean_zero_nodata_vectorised_band(
    vectorised_band_path: str,
    log: logging.Logger):
    """

    This function removes 0s and nodata values from the vectorised bands.

    Parameters
    ----------

    vectorised_band_path : str
        path to the band to filter
    log : logging.Logger
        The logger object

    Returns
    -------
    filename : str

    """

    log.info(f"filtering out zeroes and nodata from: {vectorised_band_path}")

    # read in shapefile
    shp = gpd.read_file(vectorised_band_path)

    # create fieldname variable
    fieldname = os.path.splitext(vectorised_band_path.split("_")[-1])[0]

    # filter out 0 and 32767 (nodata) values
    cleaned = shp.loc[(shp[fieldname] != 0) & (shp[fieldname] != 32767)]

    # copy to avoid SettingWithCopyWarning
    cleaned_copy = cleaned.copy()

    # assign explicit id from index
    cleaned_copy.loc[:, "id"] = cleaned.reset_index().index

    # save to shapefile
    filename = f"{os.path.splitext(vectorised_band_path)[0]}_filtered.shp"
    cleaned_copy.to_file(filename=filename, driver="ESRI Shapefile")

    # remove variables to reduce memory (RAM) consumption
    del (shp, cleaned, cleaned_copy)

    log.info(f"filtering complete and saved at  : {filename}")

    return filename

def boundingBoxToOffsets(bbox: list, geot: object) -> list[float]:
    """

    This function calculates offsets from the provided bounding box and geotransform.

    Parameters
    ----------
    bbox : list[float]
        bounding box coordinates within a list.
    geot : object
        Geotransform object.

    Returns
    -------
    list[float]
        List of offsets (floats) as [row1, row2, col1, col2].

    Notes
    -----
    The original implementation of this function was written by Konrad Hafen and can be found at:
    https://opensourceoptions.com/blog/zonal-statistics-algorithm-with-python-in-4-steps/

    """

    col1 = int((bbox[0] - geot[0]) / geot[1])
    col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
    row1 = int((bbox[3] - geot[3]) / geot[5])
    row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
    return [row1, row2, col1, col2]


def geotFromOffsets(row_offset, col_offset, geot):
    """

    This function calculates a new geotransform from offsets.

    Parameters
    ----------
    row_offset : int
    col_offset : int
    geot : object

    Returns
    -------
    new_geot : float

    Notes
    -----
    The original implementation of this function was written by Konrad Hafen and can be found at: https://opensourceoptions.com/blog/zonal-statistics-algorithm-with-python-in-4-steps/

    """

    new_geot = [
        geot[0] + (col_offset * geot[1]),
        geot[1],
        0.0,
        geot[3] + (row_offset * geot[5]),
        0.0,
        geot[5],
    ]
    return new_geot


def setFeatureStats(fid, min, max, mean, median, sd, sum, count, report_band):
    """

    This function sets the feature stats to calculate from the array.

    Parameters
    ----------

    fid : int
    min : int
    max : int
    mean : float
    median : float
    sd : float
    sum : int
    count : int
    report_band : int

    Returns
    -------

    featstats : dict
    """

    names = [
        f"rb{report_band}_min",
        f"rb{report_band}_max",
        f"rb{report_band}_mean",
        f"rb{report_band}_median",
        f"rb{report_band}_sd",
        f"rb{report_band}_sum",
        f"rb{report_band}_count",
        "id",
    ]

    featstats = {
        names[0]: min,
        names[1]: max,
        names[2]: mean,
        names[3]: median,
        names[4]: sd,
        names[5]: sum,
        names[6]: count,
        names[7]: fid,
    }

    return featstats


def zonal_statistics(
    raster_path: str,
    shapefile_path: str,
    report_band: int,
    log : logging.Logger
):
    """
    This function calculates zonal statistics on a raster.

    Parameters
    ----------
    raster_path : str
        the path to the raster to obtain the values from.
    shapefile_path : str
        the path to the shapefile which we will use as the "zones".
    band : int
        the band to run zonal statistics on.
    log : logging.Logger
        logger object
        
    Returns
    -------
    zstats_df : pd.DataFrame
        Returns None if no polygon statistics could be computer.

    Notes
    -----
    The raster at raster_path needs to be an even shape, e.g. 10980, 10980, 
      not 10979, 10979.

    The original implementation of this function was written by Konrad Hafen 
      and can be found at: 
      https://opensourceoptions.com/blog/zonal-statistics-algorithm-with-python-in-4-steps/

    Aspects of this function were amended to accommodate library updates from 
      GDAL, OGR and numpy.ma.MaskedArray().

    """

    # enable gdal to raise exceptions
    gdal.UseExceptions()

    mem_driver = ogr.GetDriverByName("Memory")
    mem_driver_gdal = gdal.GetDriverByName("MEM")
    shp_name = "temp"

    fn_raster = raster_path
    fn_zones = shapefile_path

    r_ds = gdal.Open(fn_raster)
    p_ds = ogr.Open(fn_zones)

    # lyr = shapefile layer
    lyr = p_ds.GetLayer()

    if lyr.GetFeatureCount() < 1:
        log.error(f"No features contained in the shapefile in zonal_stats: {fn_zones}")
        
    # get projection to apply to temporary files
    proj = lyr.GetSpatialRef()
    geot = r_ds.GetGeoTransform()
    nodata = r_ds.GetRasterBand(1).GetNoDataValue()

    zstats = []

    # p_feat = polygon feature
    p_feat = lyr.GetNextFeature()
    #niter = 0

    # while lyr.GetNextFeature() returns a polygon feature, do the following:
    while p_feat:
        try:
            # if a geometry is returned from p_feat, do the following:
            if p_feat.GetGeometryRef() is not None:
                if os.path.exists(shp_name):
                    mem_driver.DeleteDataSource(shp_name)

                # tp_ds = temporary datasource
                tp_ds = mem_driver.CreateDataSource(shp_name)
                tp_lyr = tp_ds.CreateLayer(
                    "polygons", srs=proj, geom_type=ogr.wkbPolygon
                )
                tp_lyr.CreateFeature(p_feat.Clone())
                offsets = boundingBoxToOffsets(
                    p_feat.GetGeometryRef().GetEnvelope(), geot
                )
                new_geot = geotFromOffsets(offsets[0], offsets[2], geot)

                # tr_ds = target datasource
                tr_ds = mem_driver_gdal.Create(
                    "",
                    offsets[3] - offsets[2],
                    offsets[1] - offsets[0],
                    1,
                    gdal.GDT_Byte,
                )

                tr_ds.SetGeoTransform(new_geot)
                gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
                tr_array = tr_ds.ReadAsArray()

                r_array = r_ds.GetRasterBand(report_band).ReadAsArray(
                    offsets[2],
                    offsets[0],
                    offsets[3] - offsets[2],
                    offsets[1] - offsets[0],
                )

                # get identifier for
                id = p_feat.GetFID()

                # if raster array was successfully read, do the following:
                if r_array is not None:
                    maskarray = np.ma.MaskedArray(
                        r_array,
                        mask=np.logical_or(
                            r_array == nodata, np.logical_not(tr_array)
                        ),
                    )

                    if maskarray is not None:
                        zstats.append(
                            setFeatureStats(
                                id,
                                maskarray.min(),
                                maskarray.max(),
                                maskarray.mean(),
                                np.ma.median(maskarray),
                                maskarray.std(),
                                maskarray.sum(),
                                maskarray.count(),
                                report_band=report_band,
                            )
                        )
                    else:
                        zstats.append(
                            setFeatureStats(
                                id,
                                nodata,
                                nodata,
                                nodata,
                                nodata,
                                nodata,
                                nodata,
                                nodata,
                                report_band=report_band,
                            )
                        )
                else:
                    zstats.append(
                        setFeatureStats(
                            id,
                            nodata,
                            nodata,
                            nodata,
                            nodata,
                            nodata,
                            nodata,
                            nodata,
                            report_band=report_band,
                        )
                    )

                # close temporary variables, resetting them for the next iteration
                tp_ds = None
                tp_lyr = None
                tr_ds = None

                # once there are no more features to retrieve, p_feat will 
                #   return as None, exiting the loop
                p_feat = lyr.GetNextFeature()

        except RuntimeError as error:
            print(error)

    if zstats == []:
        zstats_df = None
    else:
        fn_csv = f"{os.path.splitext(raster_path)[0]}_zstats_over_" +\
            f"{band_naming(report_band, log=log)}.csv"
        col_names = zstats[0].keys()
    
        zstats_df = pd.DataFrame(data=zstats, columns=col_names)
    
        with open(fn_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, col_names)
            writer.writeheader()
            writer.writerows(zstats)

    return zstats_df


def merge_and_calculate_spatial(
    rb_ndetections_zstats_df: pd.DataFrame,
    rb_confidence_zstats_df: pd.DataFrame,
    rb_first_changedate_zstats_df: pd.DataFrame,
    path_to_vectorised_binary_filtered: str,
    write_csv: bool,
    write_shapefile: bool,
    write_kml: bool,
    write_pkl: bool,
    change_report_path: str,
    log: logging.Logger,
    epsg: int,
    level_1_boundaries_path: str,
    tileid: str,
    delete_intermediates: bool=True,
):
    """
    This function takes the zonal statistics Pandas DataFrames and performs a table join
    to the vectorised binary polygons that are the basis of the vectorised change report.

    Parameters
    ------------

    rb_ndetections_zstats_df : pd.DataFrame()
        Pandas DataFrame object for report band 5 (ndetections)

    rb_confidence_zstats_df : pd.DataFrame()
        Pandas DataFrame object for report band 9 (confidence)

    rb_first_changedate_zstats_df : pd.DataFrame()
        Pandas DataFrame object for report band 4 (approved first change date)

    path_to_vectorised_binary : str
        Path to the vectorised binary shapefile

    write_pkl : bool (optional)
        whether to write to pkl, defaults to False

    write_csv : bool (optional)
        whether to write to csv, defaults to False

    write_shapefile : bool (optional)
        whether to write to shapefile, defaults to False

    write_kml : bool (optional)
        whether to write to kml file, defaults to False

    change_report_path : str
        the path of the original change_report tiff, used for filenaming if saving outputs
    
    log : logging.Logger
        a logging object
    
    epsg : int
        the epsg to work with, specified in `.ini`

    level_1_boundaries_path : str
        path to the administrative boundaries to filter by, specified in the `.ini`

    tileid : str
        tileid to work with

    delete_intermediates : bool
        a boolean indicating whether to delete or keep intermediate files. Defaults to True.

    Returns
    -------
    output_vector_files : list[str]
        list of output vector files created

    """

    # check if the required files exist
    if not Path(level_1_boundaries_path).is_file():
        log.error(f"File {level_1_boundaries_path} does not exist.")
        return([])
    else:
        log.info(f"Using admin boundary file: {level_1_boundaries_path}")
    if not Path(path_to_vectorised_binary_filtered).is_file():
        log.error(f"File {path_to_vectorised_binary_filtered} does not exist.")
        return([])
    else:
        log.info(f"Using forest alerts vector file: {path_to_vectorised_binary_filtered}")
    if rb_ndetections_zstats_df is None:
        log.error("Zonal statistics dataframe does not exist. Cannot vectorise.")
        return([])
    if len(rb_ndetections_zstats_df) == 0:
        log.error("Empty zonal statistics dataframe! Cannot vectorise.")
        return([])
    if rb_confidence_zstats_df is None:
        log.error("Zonal statistics dataframe does not exist. Cannot vectorise.")
        return([])
    if len(rb_confidence_zstats_df) == 0:
        log.error("Empty zonal statistics dataframe! Cannot vectorise.")
        return([])
    if rb_first_changedate_zstats_df is None:
        log.error("Zonal statistics dataframe does not exist. Cannot vectorise.")
        return([])
    if len(rb_first_changedate_zstats_df) == 0:
        log.error("Empty zonal statistics dataframe! Cannot vectorise.")
        return([])
 
    binary_dec = gpd.read_file(path_to_vectorised_binary_filtered)

    # convert first date of change detection in days, to change date
    columns_to_apply = ["rb4_min", "rb4_max", "rb4_mean", "rb4_median"]

    for column in columns_to_apply:
        rb_first_changedate_zstats_df[column] = rb_first_changedate_zstats_df[
            column
        ].apply(serial_date_to_string)

    # table join on id
    merged = binary_dec.merge(rb_ndetections_zstats_df, on="id", how="inner")
    log.info(f"1. Merged columns: {merged.columns}")
    merged2 = merged.merge(rb_confidence_zstats_df, on="id", how="inner")
    log.info(f"2. Merged2 columns: {merged2.columns}")
    merged3 = merged2.merge(rb_first_changedate_zstats_df, on="id", how="inner")
    log.info(f"3. Merged3 columns: {merged3.columns}")
    merged = merged3

    log.info("Merging Complete")
    # housekeeping, remove unused variables
    del (merged3, merged2, binary_dec)

    # add area
    merged["area_m2"] = merged.area

    # add lat long from centroid that falls within the polygon
    merged["long"] = merged.representative_point().map(lambda p: p.x)
    merged["lat"] = merged.representative_point().map(lambda p: p.y)
    log.info(f"4. Merged columns: {merged.columns}")

    # read in admin area boundaries from ini, and keep only admin area and geometry columns
    log.info(
        f"reading in administrative boundary information from {level_1_boundaries_path}"
    )
    boundaries = gpd.read_file(level_1_boundaries_path)

    #TODO: enable handing over column name when calling this function and put in ini file
    names = boundaries.columns
    for n in names:
        log.info(n)
    if "NAME_1" not in names:
        col_name = names[0] 
        log.warning("Did not find NAME_1 in column names of attribute table of admin boundary file.")
        log.warning(f"Using column {col_name} instead.")
    else:
        col_name = "NAME_1"

    boundaries = boundaries.filter([col_name, "geometry"]).rename(
        columns={col_name: "Admin_area"}
    )

    # check crs logic
    if boundaries.crs is not epsg:
        log.info(
            f"boundary epsg is : {boundaries.crs}, but merged dataframe has : {merged.crs}, reprojecting..."
        )
        boundaries = boundaries.to_crs(epsg)
        log.info(f"boundaries reprojected to {boundaries.crs}")

    # county spatial join
    merged = merged.sjoin(boundaries, predicate="within", how="left").drop(
        ["index_right"], axis=1
    )

    log.info(f"5. Merged columns: {merged.columns}")

    # add user and decision columns, for verification
    merged["tileid"] = tileid
    merged["user"] = pd.Series(dtype="string")
    merged["eventClass"] = pd.Series(dtype="string")
    merged["follow_up"] = pd.Series(dtype="string")
    merged["comments"] = pd.Series(dtype="string")

    log.info(f"6. Merged columns: {merged.columns}")

    # reorder geometry to be the last column
    columns = list(merged.columns)
    columns.remove("geometry")
    columns.append("geometry")
    merged = merged.reindex(columns=columns)

    log.info(f"7. Merged columns: {merged.columns}")

    shp_fname = f"{os.path.splitext(change_report_path)[0]}.shp"
    kml_fname = f"{os.path.splitext(change_report_path)[0]}.kml"
    csv_fname = f"{os.path.splitext(change_report_path)[0]}.csv"
    pkl_fname = f"{os.path.splitext(change_report_path)[0]}.pkl"
    
    output_vector_files = []

    if write_shapefile:
        merged.to_file(shp_fname)
        log.info(f"Shapefile written as ESRI Shapefile, to:  {shp_fname}")
        output_vector_files.append(shp_fname)
        
    if write_kml:
        fiona.supported_drivers['KML'] = 'rw'
        merged.to_file(kml_fname, driver='KML')
        log.info(f"Vector file written as kml file, to:  {kml_fname}")
        output_vector_files.append(kml_fname)

    if write_pkl:
        merged.to_pickle(pkl_fname)
        log.info(f"GeoDataFrame written as pickle, to:  {pkl_fname}")

    if write_csv:
        merged.to_csv(csv_fname)
        log.info(f"DataFrame written as csv, to:   {csv_fname}")

    if delete_intermediates:
        try:
            log.info("Deleting intermediate change report vectorisation files")
            directory = os.path.dirname(change_report_path)
            binary_dec_pattern = f"{directory}{os.sep}*band*"
            zstats_pattern = f"{directory}{os.sep}*zstats*"
            intermediate_files = glob.glob(binary_dec_pattern)
            zstat_files = glob.glob(zstats_pattern)
            intermediate_files.extend(zstat_files)
            for file in intermediate_files:
                os.remove(file)
        except:
            log.info("Could not delete intermediate files")
    
    return(list(output_vector_files))
