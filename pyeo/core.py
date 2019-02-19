import pdb

import os
import sys
import logging
import datetime as dt
import glob
import re
import configparser
from sentinelhub import download_safe_format
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson
import subprocess
import gdal
from osgeo import ogr, osr
import numpy as np
import numpy.ma as ma
import tempfile
from tempfile import TemporaryDirectory
import sklearn.ensemble as ens
from sklearn.model_selection import cross_val_score
from skimage import morphology as morph
import scipy.sparse as sp
import joblib
import shutil
import zipfile

import json
import csv
import requests


try:
    from google.cloud import storage
    from google.cloud.exceptions import ServiceUnavailable
except ModuleNotFoundError:
    print("google-cloud-storage required for Google downloads. Try pip install google-cloud-storage")

try:
    import tenacity
    from planet import api as planet_api
    from multiprocessing.dummy import Pool
except ModuleNotFoundError:
    print("Tenacity, Planet and Multiprocessing are required for Planet data downloading")


class ForestSentinelException(Exception):
    pass


class StackImagesException(ForestSentinelException):
    pass


class CreateNewStacksException(ForestSentinelException):
    pass


class StackImageException(ForestSentinelException):
    pass


class BadS2Exception(ForestSentinelException):
    pass


class BadGoogleURLExceeption(ForestSentinelException):
    pass


class BadDataSourceExpection(ForestSentinelException):
    pass


class FMaskException(ForestSentinelException):
    pass


def sent2_query(user, passwd, geojsonfile, start_date, end_date, cloud=50):
    """


    From Geospatial Learn by Ciaran Robb, embedded here for portability.

    Produces a dict of sentinel-2 IDs and

    Notes
    -----------


    Parameters
    -----------

    user : string
           username for esa hub

    passwd : string
             password for hub

    geojsonfile : string
                  AOI polygon of interest in EPSG 4326

    start_date : string
                 date of beginning of search

    end_date : string
               date of end of search

    cloud : string (optional)
            include a cloud filter in the search


    """
    ##set up your copernicus username and password details, and copernicus download site... BE CAREFUL if you share this script with others though!
    log = logging.getLogger(__name__)
    api = SentinelAPI(user, passwd)
    footprint = geojson_to_wkt(read_geojson(geojsonfile))
    log.info("Sending query:\nfootprint: {}\nstart_date: {}\nend_date: {}\n cloud_cover: {} ".format(
        footprint, start_date, end_date, cloud))
    products = api.query(footprint,
                         date=(start_date, end_date), platformname="Sentinel-2",
                         cloudcoverpercentage="[0 TO {}]".format(cloud))
    return products


def init_log(log_path):
    """Sets up the log format and log handlers; one for stdout and to write to a file, 'log_path'.
     Returns the log for the calling script"""
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.info("****PROCESSING START****")
    return log


def create_file_structure(root):
    """Creates the file structure if it doesn't exist already"""
    os.chdir(root)
    dirs = [
        "images/",
        "images/L1/",
        "images/L2/",
        "images/merged/",
        "images/stacked/",
        "images/planet/",
        "composite/",
        "composite/L1",
        "composite/L2",
        "composite/merged",
        "output/",
        "output/categories",
        "output/probabilities",
        "output/report_image",
        "output/display_images",
        "log/"
    ]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass


def read_aoi(aoi_path):
    """Opens the geojson file for the aoi. If FeatureCollection, return the first feature."""
    with open(aoi_path ,'r') as aoi_fp:
        aoi_dict = json.load(aoi_fp)
        if aoi_dict["type"] == "FeatureCollection":
            aoi_dict = aoi_dict["features"][0]
        return aoi_dict


def check_for_new_s2_data(aoi_path, aoi_image_dir, conf):
    """Checks the S2 API for new data; if it's there, return the result"""
    # TODO: This isn't breaking properly on existing imagery
    # TODO: In fact, just clean this up completely, it's a bloody mess.
    # set up API for query
    log = logging.getLogger(__name__)
    user = conf['sent_2']['user']
    password = conf['sent_2']['pass']
    # Get last downloaded map date
    file_list = os.listdir(aoi_image_dir)
    datetime_regex = r"\d{8}T\d{6}"     # Regex that matches an S2 timestamp
    date_matches = re.finditer(datetime_regex, file_list.__str__())
    try:
        dates = [dt.datetime.strptime(date.group(0), '%Y%m%dT%H%M%S') for date in date_matches]
        last_date = max(dates)
        # Do the query
        result = sent2_query(user, password, aoi_path,
                             last_date.isoformat(timespec='seconds')+'Z',
                             dt.datetime.today().isoformat(timespec='seconds')+'Z')
        return result
    except ValueError:
        log.error("aoi_image_dir empty, please add a starting image")
        sys.exit(1)


def check_for_s2_data_by_date(aoi_path, start_date, end_date, conf, cloud_cover=50):
    log = logging.getLogger(__name__)
    log.info("Querying for imagery between {} and {} for aoi {}".format(start_date, end_date, aoi_path))
    user = conf['sent_2']['user']
    password = conf['sent_2']['pass']
    start_timestamp = dt.datetime.strptime(start_date, '%Y%m%d').isoformat(timespec='seconds')+'Z'
    end_timestamp = dt.datetime.strptime(end_date, '%Y%m%d').isoformat(timespec='seconds')+'Z'
    result = sent2_query(user, password, aoi_path, start_timestamp, end_timestamp, cloud=cloud_cover)
    log.info("Search returned {} images".format(len(result)))
    return result


def download_s2_data(new_data, out_folder, l2_dir=None, source='scihub', user=None, passwd=None):
    """Downloads S2 imagery from AWS, google_cloud or scihub. new_data is a dict from Sentinel_2. If l2_dir is given,
    will check that directory for existing imagery and skip if exists."""
    log = logging.getLogger(__name__)
    for image_uuid in new_data:
        l1_path = os.path.join(out_folder, new_data[image_uuid]['identifier'])
        if check_for_invalid_l1_data(l1_path) == 1:
            log.info("L1 imagery exists, skipping download")
            continue
        if l2_dir:
            l2_path = os.path.join(l2_dir, new_data[image_uuid]['identifier'].replace("MSIL1C", "MSIL2A")+".SAFE")
            log.info("Checking {} for existing L2 imagery".format(l2_path))
            if os.path.isdir(l2_path):
                log.info("L2 imagery exists, skipping download.")
                continue
        log.info("Downloading {} from {}".format(new_data[image_uuid]['identifier'], source))
        if source=='aws':
            download_safe_format(product_id=new_data[image_uuid]['identifier'], folder=out_folder)
        elif source=='google':
            download_from_google_cloud([new_data[image_uuid]['identifier']], out_folder=out_folder)
        elif source=="scihub":
            download_from_scihub(image_uuid, out_folder, user, passwd)
        else:
            log.error("Invalid data source; valid values are 'aws', 'google' and 'scihub'")
            raise BadDataSourceExpection


def download_from_scihub(product_uuid, out_folder, user, passwd):
    """Downloads and unzips product_uuid from scihub"""
    log = logging.getLogger(__name__)
    api = SentinelAPI(user, passwd)
    log.info("Downloading {} from scihub".format(product_uuid))
    prod = api.download(product_uuid, out_folder)
    if not prod:
        log.error("{} failed to download".format(product_uuid))
    zip_path = os.path.join(out_folder, prod['title']+".zip")
    log.info("Unzipping {} to {}".format(zip_path, out_folder))
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(out_folder)
    zip_ref.close()
    log.info("Removing {}".format(zip_path))
    os.remove(zip_path)


@tenacity.retry(
    wait=tenacity.wait_exponential(),
    stop=tenacity.stop_after_delay(10000),
    retry=tenacity.retry_if_exception_type(ServiceUnavailable)
)
def download_from_google_cloud(product_ids, out_folder, redownload = False):
    """Passed a list of S2 product ids , downloads them into out_for"""
    log = logging.getLogger(__name__)
    log.info("Downloading following products from Google Cloud:".format(product_ids))
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("gcp-public-data-sentinel-2")
    for safe_id in product_ids:
        if not safe_id.endswith(".SAFE"):
            safe_id = safe_id+".SAFE"
        if check_for_invalid_l1_data(os.path.join(out_folder, safe_id)) and not redownload:
            log.info("File exists, skipping.")
            return
        if redownload:
            log.info("Removing {}".format(os.path.join(out_folder, safe_id)))
            shutil.rmtree(os.path.join(out_folder, safe_id))
        tile_id = get_sen_2_image_tile(safe_id)
        utm_zone = tile_id[1:3]
        lat_band = tile_id[3]
        grid_square = tile_id[4:6]
        object_prefix = r"tiles/{}/{}/{}/{}/".format(
            utm_zone, lat_band, grid_square, safe_id
        )
        object_iter = bucket.list_blobs(prefix=object_prefix, delimiter=None)
        if object_iter.num_results == 0:
            log.error("{} missing from Google Cloud, continuing".format(safe_id))
            continue
        for s2_object in object_iter:
            download_blob_from_google(bucket, object_prefix, out_folder, s2_object)
        # Need to make these two empty folders for sen2cor to work properly
        try:
            os.mkdir(os.path.join(os.path.abspath(out_folder), safe_id, "AUX_DATA"))
            os.mkdir(os.path.join(os.path.abspath(out_folder), safe_id, "HTML"))
        except FileExistsError:
            pass


@tenacity.retry(
    wait=tenacity.wait_exponential(),
    stop=tenacity.stop_after_delay(10000),
    retry=tenacity.retry_if_exception_type(ServiceUnavailable)
)
def download_blob_from_google(bucket, object_prefix, out_folder, s2_object):

    blob = bucket.get_blob(s2_object.name)
    object_out_path = os.path.join(
        os.path.abspath(out_folder),
        s2_object.name.replace(os.path.dirname(object_prefix.rstrip('/')), "").strip('/')
    )
    os.makedirs(os.path.dirname(object_out_path), exist_ok=True)
    log.info("Downloading from {} to {}".format(s2_object, object_out_path))
    with open(object_out_path, 'w+b') as f:
        blob.download_to_file(f)



def load_api_key(path_to_api):
    """Returns an API key from a single-line text file containing that API"""
    with open(path_to_api, 'r') as api_file:
        return api_file.read()


def get_planet_product_path(planet_dir, product):
    """Returns the path to a Planet product within a Planet directory"""
    planet_folder = os.path.dirname(planet_dir)
    product_file = glob.glob(planet_folder + '*' + product)
    return os.path.join(planet_dir, product_file)


def download_planet_image_on_day(aoi_path, date, out_path, api_key, item_type="PSScene4Band", search_name="auto",
                 asset_type="analytic", threads=5):
    """Queries and downloads all images on the date in the aoi given"""
    log = logging.getLogger(__name__)
    start_time = date + "T00:00:00.000Z"
    end_time = date + "T23:59:59.000Z"
    try:
        planet_query(aoi_path, start_time, end_time, out_path, api_key, item_type, search_name, asset_type, threads)
    except IndexError:
        log.warning("IndexError exception; likely no imagery available for chosen date")


def planet_query(aoi_path, start_date, end_date, out_path, api_key, item_type="PSScene4Band", search_name="auto",
                 asset_type="analytic", threads=5):
    """
    Downloads data from Planetlabs for a given time period in the given AOI

    Parameters
    ----------
    aoi : str
        Filepath of a single-polygon geojson containing the aoi

    start_date : str
        the inclusive start of the time window in UTC format

    end_date : str
        the inclusive end of the time window in UTC format

    out_path : filepath-like object
        A path to the output folder
        Any identically-named imagery will be overwritten

    item_type : str
        Image type to download (see Planet API docs)

    search_name : str
        A name to refer to the search (required for large searches)

    asset_type : str
        Planet asset type to download (see Planet API docs)

    threads : int
        The number of downloads to perform concurrently

    Notes
    -----
    IMPORTANT: Will not run for searches returning greater than 250 items.

    """
    feature = read_aoi(aoi_path)
    aoi = feature['geometry']
    session = requests.Session()
    session.auth = (api_key, '')
    search_request = build_search_request(aoi, start_date, end_date, item_type, search_name)
    search_result = do_quick_search(session, search_request)

    thread_pool = Pool(threads)
    threaded_dl = lambda item: activate_and_dl_planet_item(session, item, asset_type, out_path)
    thread_pool.map(threaded_dl, search_result)


def build_search_request(aoi, start_date, end_date, item_type, search_name):
    """Builds a search request for the planet API"""
    date_filter = planet_api.filters.date_range("acquired", gte=start_date, lte=end_date)
    aoi_filter = planet_api.filters.geom_filter(aoi)
    query = planet_api.filters.and_filter(date_filter, aoi_filter)
    search_request = planet_api.filters.build_search_request(query, [item_type])
    search_request.update({'name': search_name})
    return search_request


def do_quick_search(session, search_request):
    """Tries the quick search; returns a dict of features"""
    search_url = "https://api.planet.com/data/v1/quick-search"
    search_request.pop("name")
    print("Sending quick search")
    search_result = session.post(search_url, json=search_request)
    if search_result.status_code >= 400:
        raise requests.ConnectionError
    return search_result.json()["features"]


def do_saved_search(session, search_request):
    """Does a saved search; this doesn't seem to work yet."""
    search_url = "https://api.planet.com/data/v1/searches/"
    search_response = session.post(search_url, json=search_request)
    search_id = search_response.json()['id']
    if search_response.json()['_links'].get('_next_url'):
        return get_paginated_items(session)
    else:
        search_url = "https://api-planet.com/data/v1/searches/{}/results".format(search_id)
        response = session.get(search_url)
        items = response.content.json()["features"]
        return items


def get_paginated_items(session, search_id):
    """Let's leave this out for now."""
    raise Exception("pagination not handled yet")


class TooManyRequests(requests.RequestException):
    """Too many requests; do exponential backoff"""


@tenacity.retry(
    wait=tenacity.wait_exponential(),
    stop=tenacity.stop_after_delay(10000),
    retry=tenacity.retry_if_exception_type(TooManyRequests)
)
def activate_and_dl_planet_item(session, item, asset_type, file_path):
    """Activates and downloads a single planet item"""
    log = logging.getLogger(__name__)
    #  TODO: Implement more robust error handling here (not just 429)
    item_id = item["id"]
    item_type = item["properties"]["item_type"]
    item_url = "https://api.planet.com/data/v1/"+ \
        "item-types/{}/items/{}/assets/".format(item_type, item_id)
    item_response = session.get(item_url)
    log.info("Activating " + item_id)
    activate_response = session.post(item_response.json()[asset_type]["_links"]["activate"])
    while True:
        status = session.get(item_url)
        if status.status_code == 429:
            log.warning("ID {} too fast; backing off".format(item_id))
            raise TooManyRequests
        if status.json()[asset_type]["status"] == "active":
            break
    dl_link = status.json()[asset_type]["location"]
    item_fp = os.path.join(file_path, item_id + ".tif")
    log.info("Downloading item {} from {} to {}".format(item_id, dl_link, item_fp))
    # TODO Do we want the metadata in a separate file as well as embedded in the geotiff?
    with open(item_fp, 'wb+') as fp:
        image_response = session.get(dl_link)
        if image_response.status_code == 429:
            raise TooManyRequests
        fp.write(image_response.content)    # Don't like this; it might store the image twice. Check.
        log.info("Item {} download complete".format(item_id))


def apply_sen2cor(image_path, sen2cor_path, delete_unprocessed_image=False):
    """Applies sen2cor to the SAFE file at image_path. Returns the path to the new product."""
    # Here be OS magic. Since sen2cor runs in its own process, Python has to spin around and wait
    # for it; since it's doing that, it may as well be logging the output from sen2cor. This
    # approach can be multithreaded in future to process multiple image (1 per core) but that
    # will take some work to make sure they all finish before the program moves on.
    log = logging.getLogger(__name__)
    # added sen2cor_path by hb91
    log.info("calling subprocess: {}".format([sen2cor_path, image_path]))
    sen2cor_proc = subprocess.Popen([sen2cor_path, image_path],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)

    while True:
        nextline = sen2cor_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == '' and sen2cor_proc.poll() is not None:
            break
        if "CRITICAL" in nextline:
            #log.error(nextline)
            raise subprocess.CalledProcessError(-1, "L2A_Process")

    log.info("sen2cor processing finished for {}".format(image_path))
    log.info("Validating:")
    if not check_for_invalid_l2_data(image_path.replace("MSIL1C", "MSIL2A")):
        log.error("10m imagery not present in {}".format(image_path.replace("MSIL1C", "MSIL2A")))
        raise BadS2Exception
    if delete_unprocessed_image:
            log.info("removing {}".format(image_path))
            shutil.rmtree(image_path)
    return image_path.replace("MSIL1C", "MSIL2A")


def apply_fmask(in_safe_dir, out_file, fmask_command="fmask_sentinel2Stacked.py"):
    """Calls fmask to create a new mask for L1 data"""
    # For reasons known only to the spirits, calling subprocess.run from within this function on a HPC cause the PATH
    # to be prepended with a Windows "eoenv\Library\bin;" that breaks the environment. What follows is a large kludge.
    if "torque" in os.getenv("PATH"):  # Are we on a HPC? If so, give explicit path to fmask
        fmask_command = "/data/clcr/shared/miniconda3/envs/eoenv/bin/fmask_sentinel2Stacked.py"
    log = logging.getLogger(__name__)
    args = [
        fmask_command,
        "-o", out_file,
        "--safedir", in_safe_dir
    ]
    log.info("Creating fmask from {}, output at {}".format(in_safe_dir, out_file))
    # pdb.set_trace()
    fmask_proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    while True:
        nextline = fmask_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == '' and fmask_proc.poll() is not None:
            break


def atmospheric_correction(in_directory, out_directory, sen2cor_path, delete_unprocessed_image=False):
    """Applies Sen2cor cloud correction to level 1C images"""
    log = logging.getLogger(__name__)
    images = [image for image in os.listdir(in_directory)
              if image.startswith('MSIL1C', 4)]
    # Opportunity for multithreading here
    for image in images:
        log.info("Atmospheric correction of {}".format(image))
        image_path = os.path.join(in_directory, image)
        #image_timestamp = get_sen_2_image_timestamp(image)
        if glob.glob(os.path.join(out_directory, image.replace("MSIL1C", "MSIL2A"))):
            log.warning("{} exists. Skipping.".format(image.replace("MSIL1C", "MSIL2A")))
            continue
        try:
            l2_path = apply_sen2cor(image_path, sen2cor_path, delete_unprocessed_image=delete_unprocessed_image)
        except (subprocess.CalledProcessError, BadS2Exception):
            log.error("Atmospheric correction failed for {}. Moving on to next image.".format(image))
            pass
        else:
            l2_name = os.path.basename(l2_path)
            log.info("L2  path: {}".format(l2_path))
            log.info("New path: {}".format(os.path.join(out_directory, l2_name)))
            os.rename(l2_path, os.path.join(out_directory, l2_name))


def check_for_invalid_l2_data(l2_SAFE_file, resolution="10m"):
    """Checks the existance of the specified resolution of imagery. Returns a True-value with a warning if passed
    an invalid SAFE directory; this will prevent disconnected files from being deleted.
    Retuns 1 if imagery is valid, 0 if not and 2 if not a safe-file"""
    log = logging.getLogger(__name__)
    if not l2_SAFE_file.endswith(".SAFE") or "L2A" not in l2_SAFE_file:
        log.info("{} is not a valid L2 file")
        return 2
    log.info("Checking {} for incomplete {} imagery".format(l2_SAFE_file, resolution))
    granule_path = r"GRANULE/*/IMG_DATA/R{}/*_B0[8,4,3,2]_*.jp2".format(resolution)
    image_glob = os.path.join(l2_SAFE_file, granule_path)
    if glob.glob(image_glob):
        return 1
    else:
        return 0


def check_for_invalid_l1_data(l1_SAFE_file):
    """Checks the existance of the specified resolution of imagery. Returns True with a warning if passed
    an invalid SAFE directory; this will prevent disconnected files from being deleted.
    Retuns 1 if imagery is valid, 0 if not and 2 if not a safe-file"""
    log = logging.getLogger(__name__)
    if not l1_SAFE_file.endswith(".SAFE") or "L1C" not in l1_SAFE_file:
        log.info("{} is not a valid L1 file")
        return 2
    log.info("Checking {} for incomplete imagery".format(l1_SAFE_file))
    granule_path = r"GRANULE/*/IMG_DATA/*_B0[8,4,3,2]_*.jp2"
    image_glob = os.path.join(l1_SAFE_file, granule_path)
    if glob.glob(image_glob):
        return 1
    else:
        return 0


def clean_l2_data(l2_SAFE_file, resolution="10m", warning=True):
    """Removes any directories that don't have band 2, 3, 4 or 8 in the specified resolution folder
    If warning=True, prompts first."""
    log = logging.getLogger(__name__)
    is_valid = check_for_invalid_l2_data(l2_SAFE_file, resolution)
    if not is_valid:
        if warning:
            if not input("About to delete {}: Y/N?".format(l2_SAFE_file)).upper().startswith("Y"):
                return
        log.warning("Removing {}".format(l2_SAFE_file))
        shutil.rmtree(l2_SAFE_file)


def clean_l2_dir(l2_dir, resolution="10m", warning=True):
    """Calls clean_l2_data on every SAFE file in l2_dir"""
    log = logging.getLogger(__name__)
    log.info("Scanning {} for incomplete SAFE files".format(l2_dir))
    for safe_file_path in [os.path.join(l2_dir, safe_file_name) for safe_file_name in os.listdir(l2_dir)]:
        clean_l2_data(safe_file_path, resolution, warning)


def clean_aoi(aoi_dir, images_to_keep = 4, warning=True):
    """Removes all but the last images_to_keep newest images in the L1, L2, merged, stacked and
    composite directories. Will not affect the output folder."""
    l1_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "images/L1")), recent_first=True)
    l2_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "images/L2")), recent_first=True)
    comp_l1_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "composite/L2")), recent_first=True)
    comp_l2_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "composite/L2")), recent_first=True)
    merged_list = sort_by_timestamp(
        [image for image in os.listdir(os.path.join(aoi_dir, "images/merged")) if image.endswith(".tif")],
        recent_first=True)
    stacked_list = sort_by_timestamp(
        [image for image in os.listdir(os.path.join(aoi_dir, "images/stacked")) if image.endswith(".tif")],
        recent_first=True)
    comp_merged_list = sort_by_timestamp(
        [image for image in os.listdir(os.path.join(aoi_dir, "composite/merged")) if image.endswith(".tif")],
        recent_first=True)
    for image_list in (l1_list, l2_list, comp_l1_list, comp_l2_list):
        for safe_file in image_list[images_to_keep:]:
            os.rmdir(safe_file)
    for image_list in (merged_list, stacked_list, comp_merged_list):
        for image in image_list[images_to_keep:]:
            os.remove(image)
            os.remove(image.rsplit('.')(0)+".msk")


def create_matching_dataset(in_dataset, out_path,
                            format="GTiff", bands=1, datatype = None):
    """Creates an empty gdal dataset with the same dimensions, projection and geotransform. Defaults to 1 band.
    Datatype is set from the first layer of in_dataset if unspecified"""
    driver = gdal.GetDriverByName(format)
    if datatype is None:
        datatype = in_dataset.GetRasterBand(1).DataType
    out_dataset = driver.Create(out_path,
                                xsize=in_dataset.RasterXSize,
                                ysize=in_dataset.RasterYSize,
                                bands=bands,
                                eType=datatype)
    out_dataset.SetGeoTransform(in_dataset.GetGeoTransform())
    out_dataset.SetProjection(in_dataset.GetProjection())
    return out_dataset


def create_new_stacks(image_dir, stack_dir):
    """
    Creates new stacks with with adjacent image acquisition dates. Threshold; how small a part
    of the latest_image will be before it's considered to be fully processed.
    New_image_name must exist inside image_dir.

    Step 1: Sort directory as follows:
            Relative Orbit number (RO4O), then Tile Number (T15PXT), then
            Datatake sensing start date (YYYYMMDD) and time(THHMMSS).
            newest first.
    Step 2: For each tile number:
            new_data_polygon = bounds(new_image_name)
    Step 3: For each tiff image coverring that tile, work backwards in time:
            a. Check if it intersects new_data_polygon
            b. If it does
               - add to a to_be_stacked list,
               - subtract it's bounding box from new_data_polygon.
            c. If new_data_polygon drops having a total area less than threshold, stop.
    Step 4: Stack new rasters for each tile in new_data list.
    """
    log = logging.getLogger(__name__)
    new_images = []
    tiles = get_sen_2_tiles(image_dir)
    tiles = list(set(tiles)) # eliminate duplicates
    n_tiles = len(tiles)
    log.info("Found {} unique tile IDs for stacking:".format(n_tiles))
    for tile in tiles:
        log.info("   {}".format(tile))  # Why in its own loop?
    for tile in tiles:
        log.info("Tile ID for stacking: {}".format(tile))
        safe_files = glob.glob(os.path.join(image_dir, "*" + tile + "*.tif")) # choose all files with that tile ID
        if len(safe_files) == 0:
            raise CreateNewStacksException("Image_dir is empty: {}".format(os.path.join(image_dir, tile + "*.tif")))
        else:
            safe_files = sort_by_timestamp(safe_files)
            log.info("Image file list for pairwise stacking for this tile:")
            for file in safe_files:
                log.info("   {}".format(file))
            latest_image_path = safe_files[0]
            for image in safe_files[1:]:
                new_images.append(stack_old_and_new_images(image, latest_image_path, stack_dir))
                latest_image_path = image
    return new_images


def get_sen_2_tiles(image_dir):
    """
    gets the list of tiles present in the directory
    """
    image_files = glob.glob(os.path.join(image_dir, "*.tif"))
    if len(image_files) == 0:
        raise CreateNewStacksException("Image_dir is empty")
    else:
        tiles = []
        for image_file in image_files:
            tile = get_sen_2_image_tile(image_file)
            tiles.append(tile)
    return tiles


def sort_by_timestamp(strings, recent_first=True):
    """Takes a list of strings that contain sen2 timestamps and returns them sorted, most recent first. Does not
    guarantee ordering of strings with the same timestamp. Removes any string that does not contain a timestamp"""
    strings = list(filter(get_image_acquisition_time, strings))
    strings.sort(key=lambda x: get_image_acquisition_time(x), reverse=recent_first)
    return strings


def get_image_acquisition_time(image_name):
    """Gets the datetime object from a .safe filename of a planet image. No test. Returns None if no timestamp present"""
    try:
        return dt.datetime.strptime(get_sen_2_image_timestamp(image_name), '%Y%m%dT%H%M%S')
    except AttributeError:
        return None




def get_preceding_image_path(target_image_name, search_dir):
    """Gets the path to the image in search_dir preceding the image called image_name"""
    target_time = get_image_acquisition_time(target_image_name)
    image_paths = sort_by_timestamp(os.listdir(search_dir), recent_first=True) # Sort image list newest first
    for image_path in image_paths:   # Walk through newest to oldest
        accq_time = get_image_acquisition_time(image_path)   # Get this image time
        if accq_time < target_time:   # If this image is older than the target image, return it.
            return os.path.join(search_dir, image_path)
    raise FileNotFoundError("No image older than {}".format(target_image_name))



def open_dataset_from_safe(safe_file_path, band, resolution = "10m"):
    """Opens a dataset given a safe file. Give band as a string."""
    image_glob = r"GRANULE/*/IMG_DATA/R{}/*_{}_{}.jp2".format(resolution, band, resolution)
    # edited by hb91
    #image_glob = r"GRANULE/*/IMG_DATA/*_{}.jp2".format(band)
    fp_glob = os.path.join(safe_file_path, image_glob)
    image_file_path = glob.glob(fp_glob)
    out = gdal.Open(image_file_path[0])
    return out


def preprocess_sen2_images(l2_dir, out_dir, l1_dir, cloud_threshold=60, buffer_size=0, epsg=None):
    """For every .SAFE folder in in_dir, stacks band 2,3,4 and 8  bands into a single geotif, creates a cloudmask from
    the combined fmask and sen2cor cloudmasks and reprojects to a given EPSG if provided"""
    log = logging.getLogger(__name__)
    safe_file_path_list = [os.path.join(l2_dir, safe_file_path) for safe_file_path in os.listdir(l2_dir)]
    for l2_safe_file in safe_file_path_list:
        with TemporaryDirectory() as temp_dir:
            log.info("----------------------------------------------------")
            log.info("Merging 10m bands in SAFE dir: {}".format(l2_safe_file))
            temp_path = os.path.join(temp_dir, get_sen_2_granule_id(l2_safe_file)) + ".tif"
            log.info("Output file: {}".format(temp_path))
            stack_sentinel_2_bands(l2_safe_file, temp_path, band='10m')

            #pdb.set_trace()

            log.info("Creating cloudmask for {}".format(temp_path))
            l1_safe_file = get_l1_safe_file(l2_safe_file, l1_dir)
            mask_path = get_mask_path(temp_path)
            create_mask_from_sen2cor_and_fmask(l1_safe_file, l2_safe_file, mask_path, buffer_size=buffer_size)
            log.info("Cloudmask created")

            out_path = os.path.join(out_dir, os.path.basename(temp_path))
            out_mask_path = os.path.join(out_dir, os.path.basename(mask_path))

            if epsg:
                log.info("Reprojecting images to {}".format(epsg))
                proj = osr.SpatialReference()
                proj.ImportFromEPSG(epsg)
                wkt = proj.ExportToWkt()
                reproject_image(temp_path, out_path, wkt)
                reproject_image(mask_path, out_mask_path, wkt)
            else:
                log.info("Moving images to {}".format(out_dir))
                shutil.move(temp_path, out_path)
                shutil.move(mask_path, out_mask_path)


def stack_sentinel_2_bands(safe_dir, out_image_path, band = "10m"):
    """Stacks the contents of a .SAFE granule directory into a single geotiff"""
    log = logging.getLogger(__name__)
    granule_path = r"GRANULE/*/IMG_DATA/R{}/*_B0[8,4,3,2]_{}.jp2".format(band, band)
    image_glob = os.path.join(safe_dir, granule_path)
    file_list = glob.glob(image_glob)
    file_list.sort()   # Sorting alphabetically gives the right order for bands
    if not file_list:
        log.error("No 10m imagery present in {}".format(safe_dir))
        raise BadS2Exception
    stack_images(file_list, out_image_path, geometry_mode="intersect")
    return out_image_path


def stack_old_and_new_images(old_image_path, new_image_path, out_dir, create_combined_mask=True):
    """
    Stacks two images with the same tile
    Names the result with the two timestamps.
    First, decompose the granule ID into its components:
    e.g. S2A, MSIL2A, 20180301, T162211, N0206, R040, T15PXT, 20180301, T194348
    are the mission ID(S2A/S2B), product level(L2A), datatake sensing start date (YYYYMMDD) and time(THHMMSS),
    the Processing Baseline number (N0206), Relative Orbit number (RO4O), Tile Number field (T15PXT),
    followed by processing run date and then time
    """
    log = logging.getLogger(__name__)
    tile_old = get_sen_2_image_tile(old_image_path)
    tile_new = get_sen_2_image_tile(new_image_path)
    if (tile_old == tile_new):
        log.info("Stacking {} and".format(old_image_path))
        log.info("         {}".format(new_image_path))
        old_timestamp = get_sen_2_image_timestamp(os.path.basename(old_image_path))
        new_timestamp = get_sen_2_image_timestamp(os.path.basename(new_image_path))
        out_path = os.path.join(out_dir, tile_new + '_' + old_timestamp + '_' + new_timestamp)
        log.info("Output stacked file: {}".format(out_path + ".tif"))
        stack_images([old_image_path, new_image_path], out_path + ".tif")
        if create_combined_mask:
            out_mask_path = out_path + ".msk"
            old_mask_path = get_mask_path(old_image_path)
            new_mask_path = get_mask_path(new_image_path)
            combine_masks([old_mask_path, new_mask_path], out_mask_path, combination_func="and", geometry_func="intersect")
        return out_path + ".tif"
    else:
        log.error("Tiles  of the two images do not match. Aborted.")


def stack_image_with_composite(image_path, composite_path, out_dir, create_combined_mask=True, skip_if_exists=True):
    """Stacks an image with a cloud-free composite"""
    log = logging.getLogger(__name__)
    log.info("Stacking {} with composite {}".format(image_path, composite_path))
    composite_timestamp = get_sen_2_image_timestamp(composite_path)
    image_timestamp = get_sen_2_image_timestamp(image_path)
    tile = get_sen_2_image_tile(image_path)
    out_filename = "composite_{}_{}_{}.tif".format(tile, composite_timestamp, image_timestamp)
    out_path = os.path.join(out_dir, out_filename)
    if os.path.exists(out_path) and skip_if_exists:
        log.info("{} exists, skipping".format(out_path))
        return out_path
    stack_images([composite_path, image_path], out_path)
    if create_combined_mask:
        out_mask_path = out_path.rsplit('.')[0] + ".msk"
        image_mask_path = get_mask_path(image_path)
        comp_mask_path = get_mask_path(composite_path)
        combine_masks([comp_mask_path, image_mask_path], out_mask_path, combination_func="and", geometry_func="intersect")
    return out_path


def get_l1_safe_file(image_name, l1_dir):
    """Returns the path to the L1 .SAFE directory of image. Gets from granule and timestamp. image_name can be a path or
    a filename"""
    timestamp = get_sen_2_image_timestamp(os.path.basename(image_name))
    granule = get_sen_2_image_tile(os.path.basename(image_name))
    safe_glob = "S2[A|B]_MSIL1C_{}_*_{}_*.SAFE".format(timestamp, granule)
    out = glob.glob(os.path.join(l1_dir, safe_glob))[0]
    return out


def get_l2_safe_file(image_name, l2_dir):
    """Returns the path to the L1 .SAFE directory of image. Gets from granule and timestamp. image_name can be a path or
    a filename"""
    timestamp = get_sen_2_image_timestamp(os.path.basename(image_name))
    granule = get_sen_2_image_tile(os.path.basename(image_name))
    safe_glob = "S2[A|B]_MSIL2A_{}_*_{}_*.SAFE".format(timestamp, granule)
    out = glob.glob(os.path.join(l2_dir, safe_glob))[0]
    return out


def get_sen_2_image_timestamp(image_name):
    """Returns the timestamps part of a Sentinel 2 image"""
    timestamp_re = r"\d{8}T\d{6}"
    ts_result = re.search(timestamp_re, image_name)
    return ts_result.group(0)


def get_sen_2_image_orbit(image_name):
    """Returns the relative orbit number of a Sentinel 2 image"""
    tmp1 = image_name.split("/")[-1]  # remove path
    tmp2 = tmp1.split(".")[0] # remove file extension
    comps = tmp2.split("_") # decompose
    return comps[4]


def get_sen_2_image_tile(image_name):
    """Returns the tile number of a Sentinel 2 image or path"""
    name = os.path.basename(image_name)
    tile = re.findall(r"T\d{2}[A-Z]{3}", name)[0]  # Matches tile ID, but not timestamp
    return tile


def get_sen_2_granule_id(safe_dir):
    """Returns the unique ID of a Sentinel 2 granule from a SAFE directory path"""
    tmp = os.path.basename(safe_dir) # removes path to SAFE directory
    id = tmp.split(".")[0] # removes ".SAFE" from the ID name
    return id


def get_pyeo_timestamp(image_name):
    """Returns a list of all timestamps in a Pyeo image."""
    timestamp_re = r"\d{14}"
    ts_result = re.search(timestamp_re, image_name)
    return ts_result.group(0)


def stack_images(raster_paths, out_raster_path,
                 geometry_mode="intersect", format="GTiff", datatype=gdal.GDT_Int32):
    """Stacks multiple images in image_paths together, using the information of the top image.
    geometry_mode can be "union" or "intersect" """
    log = logging.getLogger(__name__)
    log.info("Stacking images {}".format(raster_paths))
    if len(raster_paths) <= 1:
        raise StackImagesException("stack_images requires at least two input images")
    rasters = [gdal.Open(raster_path) for raster_path in raster_paths]
    total_layers = sum(raster.RasterCount for raster in rasters)
    projection = rasters[0].GetProjection()
    in_gt = rasters[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = in_gt[5]*-1   # Y resolution in agt is -ve for Maths reasons
    combined_polygons = align_bounds_to_whole_number(get_combined_polygon(rasters, geometry_mode))

    # Creating a new gdal object
    out_raster = create_new_image_from_polygon(combined_polygons, out_raster_path, x_res, y_res,
                                               total_layers, projection, format, datatype)

    # I've done some magic here. GetVirtualMemArray lets you change a raster directly without copying
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)
    present_layer = 0
    for i, in_raster in enumerate(rasters):
        log.info("Stacking image {}".format(i))
        in_raster_array = in_raster.GetVirtualMemArray()
        out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_raster, combined_polygons)
        in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(in_raster, combined_polygons)
        if len(in_raster_array.shape) == 2:
            in_raster_array = np.expand_dims(in_raster_array, 0)
        # Gdal does band, y, x
        out_raster_view = out_raster_array[
                      present_layer:  present_layer + in_raster.RasterCount,
                      out_y_min: out_y_max,
                      out_x_min: out_x_max
                      ]
        in_raster_view = in_raster_array[
                    0:in_raster.RasterCount,
                    in_y_min: in_y_max,
                    in_x_min: in_x_max
                    ]
        np.copyto(out_raster_view, in_raster_view)
        out_raster_view = None
        in_raster_view = None
        present_layer += in_raster.RasterCount
        in_raster_array = None
        in_raster = None
    out_raster_array = None
    out_raster = None


def mosaic_images(raster_paths, out_raster_file, format="GTiff", datatype=gdal.GDT_Int32, nodata = 0):
    """
    Mosaics multiple images with the same number of layers into one single image. Overwrites
    overlapping pixels with the value furthest down raster_paths. Takes projection from the first
    raster.

    TODO: consider using GDAL:

    gdal_merge.py [-o out_filename] [-of out_format] [-co NAME=VALUE]*
              [-ps pixelsize_x pixelsize_y] [-tap] [-separate] [-q] [-v] [-pct]
              [-ul_lr ulx uly lrx lry] [-init "value [value...]"]
              [-n nodata_value] [-a_nodata output_nodata_value]
              [-ot datatype] [-createonly] input_files
    """

    # This, again, is very similar to stack_rasters
    log = logging.getLogger(__name__)
    log.info("Beginning mosaic")
    rasters = [gdal.Open(raster_path) for raster_path in raster_paths]
    projection = rasters[0].GetProjection()
    in_gt = rasters[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = in_gt[5] * -1  # Y resolution in agt is -ve for Maths reasons
    combined_polyon = align_bounds_to_whole_number(get_combined_polygon(rasters, geometry_mode='union'))
    layers = rasters[0].RasterCount
    out_raster = create_new_image_from_polygon(combined_polyon, out_raster_file, x_res, y_res, layers,
                                               projection, format, datatype)
    log.info("New empty image created at {}".format(out_raster_file))
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)
    for i, raster in enumerate(rasters):
        log.info("Now mosaicking raster no. {}".format(i))
        in_raster_array = raster.GetVirtualMemArray()
        if len(in_raster_array.shape) == 2:
            in_raster_array = np.expand_dims(in_raster_array, 0)
        in_bounds = get_raster_bounds(raster)
        out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_raster, in_bounds)
        out_raster_view = out_raster_array[:, out_y_min: out_y_max, out_x_min: out_x_max]
        np.copyto(out_raster_view, in_raster_array, where=in_raster_array != nodata)
        in_raster_array = None
        out_raster_view = None
    log.info("Raster mosaicking done")
    out_raster_array = None


def composite_images_with_mask(in_raster_path_list, composite_out_path, format="GTiff"):
    """Works down in_raster_path_list, updating pixels in composite_out_path if not masked. Masks are assumed to
    be a binary .msk file with the same path as their corresponding image. All images must have the same
    number of layers and resolution, but do not have to be perfectly on top of each other. If it does not exist,
    composite_out_path will be created. Takes projection, resolution, ect from first band of first raster in list.
    Will reproject images and masks if they do not match initial raster."""

    log = logging.getLogger(__name__)
    driver = gdal.GetDriverByName(format)
    in_raster_list = [gdal.Open(raster) for raster in in_raster_path_list]
    projection = in_raster_list[0].GetProjection()
    in_gt = in_raster_list[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = in_gt[5] * -1
    n_bands = in_raster_list[0].RasterCount
    temp_band = in_raster_list[0].GetRasterBand(1)
    datatype = temp_band.DataType
    temp_band = None

    # Creating output image + array
    log.info("Creating composite at {}".format(composite_out_path))
    log.info("Composite info: x_res: {}, y_res: {}, {} bands, datatype: {}, projection: {}"
             .format(x_res, y_res, n_bands, datatype, projection))
    out_bounds = align_bounds_to_whole_number(get_poly_bounding_rect(get_combined_polygon(in_raster_list, geometry_mode="union")))
    composite_image = create_new_image_from_polygon(out_bounds, composite_out_path, x_res, y_res, n_bands,
                                                    projection, format, datatype)
    output_array = composite_image.GetVirtualMemArray(eAccess=gdal.gdalconst.GF_Write)
    if len(output_array.shape) == 2:
        output_array = np.expand_dims(output_array, 0)

    mask_paths = []

    for i, in_raster in enumerate(in_raster_list):
        mask_paths.append(get_mask_path(in_raster_path_list[i]))

        # Get a view of in_raster according to output_array
        log.info("Adding {} to composite".format(in_raster_path_list[i]))
        in_bounds = get_raster_bounds(in_raster)
        x_min, x_max, y_min, y_max = pixel_bounds_from_polygon(composite_image, in_bounds)
        output_view = output_array[:, y_min:y_max, x_min:x_max]

        # Move every unmasked pixel in in_raster to output_view
        log.info("Mask for {} at {}".format(in_raster_path_list[i], mask_paths[i]))
        in_masked = get_masked_array(in_raster, mask_paths[i])
        np.copyto(output_view, in_masked, where=np.logical_not(in_masked.mask))

        # Deallocate
        output_view = None
        in_masked = None

    output_array = None
    composite_image = None
    log.info("Composite done")
    log.info("Creating composite mask at {}".format(composite_out_path.rsplit(".")[0]+".msk"))
    combine_masks(mask_paths, composite_out_path.rsplit(".")[0]+".msk", combination_func='or', geometry_func="union")
    return composite_out_path


def reproject_directory(in_dir, out_dir, new_projection, extension = '.tif'):
    """Reprojects every file ending with extension to new_projection and saves in out_dir"""
    log = logging.getLogger(__name__)
    image_paths = [os.path.join(in_dir, image_path) for image_path in os.listdir(in_dir) if image_path.endswith(extension)]
    for image_path in image_paths:
        reproj_path = os.path.join(out_dir, os.path.basename(image_path))
        log.info("Reprojecting {} to {}, storing in {}".format(image_path, reproj_path, new_projection))
        reproject_image(image_path, reproj_path, new_projection)


def reproject_image(in_raster, out_raster_path, new_projection, driver = "GTiff",  memory = 2e3):
    """Creates a new, reprojected image from in_raster. Wraps gdal.ReprojectImage function. Will round projection
    back to whatever 2gb memory limit by default (because it works in most places)"""
    log = logging.getLogger(__name__)
    log.info("Reprojecting {} to {}".format(in_raster, new_projection))
    if type(in_raster) is str:
        in_raster = gdal.Open(in_raster)
    res = in_raster.GetGeoTransform()[1]
    gdal.Warp(out_raster_path, in_raster, dstSRS=new_projection, warpMemoryLimit=memory, format=driver)
    # After warping, image has irregular gt; resample back to previous pixel size
    # TODO: Make this an option
    resample_image_in_place(out_raster_path, res)
    return out_raster_path


def reproject_geotransform(in_gt, old_proj_wkt, new_proj_wkt):
    """Reprojects a geotransform into a new projection."""
    old_proj = osr.SpatialReference()
    new_proj = osr.SpatialReference()
    old_proj.ImportFromWkt(old_proj_wkt)
    new_proj.ImportFromWkt(new_proj_wkt)
    transform = osr.CoordinateTransformation(old_proj, new_proj)
    (ulx, uly, _) = transform.TransformPoint(in_gt[0], in_gt[3])
    out_gt = (ulx, in_gt[1], in_gt[2], uly, in_gt[4], in_gt[5])
    return out_gt


def composite_directory(image_dir, composite_out_dir, format="GTiff"):
    """Composites every image in image_dir, assumes all have associated masks.  Will
     place a file named composite_[last image date].tif inside composite_out_dir"""
    log = logging.getLogger(__name__)
    log.info("Compositing {}".format(image_dir))
    sorted_image_paths = [os.path.join(image_dir, image_name) for image_name
                          in sort_by_timestamp(os.listdir(image_dir), recent_first=False)
                          if image_name.endswith(".tif")]
    last_timestamp = get_sen_2_image_timestamp(os.path.basename(sorted_image_paths[-1]))
    composite_out_path = os.path.join(composite_out_dir, "composite_{}.tif".format(last_timestamp))
    composite_images_with_mask(sorted_image_paths, composite_out_path, format)


def change_from_composite(image_path, composite_path, model_path, class_out_path, prob_out_path):
    """Generates a change map comparing an image with a composite"""
    with TemporaryDirectory() as td:
        stacked_path = os.path.join(td, "comp_stack.tif")
        stack_images((composite_path, image_path), stacked_path)
        classify_image(stacked_path, model_path, class_out_path, prob_out_path)


def get_masked_array(raster, mask_path, fill_value = -9999):
    """Returns a numpy.mask masked array for the raster.
    Masked pixels are FALSE in the mask image (multiplicateive map),
    but TRUE in the masked_array (nodata pixels)"""
    mask = gdal.Open(mask_path)
    mask_array = mask.GetVirtualMemArray()
    raster_array = raster.GetVirtualMemArray()
    # If the shapes do not match, assume single-band mask for multi-band raster
    if len(mask_array.shape) == 2 and len(raster_array.shape) == 3:
        mask_array = project_array(mask_array, raster_array.shape[0], 0)
    return np.ma.array(raster_array, mask=np.logical_not(mask_array))


def project_array(array_in, depth, axis):
    """Returns a new array with an extra dimension. Data is projected along that dimension to depth."""
    array_in = np.expand_dims(array_in, axis)
    array_in = np.repeat(array_in, depth, axis)
    return array_in


def flatten_probability_image(prob_image, out_path):
    """Produces a single-band raster containing the highest certainties in a input probablility raster"""
    prob_raster = gdal.Open(prob_image)
    out_raster = create_matching_dataset(prob_raster, out_path, bands=1)
    prob_array = prob_raster.GetVirtualMemArray()
    out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    out_array[:, :] = prob_array.max(axis=0)
    out_array = None
    prob_array = None
    out_raster = None
    prob_raster = None


def get_combined_polygon(rasters, geometry_mode ="intersect"):
    """Calculates the overall polygon boundary for multiple rasters"""
    raster_bounds = []
    for in_raster in rasters:
        raster_bounds.append(get_raster_bounds(in_raster))
    # Calculate overall bounding box based on either union or intersection of rasters
    if geometry_mode == "intersect":
        combined_polygons = multiple_intersection(raster_bounds)
    elif geometry_mode == "union":
        combined_polygons = multiple_union(raster_bounds)
    else:
        raise Exception("Invalid geometry mode")
    return combined_polygons


def multiple_union(polygons):
    """Takes a list of polygons and returns a geometry representing the union of all of them"""
    # Note; I can see this maybe failing(or at least returning a multipolygon)
    # if two consecutive polygons do not overlap at all. Keep eye on.
    running_union = polygons[0]
    for polygon in polygons[1:]:
        running_union = running_union.Union(polygon)
    return running_union.Simplify(0)


def pixel_bounds_from_polygon(raster, polygon):
    """Returns the pixel coordinates of the bounds of the
     intersection between polygon and raster """
    raster_bounds = get_raster_bounds(raster)
    intersection = get_poly_intersection(raster_bounds, polygon)
    bounds_geo = intersection.Boundary()
    x_min_geo, x_max_geo, y_min_geo, y_max_geo = bounds_geo.GetEnvelope()
    (x_min_pixel, y_min_pixel) = point_to_pixel_coordinates(raster, (x_min_geo, y_min_geo))
    (x_max_pixel, y_max_pixel) = point_to_pixel_coordinates(raster, (x_max_geo, y_max_geo))
    # Kludge time: swap the two values around if they are wrong
    if x_min_pixel >= x_max_pixel:
        x_min_pixel, x_max_pixel = x_max_pixel, x_min_pixel
    if y_min_pixel >= y_max_pixel:
        y_min_pixel, y_max_pixel = y_max_pixel, y_min_pixel
    return x_min_pixel, x_max_pixel, y_min_pixel, y_max_pixel


def point_to_pixel_coordinates(raster, point, oob_fail=False):
    """Returns a tuple (x_pixel, y_pixel) in a georaster raster corresponding to the point.
    Point can be an ogr point object, a wkt string or an x, y tuple or list. Assumes north-up non rotated.
    Will floor() decimal output"""
    # Equation is rearrangement of section on affinine geotransform in http://www.gdal.org/gdal_datamodel.html
    if isinstance(point, str):
        point = ogr.CreateGeometryFromWkt(point)
        x_geo = point.GetX()
        y_geo = point.GetY()
    if isinstance(point, list) or isinstance(point, tuple):  # There is a more pythonic way to do this
        x_geo = point[0]
        y_geo = point[1]
    if isinstance(point, ogr.Geometry):
        x_geo = point.GetX()
        y_geo = point.GetY()
    gt = raster.GetGeoTransform()
    x_pixel = int(np.floor((x_geo - gt[0])/gt[1]))
    y_pixel = int(np.floor((y_geo - gt[3])/gt[5]))  # y resolution is -ve
    return x_pixel, y_pixel


def multiple_intersection(polygons):
    """Takes a list of polygons and returns a geometry representing the intersection of all of them"""
    running_intersection = polygons[0]
    for polygon in polygons[1:]:
        running_intersection = running_intersection.Intersection(polygon)
    return running_intersection.Simplify(0)


def stack_and_trim_images(old_image_path, new_image_path, aoi_path, out_image):
    """Stacks an old and new S2 image and trims to within an aoi"""
    log = logging.getLogger(__name__)
    if os.path.exists(out_image):
        log.warning("{} exists, skipping.")
        return
    with TemporaryDirectory() as td:
        old_clipped_image_path = os.path.join(td, "old.tif")
        new_clipped_image_path = os.path.join(td, "new.tif")
        clip_raster(old_image_path, aoi_path, old_clipped_image_path)
        clip_raster(new_image_path, aoi_path, new_clipped_image_path)
        stack_images([old_clipped_image_path, new_clipped_image_path],
                     out_image, geometry_mode="intersect")


def clip_raster(raster_path, aoi_path, out_path, srs_id=4326):
    """Clips a raster at raster_path to a shapefile given by aoi_path. Assumes a shapefile only has one polygon.
    Will np.floor() when converting from geo to pixel units and np.absolute() y resolution form geotransform."""
    # https://gis.stackexchange.com/questions/257257/how-to-use-gdal-warp-cutline-option
    with TemporaryDirectory() as td:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(srs_id)
        intersection_path = os.path.join(td, 'intersection')
        raster = gdal.Open(raster_path)
        in_gt = raster.GetGeoTransform()
        aoi = ogr.Open(aoi_path)
        intersection = get_aoi_intersection(raster, aoi)
        min_x_geo, max_x_geo, min_y_geo, max_y_geo = intersection.GetEnvelope()
        width_pix = int(np.floor(max_x_geo - min_x_geo)/in_gt[1])
        height_pix = int(np.floor(max_y_geo - min_y_geo)/np.absolute(in_gt[5]))
        new_geotransform = (min_x_geo, in_gt[1], 0, min_y_geo, 0, in_gt[5])
        write_polygon(intersection, intersection_path)
        clip_spec = gdal.WarpOptions(
            format="GTiff",
            cutlineDSName=intersection_path,
            cropToCutline=True,
            width=width_pix,
            height=height_pix,
            srcSRS=srs,
            dstSRS=srs
        )
        out = gdal.Warp(out_path, raster, options=clip_spec)
        out.SetGeoTransform(new_geotransform)
        out = None


def write_polygon(polygon, out_path, srs_id=4326):
    """Saves a polygon to a shapefile"""
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(out_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(srs_id)
    layer = data_source.CreateLayer(
        "geometry",
        srs,
        geom_type=ogr.wkbPolygon)
    feature_def = layer.GetLayerDefn()
    feature = ogr.Feature(feature_def)
    feature.SetGeometry(polygon)
    layer.CreateFeature(feature)
    data_source.FlushCache()
    data_source = None


def get_aoi_intersection(raster, aoi):
    """Returns a wkbPolygon geometry with the intersection of a raster and an aoi"""
    raster_shape = get_raster_bounds(raster)
    aoi.GetLayer(0).ResetReading()  # Just in case the aoi has been accessed by something else
    aoi_feature = aoi.GetLayer(0).GetFeature(0)
    aoi_geometry = aoi_feature.GetGeometryRef()
    return aoi_geometry.Intersection(raster_shape)


def get_raster_intersection(raster1, raster2):
    """Returns a wkbPolygon geometry with the intersection of two raster bounding boxes"""
    bounds_1 = get_raster_bounds(raster1)
    bounds_2 = get_raster_bounds(raster2)
    return bounds_1.Intersection(bounds_2)


def get_poly_intersection(poly1, poly2):
    """Trivial function returns the intersection between two polygons. No test."""
    return poly1.Intersection(poly2)


def check_overlap(raster, aoi):
    """Checks that a raster and an AOI overlap"""
    raster_shape = get_raster_bounds(raster)
    aoi_shape = get_aoi_bounds(aoi)
    if raster_shape.Intersects(aoi_shape):
        return True
    else:
        return False


def get_raster_bounds(raster):
    """Returns a wkbPolygon geometry with the bounding rectangle of a raster calculate from its geotransform"""
    raster_bounds = ogr.Geometry(ogr.wkbLinearRing)
    geotrans = raster.GetGeoTransform()
    top_left_x = geotrans[0]
    top_left_y = geotrans[3]
    width = geotrans[1]*raster.RasterXSize
    height = geotrans[5]*raster.RasterYSize * -1  # RasterYSize is +ve, but geotransform is -ve so this should go good
    raster_bounds.AddPoint(top_left_x, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(raster_bounds)
    return bounds_poly


def get_raster_size(raster):
    """Return the height and width of a raster"""
    geotrans = raster.GetGeoTransform()
    width = geotrans[1]*raster.RasterXSize
    height = geotrans[5]*raster.RasterYSize
    return width, height


def get_aoi_bounds(aoi):
    """Returns a wkbPolygon geometry with the bounding rectangle of a single-polygon shapefile"""
    aoi_bounds = ogr.Geometry(ogr.wkbLinearRing)
    (x_min, x_max, y_min, y_max) = aoi.GetLayer(0).GetExtent()
    aoi_bounds.AddPoint(x_min, y_min)
    aoi_bounds.AddPoint(x_max, y_min)
    aoi_bounds.AddPoint(x_max, y_max)
    aoi_bounds.AddPoint(x_min, y_max)
    aoi_bounds.AddPoint(x_min, y_min)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(aoi_bounds)
    return bounds_poly


def align_bounds_to_whole_number(bounding_box):
    """Shifts bounding_box polygon so that its size becomes a whole number"""
    # This should prevent off-by-one errors caused by bad image alignment
    aoi_bounds = ogr.Geometry(ogr.wkbLinearRing)
    (x_min, x_max, y_min, y_max) = bounding_box.GetEnvelope()
    # This will create a box that has a whole number as its height and width
    x_new = x_min + np.floor(x_max-x_min)
    y_new = y_min + np.floor(y_max-y_min)
    aoi_bounds.AddPoint(x_min, y_min)
    aoi_bounds.AddPoint(x_new, y_min)
    aoi_bounds.AddPoint(x_new, y_new)
    aoi_bounds.AddPoint(x_min, y_new)
    aoi_bounds.AddPoint(x_min, y_min)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(aoi_bounds)
    return bounds_poly


def get_aoi_size(aoi):
    """Returns the width and height of the bounding box of an aoi. No test"""
    (x_min, x_max, y_min, y_max) = aoi.GetLayer(0).GetExtent()
    out = (x_max - x_min, y_max-y_min)
    return out


def get_poly_size(poly):
    """Returns the width and height of a bounding box of a polygon. No test"""
    boundary = poly.Boundary()
    x_min, y_min, not_needed = boundary.GetPoint(0)
    x_max, y_max, not_needed = boundary.GetPoint(2)
    out = (x_max - x_min, y_max-y_min)
    return out


def get_poly_bounding_rect(poly):
    """Returns a polygon of the bounding rectangle of input polygon. Can probably be combined with
    get_aoi_bounds."""
    aoi_bounds = ogr.Geometry(ogr.wkbLinearRing)
    x_min, x_max, y_min, y_max = poly.GetEnvelope()
    aoi_bounds.AddPoint(x_min, y_min)
    aoi_bounds.AddPoint(x_max, y_min)
    aoi_bounds.AddPoint(x_max, y_max)
    aoi_bounds.AddPoint(x_min, y_max)
    aoi_bounds.AddPoint(x_min, y_min)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(aoi_bounds)
    return bounds_poly


def create_mask_from_model(image_path, model_path, model_clear=0, num_chunks=10, buffer_size=0):
    """Returns a multiplicative mask (0 for cloud, shadow or haze, 1 for clear) built from the model at model_path."""
    with TemporaryDirectory() as td:
        log = logging.getLogger(__name__)
        log.info("Building cloud mask for {} with model {}".format(image_path, model_path))
        temp_mask_path = os.path.join(td, "cat_mask.tif")
        classify_image(image_path, model_path, temp_mask_path, num_chunks=10)
        temp_mask = gdal.Open(temp_mask_path, gdal.GA_Update)
        temp_mask_array = temp_mask.GetVirtualMemArray()
        mask_path = get_mask_path(image_path)
        mask = create_matching_dataset(temp_mask, mask_path, datatype=gdal.GDT_Byte)
        mask_array = mask.GetVirtualMemArray(eAccess=gdal.GF_Write)
        mask_array[:, :] = np.where(temp_mask_array != model_clear, 0, 1)
        temp_mask_array = None
        mask_array = None
        temp_mask = None
        mask = None
        if buffer_size:
            buffer_mask_in_place(mask_path, buffer_size)
        log.info("Cloud mask for {} saved in {}".format(image_path, mask_path))
        return mask_path


def create_mask_from_confidence_layer(l2_safe_path, out_path, cloud_conf_threshold=0, buffer_size=3):
    """Creates a multiplicative binary mask where cloudy pixels are 0 and non-cloudy pixels are 1. If
    cloud_conf_threshold = 0, use scl mask else use confidence image """
    log = logging.getLogger(__name__)
    log.info("Creating mask for {} with {} confidence threshold".format(l2_safe_path, cloud_conf_threshold))
    if cloud_conf_threshold:
        cloud_glob = "GRANULE/*/QI_DATA/*CLD*_20m.jp2"  # This should match both old and new mask formats
        cloud_path = glob.glob(os.path.join(l2_safe_path, cloud_glob))[0]
        cloud_image = gdal.Open(cloud_path)
        cloud_confidence_array = cloud_image.GetVirtualMemArray()
        mask_array = (cloud_confidence_array < cloud_conf_threshold)
        cloud_confidence_array = None
    else:
        cloud_glob = "GRANULE/*/IMG_DATA/R20m/*SCL*_20m.jp2"  # This should match both old and new mask formats
        cloud_path = glob.glob(os.path.join(l2_safe_path, cloud_glob))[0]
        cloud_image = gdal.Open(cloud_path)
        scl_array = cloud_image.GetVirtualMemArray()
        mask_array = np.isin(scl_array, (4, 5, 6))

    mask_image = create_matching_dataset(cloud_image, out_path)
    mask_image_array = mask_image.GetVirtualMemArray(eAccess=gdal.GF_Write)
    np.copyto(mask_image_array, mask_array)
    mask_image_array = None
    cloud_image = None
    mask_image = None
    resample_image_in_place(out_path, 10)
    if buffer_size:
        buffer_mask_in_place(out_path, buffer_size)
    log.info("Mask created at {}".format(out_path))
    return out_path


def create_mask_from_class_map(class_map_path, out_path, classes_of_interest, buffer_size=0, out_resolution=None):
    """Creates a mask from a classification mask: 1 for each pixel containing one of classes_of_interest, otherwise 0"""
    # TODO: pull this out of the above function
    class_image = gdal.Open(class_map_path)
    class_array = class_image.GetVirtualMemArray()
    mask_array = np.isin(class_array, classes_of_interest)
    out_mask = create_matching_dataset(class_image, out_path, datatype=gdal.GDT_Byte)
    out_array = out_mask.GetVirtualMemArray(eAccess=gdal.GA_Update)
    np.copyto(out_array, mask_array)
    class_array = None
    class_image = None
    out_array = None
    out_mask = None
    if out_resolution:
        resample_image_in_place(out_path, out_resolution)
    if buffer_size:
        buffer_mask_in_place(out_path)
    return out_path


def create_mask_from_fmask(in_l1_dir, out_path):
    log = logging.getLogger(__name__)
    log.info("Creating fmask for {}".format(in_l1_dir))
    with TemporaryDirectory() as td:
        #pdb.set_trace()
        temp_fmask_path = os.path.join(td, "fmask.tif")
        apply_fmask(in_l1_dir, temp_fmask_path)
        fmask_image = gdal.Open(temp_fmask_path)
        fmask_array = fmask_image.GetVirtualMemArray()
        out_image = create_matching_dataset(fmask_image, out_path, datatype=gdal.GDT_Byte)
        out_array = out_image.GetVirtualMemArray(eAccess=gdal.GA_Update)
        log.info("fmask created, converting to binary cloud/shadow mask")
        out_array[:,:] = np.isin(fmask_array, (2, 3, 4), invert=True)
        out_array = None
        out_image = None
        fmask_array = None
        fmask_image = None
        resample_image_in_place(out_path, 10)


def create_mask_from_sen2cor_and_fmask(l1_dir, l2_dir, out_mask_path, buffer_size=0):
    with TemporaryDirectory() as td:
        s2c_mask_path = os.path.join(td, "s2_mask.tif")
        fmask_mask_path = os.path.join(td, "fmask.tif")
        create_mask_from_confidence_layer(l2_dir, s2c_mask_path, buffer_size=buffer_size)
        create_mask_from_fmask(l1_dir, fmask_mask_path)
        combine_masks([s2c_mask_path, fmask_mask_path], out_mask_path, combination_func="and", geometry_func="union")


def get_mask_path(image_path):
    """A gdal mask is an image with the same name as the image it's masking, but with a .msk extension"""
    image_name = os.path.basename(image_path)
    image_dir = os.path.dirname(image_path)
    mask_name = image_name.rsplit('.')[0] + ".msk"
    mask_path = os.path.join(image_dir, mask_name)
    return mask_path


def combine_masks(mask_paths, out_path, combination_func = 'and', geometry_func ="intersect"):
    """ORs or ANDs several masks. Gets metadata from top mask. Assumes that masks are a
    Python true or false. Also assumes that all masks are the same projection for now."""
    # TODO Implement intersection and union
    log = logging.getLogger(__name__)
    log.info("Combining masks {}:\n   combination function: '{}'\n   geometry function:'{}'".format(
        mask_paths, combination_func, geometry_func))
    masks = [gdal.Open(mask_path) for mask_path in mask_paths]
    combined_polygon = align_bounds_to_whole_number(get_combined_polygon(masks, geometry_func))
    gt = masks[0].GetGeoTransform()
    x_res = gt[1]
    y_res = gt[5]*-1  # Y res is -ve in geotransform
    bands = 1
    projection = masks[0].GetProjection()
    out_mask = create_new_image_from_polygon(combined_polygon, out_path, x_res, y_res,
                                             bands, projection, datatype=gdal.GDT_Byte, nodata=0)

    # This bit here is similar to stack_raster, but different enough to not be worth spinning into a combination_func
    # I might reconsider this later, but I think it'll overcomplicate things.
    out_mask_array = out_mask.GetVirtualMemArray(eAccess=gdal.GF_Write)
    out_mask_array[:, :] = 1
    for i, in_mask in enumerate(masks):
        in_mask_array = in_mask.GetVirtualMemArray()
        if geometry_func == "intersect":
            out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_mask, combined_polygon)
            in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(in_mask, combined_polygon)
        elif geometry_func == "union":
            out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_mask, get_raster_bounds(in_mask))
            in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(in_mask, get_raster_bounds(in_mask))
        else:
            raise Exception("Invalid geometry_func; can be 'intersect' or 'union'")
        out_mask_view = out_mask_array[out_y_min: out_y_max, out_x_min: out_x_max]
        in_mask_view = in_mask_array[in_y_min: in_y_max, in_x_min: in_x_max]
        if i is 0:
            out_mask_view[:,:] = in_mask_view
        else:
            if combination_func is 'or':
                out_mask_view[:,:] = np.bitwise_or(out_mask_view, in_mask_view, dtype=np.uint8)
            elif combination_func is 'and':
                out_mask_view[:,:] = np.bitwise_and(out_mask_view, in_mask_view, dtype=np.uint8)
            elif combination_func is 'nor':
                out_mask_view[:,:] = np.bitwise_not(np.bitwise_or(out_mask_view, in_mask_view, dtype=np.uint8), dtype=np.uint8)
            else:
                raise Exception("Invalid combination_func; valid values are 'or', 'and', and 'nor'")
        in_mask_view = None
        out_mask_view = None
        in_mask_array = None
        in_mask = None
    out_mask_array = None
    out_mask = None
    return out_path


def buffer_mask_in_place(mask_path, buffer_size):
    """Expands a mask in-place, overwriting the previous mask"""
    log = logging.getLogger(__name__)
    log.info("Buffering {} with buffer size {}".format(mask_path, buffer_size))
    mask = gdal.Open(mask_path, gdal.GA_Update)
    mask_array = mask.GetVirtualMemArray(eAccess=gdal.GA_Update)
    cache = morph.binary_erosion(mask_array, selem=morph.disk(buffer_size))
    np.copyto(mask_array, cache)
    mask_array = None
    mask = None


def create_new_image_from_polygon(polygon, out_path, x_res, y_res, bands,
                           projection, format="GTiff", datatype = gdal.GDT_Int32, nodata = -9999):
    """Returns an empty image of the extent of input polygon"""
    # TODO: Implement nodata
    bounds_x_min, bounds_x_max, bounds_y_min, bounds_y_max = polygon.GetEnvelope()
    if bounds_x_min >= bounds_x_max:
        bounds_x_min, bounds_x_max = bounds_x_max, bounds_x_min
    if bounds_y_min >= bounds_y_max:
        bounds_y_min, bounds_y_max = bounds_y_max, bounds_y_min
    final_width_pixels = int(np.abs(bounds_x_max - bounds_x_min) / x_res)
    final_height_pixels = int(np.abs(bounds_y_max - bounds_y_min) / y_res)
    driver = gdal.GetDriverByName(format)
    out_raster = driver.Create(
        out_path, xsize=final_width_pixels, ysize=final_height_pixels,
        bands=bands, eType=datatype
    )
    out_raster.SetGeoTransform([
        bounds_x_min, x_res, 0,
        bounds_y_max, 0, y_res * -1
    ])
    out_raster.SetProjection(projection)
    return out_raster


def resample_image_in_place(image_path, new_res):
    """Resamples an image in-place using gdalwarp to new_res in metres"""
    # I don't like using a second object here, but hey.
    with TemporaryDirectory() as td:
        args = gdal.WarpOptions(
            xRes=new_res,
            yRes=new_res
        )
        temp_image = os.path.join(td, "temp_image.tif")
        gdal.Warp(temp_image, image_path, options=args)
        shutil.move(temp_image, image_path)


def apply_array_image_mask(array, mask, fill_value=0):
    """Applies a mask of (y,x) to an image array of (bands, y, x). Replaces any masked pixels with fill_value"""
    if array.ndim == 2:
        band_count = 1
    else:
        band_count = array.shape[0]
    stacked_mask = np.stack([mask]*band_count, axis=0)
    return np.where(stacked_mask == 1, array, fill_value)


def classify_image(image_path, model_path, class_out_path, prob_out_path=None,
                   apply_mask=False, out_type="GTiff", num_chunks=10, nodata=0, skip_existing = False):
    """
    Classifies change between two stacked images.
    Images need to be chunked, otherwise they cause a memory error (~16GB of data with a ~15GB machine)
    TODO: This has gotten very hairy; rewrite when you update this to take generic models
    """
    log = logging.getLogger(__name__)
    if skip_existing:
        log.info("Checking for existing classification {}".format(class_out_path))
        if os.path.isfile(class_out_path):
            log.info("Class image exists, skipping.")
            return class_out_path
    log.info("Classifying file: {}".format(image_path))
    log.info("Saved model     : {}".format(model_path))
    image = gdal.Open(image_path)
    if num_chunks == None:
        log.info("No chunk size given, attempting autochunk.")
        num_chunks = autochunk(image)
        log.info("Autochunk to {} chunks".format(num_chunks))
    model = joblib.load(model_path)
    class_out_image = create_matching_dataset(image, class_out_path, format=out_type, datatype=gdal.GDT_Byte)
    log.info("Created classification image file: {}".format(class_out_path))
    if prob_out_path:
        log.info("n classes in the model: {}".format(model.n_classes_))
        prob_out_image = create_matching_dataset(image, prob_out_path, bands=model.n_classes_, datatype=gdal.GDT_Float32)
        log.info("Created probability image file: {}".format(prob_out_path))
    model.n_cores = -1
    image_array = image.GetVirtualMemArray()

    if apply_mask:
        mask_path = get_mask_path(image_path)
        log.info("Applying mask at {}".format(mask_path))
        mask = gdal.Open(mask_path)
        mask_array = mask.GetVirtualMemArray()
        image_array = apply_array_image_mask(image_array, mask_array)
        mask_array = None
        mask = None

    # Mask out missing values from the classification
    # at this point, image_array has dimensions [band, y, x]
    log.info("Reshaping image from GDAL to Scikit-Learn dimensions")
    image_array = reshape_raster_for_ml(image_array)
    # Now it has dimensions [x * y, band] as needed for Scikit-Learn

    # Determine where in the image array there are no missing values in any of the bands (axis 1)
    log.info("Finding good pixels without missing values")
    log.info("image_array.shape = {}".format(image_array.shape))
    n_samples = image_array.shape[0]  # gives x * y dimension of the whole image
    if 0 in image_array:  # a quick pre-check
        good_samples = image_array[np.all(image_array != nodata, axis=1), :]
        good_indices = [i for (i, j) in enumerate(image_array) if np.all(j != nodata)] # This is slowing things down too much
        n_good_samples = len(good_samples)
    else:
        good_samples = image_array
        good_indices = range(0, n_samples)
        n_good_samples = n_samples
    log.info("   All  samples: {}".format(n_samples))
    log.info("   Good samples: {}".format(n_good_samples))
    classes = np.full(n_good_samples, nodata, dtype=np.ubyte)
    if prob_out_path:
        probs = np.full((n_good_samples, model.n_classes_), nodata, dtype=np.float32)

    chunk_size = int(n_good_samples / num_chunks)
    chunk_resid = n_good_samples - (chunk_size * num_chunks)
    log.info("   Number of chunks {} Chunk size {} Chunk residual {}".format(num_chunks, chunk_size, chunk_resid))
    # The chunks iterate over all values in the array [x * y, bands] always with 8 bands per chunk
    for chunk_id in range(num_chunks):
        offset = chunk_id * chunk_size
        # process the residual pixels with the last chunk
        if chunk_id == num_chunks - 1:
            chunk_size = chunk_size + chunk_resid
        log.info("   Classifying chunk {} of size {}".format(chunk_id, chunk_size))
        chunk_view = good_samples[offset : offset + chunk_size]
        #indices_view = good_indices[offset : offset + chunk_size]
        out_view = classes[offset : offset + chunk_size]  # dimensions [chunk_size]
        out_view[:] = model.predict(chunk_view)

        if prob_out_path:
            log.info("   Calculating probabilities")
            prob_view = probs[offset : offset + chunk_size, :]
            prob_view[:, :] = model.predict_proba(chunk_view)

    log.info("   Creating class array of size {}".format(n_samples))
    class_out_array = np.full((n_samples), nodata)
    for i, class_val in zip(good_indices, classes):
        class_out_array[i] = class_val

    log.info("   Creating GDAL class image")
    class_out_image.GetVirtualMemArray(eAccess=gdal.GF_Write)[:, :] = \
        reshape_ml_out_to_raster(class_out_array, image.RasterXSize, image.RasterYSize)

    if prob_out_path:
        log.info("   Creating probability array of size {}".format(n_samples * model.n_classes_))
        prob_out_array = np.full((n_samples, model.n_classes_), nodata)
        for i, prob_val in zip(good_indices, probs):
            prob_out_array[i] = prob_val
        log.info("   Creating GDAL probability image")
        log.info("   N Classes = {}".format(prob_out_array.shape[1]))
        log.info("   Image X size = {}".format(image.RasterXSize))
        log.info("   Image Y size = {}".format(image.RasterYSize))
        prob_out_image.GetVirtualMemArray(eAccess=gdal.GF_Write)[:, :, :] = \
            reshape_prob_out_to_raster(prob_out_array, image.RasterXSize, image.RasterYSize)

    class_out_image = None
    prob_out_image = None
    if prob_out_path:
        return class_out_path, prob_out_path
    else:
        return class_out_path


def autochunk(dataset, mem_limit=None):
    """Calculates the number of chunks to break a dataset into without a memory error.
    We want to break the dataset into as few chunks as possible without going over mem_limit.
    mem_limit defaults to total amount of RAM available on machine if not specified"""
    pixels = dataset.RasterXSize * dataset.RasterYSize
    bytes_per_pixel = dataset.GetVirtualMemArray().dtype.itemsize*dataset.RasterCount
    image_bytes = bytes_per_pixel*pixels
    if not mem_limit:
        mem_limit = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
        # Lets assume that 20% of memory is being used for non-map bits
        mem_limit = int(mem_limit*0.8)
    # if I went back now, I would fail basic programming here.
    for num_chunks in range(1, pixels):
        if pixels % num_chunks != 0:
            continue
        chunk_size_bytes = (pixels/num_chunks)*bytes_per_pixel
        if chunk_size_bytes < mem_limit:
            return num_chunks


def covert_image_format(image, format):
    pass


def classify_directory(in_dir, model_path, class_out_dir, prob_out_dir,
                       apply_mask=False, out_type="GTiff", num_chunks=None):
    """
    Classifies every .tif in in_dir using model at model_path. Outputs are saved
    in class_out_dir and prob_out_dir, named [input_name]_class and _prob, respectively.
    """
    log = logging.getLogger(__name__)
    log.info("Classifying files in {}".format(in_dir))
    log.info("Class files saved in {}".format(class_out_dir))
    log.info("Prob. files saved in {}".format(prob_out_dir))
    for image_path in glob.glob(in_dir+r"/*.tif"):
        image_name = os.path.basename(image_path).split('.')[0]
        class_out_path = os.path.join(class_out_dir, image_name+"_class.tif")
        prob_out_path = os.path.join(prob_out_dir, image_name+"_prob.tif")
        classify_image(image_path, model_path, class_out_path, prob_out_path,
                       apply_mask, out_type, num_chunks)


def reshape_raster_for_ml(image_array):
    """Reshapes an array from gdal order [band, y, x] to scikit order [x*y, band]"""
    bands, y, x = image_array.shape
    image_array = np.transpose(image_array, (1, 2, 0))
    image_array = np.reshape(image_array, (x * y, bands))
    return image_array


def reshape_ml_out_to_raster(classes, width, height):
    """Reshapes an output [x*y] to gdal order [y, x]"""
    # TODO: Test this.
    image_array = np.reshape(classes, (height, width))
    return image_array


def reshape_prob_out_to_raster(probs, width, height):
    """reshapes an output of shape [x*y, classes] to gdal order [classes, y, x]"""
    classes = probs.shape[1]
    image_array = np.transpose(probs, (1, 0))
    image_array = np.reshape(image_array, (classes, height, width))
    return image_array


def create_trained_model(training_image_file_paths, cross_val_repeats = 5, attribute="CODE"):
    """Returns a trained random forest model from the training data. This
    assumes that image and model are in the same directory, with a shapefile.
    Give training_image_path a path to a list of .tif files. See spec in the R drive for data structure.
    At present, the model is an ExtraTreesClassifier arrived at by tpot; see tpot_classifier_kenya -> tpot 1)"""
    # This could be optimised by pre-allocating the training array. but not now.
    learning_data = None
    classes = None
    for training_image_file_path in training_image_file_paths:
        training_image_folder, training_image_name = os.path.split(training_image_file_path)
        training_image_name = training_image_name[:-4]  # Strip the file extension
        shape_path = os.path.join(training_image_folder, training_image_name, training_image_name + '.shp')
        this_training_data, this_classes = get_training_data(training_image_file_path, shape_path, attribute)
        if learning_data is None:
            learning_data = this_training_data
            classes = this_classes
        else:
            learning_data = np.append(learning_data, this_training_data, 0)
            classes = np.append(classes, this_classes)
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
                                     min_samples_split=16, n_estimators=100, n_jobs=4, class_weight='balanced')
    model.fit(learning_data, classes)
    scores = cross_val_score(model, learning_data, classes, cv=cross_val_repeats)
    return model, scores


def create_model_for_region(path_to_region, model_out, scores_out, attribute="CODE"):
    """Creates a model based on training data for files in a given region"""
    image_glob = os.path.join(path_to_region, r"*.tif")
    image_list = glob.glob(image_glob)
    model, scores = create_trained_model(image_list, attribute=attribute)
    joblib.dump(model, model_out)
    with open(scores_out, 'w') as score_file:
        score_file.write(str(scores))


def create_model_from_signatures(sig_csv_path, model_out):
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
                                     min_samples_split=16, n_estimators=100, n_jobs=4, class_weight='balanced')
    data = np.loadtxt(sig_csv_path, delimiter=",").T
    model.fit(data[1:, :].T, data[0, :])
    joblib.dump(model, model_out)


def get_training_data(image_path, shape_path, attribute="CODE", shape_projection_id=4326):
    """Given an image and a shapefile with categories, return x and y suitable
    for feeding into random_forest.fit.
    Note: THIS WILL FAIL IF YOU HAVE ANY CLASSES NUMBERED '0'
    WRITE A TEST FOR THIS TOO; if this goes wrong, it'll go wrong quietly and in a way that'll cause the most issues
     further on down the line."""
    with TemporaryDirectory() as td:
        shape_projection = osr.SpatialReference()
        shape_projection.ImportFromEPSG(shape_projection_id)
        image = gdal.Open(image_path)
        image_gt = image.GetGeoTransform()
        x_res, y_res = image_gt[1], image_gt[5]
        ras_path = os.path.join(td, "poly_ras")
        ras_params = gdal.RasterizeOptions(
            noData=0,
            attribute=attribute,
            xRes=x_res,
            yRes=y_res,
            outputType=gdal.GDT_Int16,
            outputSRS=shape_projection
        )
        # This produces a rasterised geotiff that's right, but not perfectly aligned to pixels.
        # This can probably be fixed.
        gdal.Rasterize(ras_path, shape_path, options=ras_params)
        rasterised_shapefile = gdal.Open(ras_path)
        shape_array = rasterised_shapefile.GetVirtualMemArray()
        local_x, local_y = get_local_top_left(image, rasterised_shapefile)
        shape_sparse = sp.coo_matrix(shape_array)
        y, x, features = sp.find(shape_sparse)
        training_data = np.empty((len(features), image.RasterCount))
        image_array = image.GetVirtualMemArray()
        image_view = image_array[:,
                    local_y: local_y + rasterised_shapefile.RasterYSize,
                    local_x: local_x + rasterised_shapefile.RasterXSize
                    ]
        for index in range(len(features)):
            training_data[index, :] = image_view[:, y[index], x[index]]
        return training_data, features


def get_local_top_left(raster1, raster2):
    """Gets the top-left corner of raster1 in the array of raster 2; WRITE A TEST FOR THIS"""
    inner_gt = raster2.GetGeoTransform()
    return point_to_pixel_coordinates(raster1, [inner_gt[0], inner_gt[3]])
