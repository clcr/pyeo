"""
pyeo.queries_and_downloads
==========================
Functions for querying, filtering and downloading data.

SAFE files
----------
Sentinel 2 data is downloaded in the form of a .SAFE file; all download functions will end with data in this structure.
This is a directory structure that contains the imagery, metadata and supplementary data of a Sentinel 2 image. The
rasters themeselves are the in the GRANULE/[granule_id]/IMG_DATA/[resolution]/ folder; each band is contained in
its own .jp2 file. For full details, see https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/data-formats

There are two ways to refer to a given Sentinel-2 products: the UUID and the product ID.
The UUID is a set of five five-character strings (EXMAPLE HERE)
The product ID is a human-readable string (more or less) containing all the information needed for unique identification
of an product, split by the underscore character. For more information on the structure of a product ID,
see (EXAMPLE HERE)

Query data structure
--------------------
All query functions return a dictionary. The key of the dictionary is the UUID of the product id; the product is
a further set of nested dictionaries containing information about the product to be downloaded. (PUT STRUCTURE HERE)

Data download sources
---------------------
This library presently offers two options for download sources; Scihub and Amazon Web Services. If in doubt, use Scihub.

- Scihub

   The Copernicus Open-Access Hub is the default option for downloading sentinel-2 images. Images are downloaded in .zip
   format, and then automatically unzipped. Users are required to register with a username and password before downloading,
   and there is a limit to no more than two concurrent downloads per username at a time. Scihub is entirely free.

- AWS

   Sentinel data is also publically hosted on Amazon Web Services. This storage is provided by Sinergise, and is normally
   updated a few hours after new products are made available. There is a small charge associated with downloading this
   data. To access the AWS repository, you are required to register an Amazon Web Services account (including providing
   payment details) and obtain an API key for that account. See https://aws.amazon.com/s3/pricing/ for pricing details;
   the relevant table is Data Transfer Pricing for the EU (Frankfurt) region. There is no limit to the concurrent downloads
   for the AWS bucket.

Functions
---------
"""

import datetime as dt
import glob
import itertools
import json
import logging
import os
import shutil
import zipfile
from multiprocessing.dummy import Pool

import requests
import tenacity
from botocore.exceptions import ClientError

from sentinelhub import download_safe_format
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson

from pyeo.filesystem_utilities import check_for_invalid_l2_data, check_for_invalid_l1_data, get_sen_2_image_tile
from pyeo.exceptions import NoL2DataAvailableException, BadDataSourceExpection, TooManyRequests

log = logging.getLogger("pyeo")

try:
    from google.cloud import storage
except ImportError:
    pass


def sent2_query(user, passwd, geojsonfile, start_date, end_date, cloud=50):
    """
    Fetches a list of Sentienl-2 products

    Parameters
    -----------

    user : string
           Username for ESA hub. Register at https://scihub.copernicus.eu/dhus/#/home

    passwd : string
             password for the ESA Open Access hub

    geojsonfile : string
                  Path to a geojson file containing a polygon of the outline of the area you wish to download.
                  See www.geojson.io for a tool to build these.

    start_date : string
                 Date of beginning of search in the format YYYY-MM-DDThh:mm:ssZ (ISO standard)

    end_date : string
               Date of end of search in the format yyyy-mm-ddThh:mm:ssZ
               See https://www.w3.org/TR/NOTE-datetime, or use cehck_for_s2_data_by_date

    cloud : string (optional)
            The maximum cloud clover (as calculated by Copernicus) to download.

    Returns
    -------
    A dictionary of Sentinel-2 granule products that are touched by your AOI polygon, keyed by product ID.
    Returns both level 1 and level 2 data.

    Notes
    -----
    If you get a 'request too long' error, it is likely that your polygon is too complex. The following functions
    download by granule; there is no need to have a precise polygon at this stage.

    """
    # Originally by Ciaran Robb
    api = SentinelAPI(user, passwd)
    footprint = geojson_to_wkt(read_geojson(geojsonfile))
    log.info("Sending query:\nfootprint: {}\nstart_date: {}\nend_date: {}\n cloud_cover: {} ".format(
        footprint, start_date, end_date, cloud))
    products = api.query(footprint,
                         date=(start_date, end_date), platformname="Sentinel-2",
                         cloudcoverpercentage="[0 TO {}]".format(cloud))
    return products


def check_for_s2_data_by_date(aoi_path, start_date, end_date, conf, cloud_cover=50):
    """
    Gets all the products between start_date and end_date. Wraps sent2_query to avoid having passwords and
    long-format timestamps in code.

    Parameters
    ----------
    aoi_path
        Path to a geojson file containing a polygon of the outline of the area you wish to download.
        See www.geojson.io for a tool to build these.

    start_date
        Start date in the format yyyymmdd.

    end_date
        End date of the query in the format yyyymmdd

    conf
        Output from a configuration file containing your username and password for the ESA hub.
        If needed, this can be dummied with a dictionary of the following format:
        conf={'sent_2':{'user':'your_username', 'pass':'your_pass'}}

    cloud_cover
        The maximem level of cloud cover in images to be downloaded.

    Returns
    -------

    """
    log.info("Querying for imagery between {} and {} for aoi {}".format(start_date, end_date, aoi_path))
    user = conf['sent_2']['user']
    password = conf['sent_2']['pass']
    start_timestamp = dt.datetime.strptime(start_date, '%Y%m%d').isoformat(timespec='seconds')+'Z'
    end_timestamp = dt.datetime.strptime(end_date, '%Y%m%d').isoformat(timespec='seconds')+'Z'
    result = sent2_query(user, password, aoi_path, start_timestamp, end_timestamp, cloud=cloud_cover)
    log.info("Search returned {} images".format(len(result)))
    return result


def filter_to_l1_data(query_output):
    """
    Takes list of products from check_for_s2_data_by_date and removes all non Level 1 products.

    Parameters
    ----------
    query_output
        A dictionary of products


    Returns
    -------
    A dictionary of products containing only the L1C data products

    """
    log.info("Extracting only L1 data from {} products".format(len(query_output)))
    filtered_query = {key: value for (key, value) in query_output.items() if get_query_level(value) == "Level-1C"}
    return filtered_query


def filter_to_l2_data(query_output):
    """
    Takes list of products from check_for_s2_data_by_date and removes all non Level 2A products.

    Parameters
    ----------
    query_output
        A dictionary of products


    Returns
    -------
    A dictionary of products containing only the L2A data products

    """
    log.info("Extracting only L2 data from {} products".format(len(query_output)))
    filtered_query = {key: value for (key, value) in query_output.items() if get_query_level(value) == "Level-2A"}
    return filtered_query


def filter_non_matching_s2_data(query_output):
    """
    Filters a query such that it only contains paired level 1 and level 2 data products.

    Parameters
    ----------
    query_output
        Query list

    Returns
    -------
    A dictionary of products contaiing only L1 and L2 data.

    """
    # Here be algorithms
    # A L1 and L2 image are related if and only if the following fields match:
    #    Satellite (S2[A|B])
    #    Intake date (FIRST timestamp)
    #    Orbit number (Rxxx)
    #    Granule ID (Txxaaa)
    # So if we succeviely partition the query, we should get a set of products with either 1 or
    # 2 entries per granule / timestamp combination
    sorted_query = sorted(query_output.values(), key=get_query_granule)
    granule_groups = {str(key): list(group) for key, group in itertools.groupby(sorted_query, key=get_query_granule)}
    granule_date_groups = {}

    # Partition as above.
    # We can probably expand this to arbitrary lengths of queries. If you catch me doing this, please restrain me.
    for granule, item_list in granule_groups.items():
        item_list.sort(key=get_query_datatake)
        granule_date_groups.update(
            {str(granule)+str(key): list(group) for key, group in itertools.groupby(item_list, key=get_query_datatake)})

    # On debug inspection, turns out sometimes S2 products get replicated. Lets filter those.
    out_set = {}
    for key, image_set in granule_date_groups.items():
        #if sum(1 for image in image_set if get_query_level(image) == "Level-2A") <= 2:
            #list(filter(lambda x: get_query_level(x) == "Level-2A", image_set)).sort(key=get_query_processing_time)[0].pop()
        if (sum(1 for image in image_set if get_query_level(image) == "Level-2A") == 1
        and sum(1 for image in image_set if get_query_level(image) == "Level-1C") == 1):
            out_set.update({image["uuid"]: image for image in image_set})

    #Finally, check that there is actually something here.
    if len(out_set) == 0:
        log.error("No L2 data detected for query. Please remove the --download_l2_data flag or request more recent images.")
        raise NoL2DataAvailableException
    return out_set


def get_query_datatake(query_item):
    """
    Gets the datatake timestamp of a query item.

    Parameters
    ----------
    query_item
        An item from a query results dictionary.

    Returns
    -------
    The timestamp of that item's datatake.
    """
    return query_item['beginposition']


def get_query_granule(query_item):
    """
    Gets the granule ID (ex: 48MXU) of a query

    Parameters
    ----------
    query_item
        An item from a query results dictionary.

    Returns
    -------
    The granule ID of that item.

    """
    return query_item["title"].split("_")[5]


def get_query_processing_time(query_item):
    """
    Returns the processing timestamps of a query item

    Parameters
    ----------
    query_item
        An item from a query results dictionary.

    Returns
    -------
    The date processing timestamp in the format yyyymmddThhmmss (Ex: 20190613T123002)

    """
    ingestion_string = query_item["title"].split("_")[6]
    return dt.datetime.strptime(ingestion_string, "%Y%m%dT%H%M%S")


def get_query_level(query_item):
    """
    Returns the processing level of the query item.

    Parameters
    ----------
    query_item
         An item from a query results dictionary.

    Returns
    -------
    A string of either 'Level-1C' or 'Level-2A'.

    """
    return query_item["processinglevel"]


def get_granule_identifiers(safe_product_id):
    """
    Returns the parts of a S2 name that uniquely identify that granulate at a moment in time
    Parameters
    ----------
    safe_product_id
        The filename of a SAFE product

    Returns
    -------
    satellite
        A string of either "L2A" or "L2B"
    intake_date
        The timestamp of the data intoake of this granule
    orbit number
        The orbit number of this granule
    granule
        The ID of this granule

    """
    satellite, _, intake_date, _, orbit_number, granule, _ = safe_product_id.split('_')
    return satellite, intake_date, orbit_number, granule


def download_s2_data(new_data, l1_dir, l2_dir, source='scihub', user=None, passwd=None, try_scihub_on_fail=False):
    """
    Downloads S2 imagery from AWS, google_cloud or scihub. new_data is a dict from Sentinel_2.

    Parameters
    ----------
    new_data
        A query dictionary contining the products you want to download
    l1_dir
        The directory to download level 1 products to.
    l2_dir
        The directory to download level 2 products to.
    source
        The source to download the data from. Can be 'scihub' or 'aws'; see section introduction for details
    user
        The username for sentinelhub
    passwd
        The password for sentinelheub
    try_scihub_on_fail
        If true, this function will roll back to downloading from Scihub on a failure of any other downloader.

    Raises
    ------
    BadDataSource
        Raised when passed either a bad datasource or a bad image ID

    """
    for image_uuid in new_data:
        identifier = new_data[image_uuid]['identifier']
        if 'L1C' in identifier:
            out_path = os.path.join(l1_dir, identifier+".SAFE")
            if check_for_invalid_l1_data(out_path) == 1:
                log.info("L1 imagery exists, skipping download")
                continue
        elif 'L2A' in identifier:
            out_path = os.path.join(l2_dir, identifier+".SAFE")
            if check_for_invalid_l2_data(out_path) == 1:
                log.info("L2 imagery exists, skipping download")
                continue
        else:
            log.error("{} is not a Sentinel 2 product".format(identifier))
            raise BadDataSourceExpection

        log.info("Downloading {} from {} to {}".format(new_data[image_uuid]['identifier'], source, out_path))
        if source=='aws':
            if try_scihub_on_fail:
                download_from_aws_with_rollback(product_id=new_data[image_uuid]['identifier'], folder=out_path,
                                                uuid=image_uuid, user=user, passwd=passwd)
            else:
                download_safe_format(product_id=new_data[image_uuid]['identifier'], folder=out_path)
        elif source=='google':
            download_from_google_cloud([new_data[image_uuid]['identifier']], out_folder=out_path)
        elif source=="scihub":
            download_from_scihub(image_uuid, out_path, user, passwd)
        else:
            log.error("Invalid data source; valid values are 'aws', 'google' and 'scihub'")
            raise BadDataSourceExpection


def download_from_aws_with_rollback(product_id, folder, uuid, user, passwd):
    """
    Attempts to download a single product from AWS using product_id; if not found, rolls back to Scihub using the UUID

    Parameters
    ----------
    product_id
        The product ID ("L2A_...")
    folder
        The folder to download the .SAFE file to.
    uuid
        The product UUID (4dfB4-432df....)
    user
        Scihub username
    passwd
        Scihub password

    """
    log = logging.getLogger(__file__)
    try:
        download_safe_format(product_id=product_id, folder=folder)
    except ClientError:
        log.warning("Something wrong with AWS for products id {}; rolling back to Scihub using uuid {}".format(product_id, uuid))
        download_from_scihub(uuid, folder, user, passwd)


def download_from_scihub(product_uuid, out_folder, user, passwd):
    """
    Downloads and unzips product_uuid from scihub

    Parameters
    ----------
    product_uuid
        The product UUID (4dfB4-432df....)
    out_folder
        The folder to save the .SAFE file to
    user
        Scihub username
    passwd
        Scihub password

    Notes
    -----
    If interrupted mid-download, there will be a .incomplete file in the download folder. You might need to remove
    this for further processing.

    """
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


def download_from_google_cloud(product_ids, out_folder, redownload = False):
    """Still experimental."""
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


def download_blob_from_google(bucket, object_prefix, out_folder, s2_object):
    """Still experimental."""
    log = logging.getLogger(__name__)
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
    """
    Returns an API key from a single-line text file containing that API

    Parameters
    ----------
    path_to_api
        The path a text file containing only the API key
    Returns
    -------
    Returns the API key
    """
    with open(path_to_api, 'r') as api_file:
        return api_file.read()


def get_planet_product_path(planet_dir, product):
    """
    Returns the path to a Planet product within a Planet directory
    """
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


def read_aoi(aoi_path):
    """
    Opens the geojson file for the aoi. If FeatureCollection, return the first feature.

    Parameters
    ----------
    aoi_path
        The path to the geojson file
    Returns
    -------
    A dictionary translation of the feature inside the .json file

    """
    with open(aoi_path, 'r') as aoi_fp:
        aoi_dict = json.load(aoi_fp)
        if aoi_dict["type"] == "FeatureCollection":
            aoi_dict = aoi_dict["features"][0]
        return aoi_dict