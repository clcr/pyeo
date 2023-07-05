import logging
import os
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import toml
from sentinelsat import QueryLengthError, SentinelAPI
from shapely.geometry import box

import pyeo.utils.pyeo_exceptions as exceptions

CONFIG = toml.load(os.path.abspath("config/image_acquisition_config.toml"))
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    filename=f"pyeo_run_{datetime.now().strftime('%d-%b-%Y(%H:%M:%S.%f)')}.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("pyeo")
ROI_GEOMETRY = gpd.read_file(CONFIG["sentinel2_properties"]["tiles_to_process"])
ROI_GEOMETRY["centroid"] = ROI_GEOMETRY.representative_point()
TILE_DICT = {
    ROI_GEOMETRY["name"][i]: str(ROI_GEOMETRY["centroid"][i])
    for i in range(ROI_GEOMETRY.shape[0])
}

user = CONFIG["authentication"]["username"]
password = CONFIG["authentication"]["password"]

API_URL = "https://scihub.copernicus.eu/dhus/"
REST_URL = "https://apihub.copernicus.eu/apihub/search"

API = SentinelAPI(user, password, timeout=600)
API.api_url = API_URL


def get_tiles_to_process(file_path: str) -> Union[pd.DataFrame, List]:
    """
    This function returns a DataFrame of Sentinel-2 tiles generated either from a given geometry,
    or from a .csv file of defined Sentinel-2 tiles.

    Parameters
    ----------
    file_path: path to either geometry or a .csv with Sentinel-2 tiles defined.

    Returns
    -------
    DataFrame of Sentinel-2 tiles or List of Tiles

    """
    if file_path.endswith(
        ".csv"
    ):  # The .csv file must be a single column .csv with a column named "name".
        return pd.read_csv(file_path)
    elif file_path.endswith(".shp"):
        aoi = gpd.read_file(file_path)
    elif file_path.endswith(".json"):
        aoi = gpd.read_file(file_path)
    elif file_path.endswith(".geojson"):
        aoi = gpd.read_file(file_path)
    else:
        LOGGER.error(
            f"Unsuitable file type: {file_path}, get_tiles_to_process() must get a .geojson, .json, .shp, or .csv file."
        )
        raise exceptions.PyeoException("Unsupported file type supplied")

    intersection = ROI_GEOMETRY.sjoin(aoi)
    tile_list = list(intersection["name_right"].unique())[:3]

    return tile_list


def sentinel_query(
    api: SentinelAPI, config: Dict, from_tile_dict: bool, tile_dict=None, tile=None
) -> Dict:
    """
    This function generates a query for Sentinel Scihub and returns a response
    of available imagery

    Parameters
    ----------
    api: Instance of SentinelAPI class.
    config: Configuration dictionary
    from_tile_dict: Boolean, use TILE_DICT or not
    tile_dict: Dictionary of tile names
    tile: Key within tile_dict (tile_name)

    Returns
    -------
    Products dictionary

    """

    if from_tile_dict:
        area = tile_dict[tile]
    else:
        tile_id_geoms = CONFIG["sentinel2_properties"]["tiles_to_process"]
        geom = gpd.read_file(tile_id_geoms)
        x, y, x1, y1 = geom["geometry"][0].bounds
        bounds = box(x, y, x1, y1)
        area = bounds  # simplifies geometry by getting the bounds

    def query(*args, **kwargs):
        return api.query(*args, **kwargs)

    query_kwargs = {
        "area": area,
        "platformname": "Sentinel-2",
        "cloudcoverpercentage": (0, config["sentinel2_properties"]["max_cloud_cover"]),
        "date": (
            config["sentinel2_properties"]["start_date"],
            config["sentinel2_properties"]["end_date"],
        ),
    }

    try:
        products = query(**query_kwargs)
    except QueryLengthError:
        LOGGER.info(f"Query Length has exceeded the limit.")
        raise exceptions.PyeoException("Query Length has exceeded the limit.")

    return products


def get_available_s2_imagery(use_config_roi: True) -> List[Dict]:
    """
    Returns a dictionary of all available Sentinel-2 imagery for a given Area of Interest
    and between given dates. The area of interest, date range to search, user credentials
    and acceptable cloud cover are defined in the change_detection_config.toml file.

    Parameters
    ----------
    use_config_roi: a boolean statement indicating whether to use the roi defined within the config,
                    under [sentinel2_properties][tiles_to_process]

    Returns
    -------
    available_imagery : list

    """
    LOGGER.info("Searching for available Sentinel-2 imagery")

    available_imagery = []

    if use_config_roi:
        LOGGER.info("Using config roi to query imagery")
        response = sentinel_query(api=API, config=CONFIG, from_tile_dict=False)
        available_imagery.append(response)

        if len(available_imagery) == 0:
            LOGGER.info(f"No Images retrieved, as no images satisfy the query.")

    else:
        LOGGER.info("Using tile dictionary to query imagery")
        tiles = get_tiles_to_process(
            file_path=CONFIG["sentinel2_properties"]["tiles_to_process"]
        )
        for tile in tiles:
            response = sentinel_query(
                api=API,
                config=CONFIG,
                tile_dict=TILE_DICT,
                tile=tile,
                from_tile_dict=True,
            )
            available_imagery.append(response)
            LOGGER.info(f"Received tile: {tile} response from Scihub.")

        if len(available_imagery) == 0:
            LOGGER.info(f"No Images retrieved, as no images satisfy the query.")

    return available_imagery


def filter_valid_s2_products(available_imagery: List) -> pd.DataFrame:
    """
    This function filters the whole API response, to only show valid imagery to download.
    Valid imagery is imagery with size larger than a defined value (e.g. 500MB).

    Parameters
    ----------
    available_imagery: All imagery from the Scihub API response

    Returns
    -------
    Filtered DataFrame of Sentinel-2 Scihub API response

    """
    number_of_products = sum([len(x) for x in available_imagery])
    LOGGER.info(f"Found {number_of_products} L1C and L2A products for the composite")
    columns = list(list(available_imagery[0].values())[0].keys())

    all_products = pd.DataFrame([], index=columns).T

    for api_response in available_imagery:
        for image in api_response.values():
            all_products = pd.concat(
                [all_products, pd.DataFrame.from_dict(image, orient="index").T]
            )

    all_products = all_products.reset_index(drop=True)

    all_products["size"] = (
        all_products["size"]
        .str.split(" ")
        .apply(lambda x: float(x[0]) * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]])
    )

    filtered_products = all_products.query("size >= " + str(500))
    LOGGER.info(
        f"Removed {len(all_products) - len(filtered_products)} faulty scenes <500MB in size from the list."
    )

    return filtered_products


def filter_preprocessing_level(
    filtered_products: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function identifies any overlapping images within the differing processing levels.
    It then splits up the API response DataFrame based on preprocessing level
    (e.g. L1C or L2A Data).

    Parameters
    ----------
    filtered_products - Filtered data dataframe from filter_valid_s2_products function output.

    Returns
    -------
    Dataframe of products filtered by preprocessing level.

    """
    filtered_products = filtered_products.sort_values(["tileid", "beginposition"])
    rows2drop = []
    for r in range(filtered_products.shape[0] - 1):
        if (
            filtered_products.iloc[r, :]["tileid"]
            == filtered_products.iloc[r + 1, :]["tileid"]
        ):
            if (
                filtered_products.iloc[r, :]["beginposition"]
                == filtered_products.iloc[r + 1, :]["beginposition"]
            ):
                if filtered_products.iloc[r, :]["processinglevel"] == "Level-1C":
                    rows2drop.append(r)
                else:
                    if (
                        filtered_products.iloc[r + 1, :]["processinglevel"]
                        == "Level-1C"
                    ):
                        rows2drop.append(r + 1)
                    else:
                        LOGGER.warning(
                            "Neither of the matching rows in the query result is L1C."
                        )

    if len(rows2drop) > 0:
        filtered_products = filtered_products.drop(
            index=list(filtered_products.iloc[rows2drop, :].index)
        )

    l1c_products = filtered_products[
        filtered_products.processinglevel == "Level-1C"
    ].reset_index(drop=True)
    l2a_products = filtered_products[
        filtered_products.processinglevel == "Level-2A"
    ].reset_index(drop=True)

    LOGGER.info(f"{len(l1c_products)} L1C products found")
    LOGGER.info(f"{len(l2a_products)} L2A products found")

    return l1c_products, l2a_products


def stratify_by_relative_orbit_number(filtered_products: pd.DataFrame) -> pd.DataFrame:
    """
    This function stratifies low cloud cover imagery by relative orbit number.

    Parameters
    ----------
    filtered_products: Dataframe with valid filtered products.

    Returns
    -------
    stratified_products: Dataframe of products stratified by cloud cover.

    """
    max_image_number = CONFIG["sentinel2_properties"]["max_image_number"]
    rel_orbits = np.unique(filtered_products["relativeorbitnumber"])
    stratified_products = filtered_products.copy(deep=True)

    if len(rel_orbits) > 0:
        if stratified_products.shape[0] > max_image_number / len(rel_orbits):
            uuids = []
            for orb in rel_orbits:
                uuids = (
                    uuids
                    + list(
                        stratified_products[
                            stratified_products["relativeorbitnumber"] == orb
                        ].sort_values(by=["cloudcoverpercentage"], ascending=True)[
                            "uuid"
                        ]
                    )[: int(max_image_number / len(rel_orbits))]
                )
            stratified_products = stratified_products[
                stratified_products["uuid"].isin(uuids)
            ]

    LOGGER.info(
        f"{len(stratified_products)} Stratified Products ordered by cloud cover percentage."
    )

    return stratified_products


def unzip_downloaded_products(write_path: str, product: str) -> None:
    """
    This function unzips the downloaded SAFE .zip archives to a given path.

    Parameters
    ----------
    write_path: Path to write the unzipped SAFE directory to.
    product: Product name

    Returns
    -------

    """

    try:
        zip_path = os.path.join(write_path, product + ".zip")
        zip_ref = zipfile.ZipFile(zip_path, "r")
        zip_ref.extractall(write_path)
        zip_ref.close()
        os.remove(zip_path)
        LOGGER.info(f"Unzipped {zip_path} to {write_path}")
    except UnboundLocalError:
        LOGGER.info(f"Download for {write_path} failed.")

    return


def check_file_exists(write_path: str, product_title: str) -> bool:
    """
    This function checks whether a product is already downloaded and currently
    exists on the filesystem. If it does, the function returns True, and allows
    for the product download to be skipped within the main image acquisition pipeline.

    Parameters
    ----------
    write_path: Path where the products are to be downloaded to
    product_title: Name of the product

    Returns
    -------
    Boolean - True if product exists, False otherwise
    """

    if os.path.exists(f"{write_path}/{product_title}"):
        LOGGER.info(
            f"Product {product_title} at {write_path} already exists. Skipping."
        )
        return True
    else:
        return False
