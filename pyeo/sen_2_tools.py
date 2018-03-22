from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
import configparser
import shapefile


def sen2_json_query(geojson_path, cloud, start_date, end_date, conf):
    """

    Parameters
    ----------
    geojson_path
    cloud
    start_date
    end_date
    conf

    Returns
    -------

    """
    #TODO: Support for .shp libraries
    api = SentinelAPI(conf["sen2"]["user"], conf["sen2"]["pass"], 'https://scihub.copernicus.eu/dhus')
    footprint = geojson_to_wkt(read_geojson(geojson_path))
    products = api.query(footprint,
                        platformname = 'Sentinel-2',
                        cloudcoverpercentage = (0, cloud),
                        date = (start_date, end_date))
    return products


def sen2_download(products, conf):
    #TODO: Specify download location, file resuming.
    api = SentinelAPI(conf["sen2"]["user"], conf["sen2"]["pass"], 'https://scihub.copernicus.eu/dhus')
    api.download_all(products)


def shp_to_geojson(shp_path, outpath = None):
    """
    Converts a shapefile's geometry to a Geojson
    Parameters
    ----------
    shp_path:
        Path to a shapefile folder
    outpath
        If not None, saves the geojson to disk

    Returns
    -------
        An open geojson object to be passed along
    """
    shp = shapefile.Reader(shp_path)
    for shape in shp.iterShapes():



if __name__ == "__main__":
    #TODO Add fine-grained processing control here
    config = configparser.ConfigParser()
    config.read('conf.ini')
    query = config['query']
    products = sen2_json_query(geojson_path=query['geojson_path'],
                               cloud=query['max_cloud_cover'],
                               start_date=query['start_date'],
                               end_date=query['end_date'],
                               conf=config)
    sen2_download(products, config)

