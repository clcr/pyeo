from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
import configparser


def sen2_json_query(geojson_path, cloud, start_date, end_date, conf):
    """
    :param geojson_path:
    :param cloud:
    :param start_date:
    :param end_date:
    :param conf:
    :return:
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

   
if __name__ == "__main__":
    #TODO Add fine-grained processing control here
    config = configparser.ConfigParser()
    config.read('conf.ini')
    query = config['query']
    products = sen2_json_query(geojson_path = query['geojson_path'],
                          cloud = query['max_cloud_cover'],
                          start_date = query['start_date'],
                          end_date = query['end_date'],
                          conf = config)
    sen2_download(products, config)

