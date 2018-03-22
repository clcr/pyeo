from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
import configparser
import shapefile
import json
import tempfile


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
        Where to save the shapefile

    Returns
    -------
    If outpath is None, a geojson-like dict

    """
    #Thanks to http://geospatialpython.com/2013/07/shapefile-to-geojson.html
    shp = shapefile.Reader(shp_path)
    fields = shp.fields[1:]
    field_names = [field[0] for field in fields]
    buffer = []
    for sr in shp.shapeRecords():
        atr = dict(zip(field_names, sr.record))
        geom = sr.shape.__geo_interface__
        buffer.append(dict(type="Feature", \
                           geometry=geom, properties=atr))

    if outpath:
        with open(outpath, "w") as geojson:
            geojson.write(json.dumps({"type": "FeatureCollection", \
                                "features": buffer}, indent=2) + "\n")
    else:
        return {"type": "FeatureCollection", "features": buffer}


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

