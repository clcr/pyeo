import configparser
from Sen2Search import sen2_shp_query, sen2_json_query, save_query_output
from Sen2Get import sen2_download


def main():
    #TODO Add fine-grained processing control here
    config = configparser.ConfigParser()
    config.read('conf.ini')
    query = config['query']
    if query['aoi_format'] == "shapefile":
        products = sen2_shp_query(shp_path=query['aoi_path'],
                                cloud=query['max_cloud_cover'],
                                start_date=query['start_date'],
                                end_date=query['end_date'],
                                conf=config)
    elif query['aoi_format'] == "geojson":
        products = sen2_json_query(geojson_path=query['aoi_path'],
                                  cloud=query['max_cloud_cover'],
                                  start_date=query['start_date'],
                                  end_date=query['end_date'],
                                  conf=config)
    save_query_output(products)
    sen2_download(products, config["data"]["out_folder"])


if __name__ == "__main__":
    main()

