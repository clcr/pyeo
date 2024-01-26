import os
import sys
import glob
#import logging
from pyeo import filesystem_utilities
from pyeo import vectorisation


def vector_report_generation(config_path: str, tile: str):
    """
    This function vectorises the Change Report Raster, with the aim of producing 
    shapefiles that can be filtered and summarised spatially, and displayed in a GIS.

    Parameters
    ----------
    config_path : str
        path to pyeo.ini
    tile : str
        Sentinel-2 tile name to process

    Returns
    -------
    output_vector_products : list of str
        list of output vector file names created

    """

    # get the config
    config_dict = filesystem_utilities.config_path_to_config_dict(
        config_path=config_path
    )

    # changes directory to pyeo_dir, enabling the use of relative paths from the config file
    os.chdir(config_dict["pyeo_dir"])
    
    # get other parameters
    epsg = config_dict["epsg"]
    level_1_boundaries_path = config_dict["level_1_boundaries_path"]

    ## setting up the per tile logger
    # get path where the tiles are downloaded to
    tile_directory_path = config_dict["tile_dir"]
    # check for and create the folder structure pyeo expects
    individual_tile_directory_path = os.path.join(tile_directory_path, tile)
    # get the logger for this tile
    tile_log = filesystem_utilities.init_log_acd(
        log_path=os.path.join(individual_tile_directory_path, "log", tile + "_log.txt"),
        logger_name=f"pyeo_tile_{tile}_log",
    )


    # Matt: get the report raster from the previous functions
    # for parallelism reasons, the report path cannot be passed to this function
    # so we run the report glob again

    # get all report.tif file names that are within the root_dir with search pattern
    # from the probabilities subdirectory ...
    report_tif_pattern = f"{os.sep}output{os.sep}probabilities{os.sep}report*.tif"
    search_pattern = f"{tile}{report_tif_pattern}"

    change_report_paths = glob.glob(
        os.path.join(config_dict["tile_dir"], search_pattern)
    )

    # ... and from the report_image_dir subdirectory
    report_tif_pattern = f"{os.sep}output{os.sep}report_image{os.sep}report*.tif"
    search_pattern = f"{tile}{report_tif_pattern}"

    for g in glob.glob(os.path.join(config_dict["tile_dir"], search_pattern)):
        change_report_paths.append(g)
        tile_log.info(f"{g}")

    if len(change_report_paths) == 0:
        tile_log.error("No change report path(s) found.")
        sys.exit()
    else:
        tile_log.info("Change report path(s) found:")
        for p in change_report_paths:
            tile_log.info(f"  {p}")

    # use first element from the list of report image files
    #TODO: use the most recent one instead
    change_report_path = change_report_paths[0]
    
    if len(change_report_paths) > 1:
        tile_log.warning("More than one change report paths found.")
        tile_log.info(f"Using report image file: {change_report_path}")
    
    tile_log.info("--" * 20)
    tile_log.info(f"Starting Vectorisation of the Change Report Raster of Tile: {tile}")
    tile_log.info("--" * 20)

    path_vectorised_binary = vectorisation.vectorise_from_band(
        change_report_path=change_report_path,
        band=15,
        log=tile_log
    )
    # was band=6

    path_vectorised_binary_filtered = vectorisation.clean_zero_nodata_vectorised_band(
        vectorised_band_path=path_vectorised_binary,
        log=tile_log
    )

    tile_log.info(f"change_report_path = {change_report_path}")
    tile_log.info(f"shapefile_path = {path_vectorised_binary_filtered}")
    tile_log.info(f"report_band = {5}")

    rb_ndetections_zstats_df = vectorisation.zonal_statistics(
        raster_path=change_report_path,
        shapefile_path=path_vectorised_binary_filtered,
        report_band=5,
        log=tile_log
        )
    # was band=2

    rb_confidence_zstats_df = vectorisation.zonal_statistics(
        raster_path=change_report_path,
        shapefile_path=path_vectorised_binary_filtered,
        report_band=9,
        log=tile_log
    )
    # was band=5

    rb_first_changedate_zstats_df = vectorisation.zonal_statistics(
        raster_path=change_report_path,
        shapefile_path=path_vectorised_binary_filtered,
        report_band=4,
        log=tile_log
    )
    # was band=7

    # table joins, area, lat lon, county
    output_vector_files = vectorisation.merge_and_calculate_spatial(
                                        rb_ndetections_zstats_df=rb_ndetections_zstats_df,
                                        rb_confidence_zstats_df=rb_confidence_zstats_df,
                                        rb_first_changedate_zstats_df=rb_first_changedate_zstats_df,
                                        path_to_vectorised_binary_filtered=path_vectorised_binary_filtered,
                                        write_csv=False,
                                        write_shapefile=True,
                                        write_kml=False,
                                        write_pkl=False,
                                        change_report_path=change_report_path,
                                        log=tile_log,
                                        epsg=epsg,
                                        level_1_boundaries_path=level_1_boundaries_path,
                                        tileid=tile,
                                        delete_intermediates=True,
                                    )

    tile_log.info("---------------------------------------------------------------")
    tile_log.info("Vectorisation of the Change Report Raster complete")
    tile_log.info("---------------------------------------------------------------")

    return(list(output_vector_files))

if __name__ == "__main__":
    # assuming argv[0] is script name, config_path passed as index 1 and tile string as index 2
    vector_report_generation(config_path=sys.argv[1], tile=sys.argv[2])
