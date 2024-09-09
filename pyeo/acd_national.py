"""
Functions for running the raster and vector services in the pipeline.

Key functions
-------------

:py:func:`automatic_change_detection_national` works with the initialisation file that provides all the parameters PyEO needs to make raster and vector processing decisions. `See SEPAL pipeline training notebook within the notebooks folder on the GitHub Repository for an explanation of the initialisation file.` 

:py:func:`automatic_change_detection_national` is composed of the following functions:

- :py:func:`acd_initialisation` : 
-   - Initialises the log file and creates the configuration dictionary which passes the configuration parameters to the downstream functions.

- :py:func:`acd_config_to_log` :
-   - Prints the configuration parameters to the main log, the name of which is customised in the `.ini` file.

- :py:func:`acd_roi_tile_intersection` :
-   - Takes a Region of Interest (ROI) and calculates which Sentinel-2 (S2) tiles overlap with the ROI.

These next functions are executed on a *per-tile* basis, i.e. if the ROI of choice covers 2 tiles, then these functions run twice (once per tile).

- :py:func:`acd_integrated_raster` :
-   - This function calls all the raster processes, which:
-   -   - Build a Baseline Composite to compare land cover changes against, by downloading S2 images and calculating the median of these images.
-   -   - Downloads images over the Change Period
-   -   - Classifies the Composite and the Change images using a classifier in ./models/
-   -   - Calculates the change between the from classes and the to classes, for each classified image. This could be changes from forest to bare soil.
-   -   - Creates a Change Report describing the consistency of the class changes, highlighting the changes that PyEO is confident.

- :py:func:`acd_integrated_vectorisation` :
-   - This function calls all the vectorisation processes, which:
-   -   - Vectorises the Change Report, removing any changes observed outside of the ROI.
-   -   - Performs Zonal Statistics on the change polygons.
-   -   - Filters the change polygons based on Counties of Interest.

These functions run on an *all-tile* basis, i.e. they run once for all tiles.

- :py:func:`acd_national_integration` :
-   - If the ROI covers more than one Sentinel-2 tile, then this function will merge the individual vectorised change polygons into one convenient shapefile or kml, :code:`national_geodataframe`.

- :py:func:`acd_national_filtering` :
-   - Applies County and Minimum Area filters to :code:`national_geodataframe`.

Function reference
------------------
"""

import configparser
import glob
import logging
import os
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

import fiona
import geopandas as gpd
import pandas as pd
from pyeo import filesystem_utilities
from pyeo.apps.acd_national import (acd_by_tile_raster,
                                      acd_by_tile_vectorisation)


# acd_national is the top-level function which controls the raster and vector processes for pyeo
def automatic_change_detection_national(config_path) -> None:
    """
    This function:

    - Acts as the singular call to run automatic change detection for all RoI intersecting tiles and then to vectorise and aggregate the results into a national scale shape file of the change alerts suitable for use within QGIS.

    Parameters
    ----------
    config_path : str
        The path to the config file, which is an `.ini`

    Returns
    -------
    None

    """

    # starting acd_initialisation()
    config_dict, acd_log = acd_initialisation(config_path)

    acd_log.info("---------------------------------------------------------------")
    acd_log.info("Starting acd_config_to_log()")
    acd_log.info("---------------------------------------------------------------")

    # echo configuration to log
    acd_config_to_log(config_dict, acd_log)

    acd_log.info("---------------------------------------------------------------")
    acd_log.info("Starting acd_roi_tile_intersection()")
    acd_log.info("---------------------------------------------------------------")

    tilelist_filepath = acd_roi_tile_intersection(config_dict, acd_log)

    if config_dict["do_tile_intersection"]:
        tilelist_filepath = acd_roi_tile_intersection(config_dict, acd_log)

    if config_dict["do_raster"] and config_dict["do_tile_intersection"]:
        acd_log.info("---------------------------------------------------------------")
        acd_log.info("Starting acd_integrated_raster():")
        acd_log.info("---------------------------------------------------------------")

        acd_integrated_raster(config_dict, acd_log, tilelist_filepath, config_path)

    # and skip already existing vectors
    if config_dict["do_vectorise"] and config_dict["do_tile_intersection"] :
        acd_log.info("---------------------------------------------------------------")
        acd_log.info("Starting acd_integrated_vectorisation()")
        acd_log.info("  vectorising each change report raster, by tile")
        acd_log.info("---------------------------------------------------------------")

        acd_integrated_vectorisation(
            log=acd_log, tilelist_filepath=tilelist_filepath, config_path=config_path
        )

    if config_dict["do_integrate"]:
        acd_log.info("---------------------------------------------------------------")
        acd_log.info("Starting acd_national_integration")
        acd_log.info("---------------------------------------------------------------")

        acd_national_integration(
            root_dir=config_dict["tile_dir"],
            log=acd_log,
            epsg=config_dict["epsg"],
            config_dict=config_dict,
            write_kml=True
        )

    if config_dict["do_filter"]:
        acd_log.info("---------------------------------------------------------------")
        acd_log.info("Starting acd_national_filtering")
        acd_log.info("---------------------------------------------------------------")
        if config_dict["counties_of_interest"]:
            acd_national_filtering(log=acd_log, config_dict=config_dict)

    # if config_dict["do_distribution"]:
    #     acd_log.info("---------------------------------------------------------------")
    #     acd_log.info("Starting acd_national_distribution()")
    #     acd_log.info("---------------------------------------------------------------")
        # acd_national_distribution()
        # messaging services to Park Rangers (e.g. WhatsApp, Maps2Me)

    acd_log.info("---------------------------------------------------------------")
    acd_log.info("---                  INTEGRATED PROCESSING END              ---")
    acd_log.info("---------------------------------------------------------------")

    # This is the end of the function


# def acd_composite_update():
#     """
#     This function updates the composite to a new specified start and end date.

#     We could potentially streamline the composite update process by:

#         - Move out of date change images to the composite folder

#             - "Out of Date" = time period parameter e.g. every 3 months

#         - rebuild composite based on whichever .tiffs are within the composite folder.

#     """


############################
# the functions below are those required by acd_national()
############################


def acd_initialisation(config_path):
    """

    This function initialises the .log file, making the log object available.

    Parameters
    ----------
    config_path : str
        The path to the config file, which is an .ini

    Returns
    -------
    config_dict : dict
        A dictionary composed of configuration parameters read from the `.ini` file.
    log : logging.Logger
        A log object

    """

    # build dictionary of configuration parameters
    config_dict = filesystem_utilities.config_path_to_config_dict(config_path)

    # changes directory to pyeo_dir, enabling the use of relative paths from the config file
    os.chdir(config_dict["pyeo_dir"])

    # check that log directory exists and create if not
    if not os.path.exists(config_dict["log_dir"]):
        os.makedirs(config_dict["log_dir"])

    # initialise log file
    log = filesystem_utilities.init_log_acd(
        log_path=os.path.join(config_dict["log_dir"], config_dict["log_filename"]),
        logger_name="pyeo_acd_log",
    )

    # check conda directory exists
    if config_dict["environment_manager"] == "conda":
        conda_boolean = filesystem_utilities.conda_check(config_dict=config_dict, log=log)
        log.info(conda_boolean)
        if not conda_boolean:
            log.error("Conda Environment Directory does not exist")
            log.error("Ensure this exists")
            log.error("now exiting the pipeline")
            sys.exit(1)

    log.info("---------------------------------------------------------------")
    log.info("---                  INTEGRATED PROCESSING START            ---")
    log.info("---------------------------------------------------------------")

    log.info(f"Reading in parameters defined in: {config_path}")
    log.info("---------------------------------------------------------------")

    return config_dict, log


def acd_config_to_log(config_dict: dict, log: logging.Logger) -> None:
    """
    This function echoes the contents of config_dict to the log file.
    It does not return anything.

    Parameters
    ----------

    config_dict : dict
        config_dict variable

    log : logging.Logger
        log variable

    Returns
    -------

    None

    """
    for key in config_dict:
        value = config_dict[key]
        found = False # flag whether the key was found in and logged
        if key == "pyeo_dir":
            log.info(f"Pyeo Working Directory is   : {config_dict['pyeo_dir']}")
            log.info(f"  Integrated Directory           : {config_dict['integrated_dir']}")
            log.info(f"  ROI Directory for image search : {config_dict['roi_dir']}")
            log.info(f"  Geometry Directory for admin shapefile : {config_dict['geometry_dir']}")
            log.info(
                f"  Path to the Admin Boundaries for Vectorisation : {config_dict['level_1_boundaries_path']}"
            )
            found = True
        if key == "tile_dir":
            log.info(f"Main Tile Directory for tile subdirs : {config_dict['tile_dir']}")
            found = True
        if key == "environment_manager":
            log.info(f"    Environment Manager to use is : {value}")
            if config_dict["environment_manager"] == "conda":
                log.info(
                    f"The Conda Environment specified in .ini file is :  {config_dict['conda_env_name']}"
                    )
            found = True
        if key == "epsg":
            log.info(f"  EPSG code for output map projection: {config_dict['epsg']}")
            found = True
        if key == "do_parallel":
            log.warning("   --do_parallel is depracated")
            found = True
        if key == "do_dev":
            log.warning("   --do_dev is depracated")
            found = True
        if key == "do_tile_intersection" and value:
            log.info("  --do_tile_intersection enables Sentinel-2 tile " +
                     "intersection with region of interest (ROI).")
            found = True
        if key == "do_raster":
            log.warning("   --do_raster is depracated")
            found = True
        if key == "do_all" and value:
            log.info("  --do_all enables all processing steps")
            found = True
        if key == "build_composite" and value:
            log.info("  --build_composite makes a baseline image composite")
            log.info(f"    --download_source = {config_dict['download_source']}")
            log.info(f"      composite start date :  {config_dict['composite_start']}")
            log.info(f"      composite end date   : {config_dict['composite_end']}")
            found = True
        if key == "do_download" and value:
            log.info("  --download of change detection images enabled")
            found = True
        if key == "download_source":
            if config_dict["download_source"] == "scihub":
                log.info("scihub API is selected as download source.")
            else:
                if config_dict["download_source"] == "dataspace":
                    log.info("dataspace selected as download source for the Copernicus"+
                             " Data Space Ecosystem.")
                else:
                    log.error(f"{config_dict['download_source']} is selected as "+
                              "download source.")
                    log.error("Use 'dataspace' instead to access the Copernicus Data "+
                              "Space Ecosystem.")
                    sys.exit(1)
            log.info(
                f"    Faulty Granule Threshold: {config_dict['faulty_granule_threshold']}"
            )
            found == True
        if key == "sen2cor_path":
            log.info(f"Path to Sen2Cor is   : {config_dict['sen2cor_path']}")
            found = True
        if key == "do_classify" and value:
            log.info(
                "  --do_classify applies the random forest model and creates "+
                "classification layers"
            )
            found = True
        if key == "bands":
            log.info(f"  List of image bands: {config_dict['bands']}")
            found = True
        if key == "class_labels":
            log.info("  List of class labels:")
            for c, this_class in enumerate(config_dict["class_labels"]):
                log.info(f"    {c + 1} : {this_class}")
            log.info(
                f"Detecting changes from any of the classes: {config_dict['from_classes']}"
            )
            log.info(f"                    to any of the classes: {config_dict['to_classes']}")
            found = True
        if key == "model_path":
            log.info(f"Model used: {config_dict['model_path']}")
            found = True
        if key == "build_prob_image" and value:
            log.info("  --build_prob_image saves classification probability layers")
            found = True
        if key == "do_change" and value:
            log.info("  --do_change produces change detection layers and "+
                     "report images")
            log.info(f"    --download_source = {config_dict['download_source']}")
            log.info(f"      change start date : {config_dict['start_date']}")
            log.info(f"      change end date   : {config_dict['end_date']}")
            found = True
        if key == "do_update":
            log.warning("   --do_update is depracated")
            found = True
        if key == "do_vectorise" and value:
            log.info("  --do_vectorise produces vector files from raster "+
                     "report images")
            found = True
        if key == "do_delete_existing_vector" and value:
            log.info(
                "  --do_delete_existing_vector, when vectorising the change report rasters,"
            )
            log.info(
                "    existing vectors files will be deleted and new vector files created."
            )
            found = True
        if key == "do_integrate" and value:
            log.info("  --do_integrate merges vectorised reports together")
            found = True
        if key == "counties_of_interest":
            log.info("  --counties_of_interest")
            log.info("        Counties to filter the national geodataframe:")
            for n, county in enumerate(config_dict["counties_of_interest"]):
                log.info(f"        {n}  :  {county}")
            log.info("  --minimum_area_to_report_m2")
            log.info(
                "    Only Change Detections > "+
                f"{config_dict['minimum_area_to_report_m2']} square metres "+
                "will be reported"
            )
            found = True
        if key == "do_quicklooks" and value:
            log.info("  --quicklooks saves image quicklooks for visual quality checking")
            found = True
        if key == "do_delete" and value:
            log.info("  --do_delete removes downloaded images and intermediate"+
                     "    image products after processing to free up disk space.")
            log.info(
                "    Overrides --zip for the files for deletion. WARNING! FILE LOSS!"
            )
            found = True
        if key == "do_zip" and value:
            log.info(
                "  --do_zip archives downloaded and intermediate image products"+
                "    to reduce disk space usage."
            )
            found = True
        if not found:
            log.info(f"  {key} :  {value}")
    log.info("-----------------------------------------------------------")
    return


def acd_roi_tile_intersection(config_dict: dict, log: logging.Logger) -> str:
    """

    This function:

    - accepts a Region of Interest (RoI) (specified by config_dict) and writes 
      a tilelist.txt of the Sentinel-2 tiles that the RoI covers.

    - the tilelist.csv is then used to perform the tile-based processes, 
      raster and vector.

    Parameters
    ----------
    config_dict : dict
        Dictionary of the Configuration Parameters specified in the `.ini`

    log : logging.Logger
        Logger object

    Returns
    -------
    tilelist_filepath : str
        Filepath of a .csv containing the list of tiles on which to perform 
          raster processes

    """

    #log.info("Checking which Sentinel-2 tiles overlap with the ROI (region of interest)")

    # roi_filepath is relative to pyeo_dir supplied in pyeo.ini
    roi_filepath = os.path.join(config_dict["roi_dir"], config_dict["roi_filename"])
    roi = gpd.read_file(roi_filepath)
    
    # check if s2_tiles exists (it should, as is provided with git clone pyeo)
    s2_tiles_filepath = os.path.join(config_dict["geometry_dir"], config_dict["s2_tiles_filename"])
    s2_tile_geometry = gpd.read_file(s2_tiles_filepath)

    # change projection
    roi = roi.to_crs(s2_tile_geometry.crs)

    # intersect roi with s2 tiles to return
    intersection = s2_tile_geometry.sjoin(roi)
    tiles_list = list(intersection["Name"].unique())
    tiles_list.sort()
    log.info(f"The provided ROI intersects with {len(tiles_list)} Sentinel-2 tiles:")
    for n, this_tile in enumerate(tiles_list):
        log.info("  {} : {}".format(n + 1, this_tile))

    tilelist_filepath = os.path.join(config_dict["roi_dir"], "tilelist.csv")
    #log.info(f"Writing Sentinel-2 tile list to : {tilelist_filepath}")

    try:
        tiles_list_df = pd.DataFrame({"tile": tiles_list})
        tiles_list_df.to_csv(tilelist_filepath, header=True, index=False)
    except:
        log.error(f"Could not write to {tilelist_filepath}")
    #log.info("Finished ROI tile intersection")

    return tilelist_filepath


def acd_integrated_raster(
    config_dict: dict, 
    log: logging.Logger,
    tilelist_filepath: str,
    config_path: str
    ) -> None:
    """
    This function:

    - checks whether `tilelist.csv` exists before running acd_by_tile_raster for each tile

    - calls `acd_by_tile_raster` for all active tiles

    Parameters
    ----------
    config_dict : dict
        Dictionary of the Configuration Parameters specified in the `.ini`

    log : logging.Logger
        Logger object

    tilelist_filepath : str
        Filepath of a .csv containing the list of tiles on which to perform raster processes

    config_path : str
        filepath of the config (pyeo.ini) for `acd_by_tile_raster`, this is present to enable the parallel processing option.

    Returns
    -------
    None
    """

    ####### reads in tilelist.txt, then runs acd_per_tile_raster, per tile
    # check if tilelist_filepath exists
    if os.path.exists(tilelist_filepath):
        try:
            tilelist_df = pd.read_csv(tilelist_filepath)
        except:
            log.error(f"Could not open {tilelist_filepath}")
    else:
        log.error(
            f"{tilelist_filepath} does not exist, check that you ran the acd_roi_tile_intersection beforehand"
        )
        sys.exit(1)

    # check and read in credentials for downloading Sentinel-2 data
    credentials_path = config_dict["credentials_path"]
    if os.path.exists(credentials_path):
        try:
            conf = configparser.ConfigParser(allow_no_value=True)
            conf.read(credentials_path)
            credentials_dict = {}
            credentials_dict["sent_2"] = {}
            credentials_dict["sent_2"]["user"] = conf["sent_2"]["user"]
            credentials_dict["sent_2"]["pass"] = conf["sent_2"]["pass"]
        except:
            log.error(f"Could not open {credentials_path}")
    else:
        log.error(
            f"{credentials_path} does not exist, did you write the correct filepath in pyeo.ini?"
        )

    # check for and create tile directory, which will hold the tiles
    tile_directory = config_dict["tile_dir"]
    if not os.path.exists(tile_directory):
        log.info(
            f"The following directory for tile_dir did not exist, creating it at: {tile_directory}"
        )
        os.makedirs(tile_directory)
    else:
        pass

    ######## run acd_by_tile_raster
    for _, tile in tilelist_df.iterrows():
        # try:
        log.info(f"Starting ACD Raster Processes for Tile :  {tile[0]}")

        if not config_dict["do_parallel"]:
            acd_by_tile_raster.acd_by_tile_raster(config_path, tile[0])
            log.info(f"Finished ACD Raster Processes for Tile :  {tile[0]}")
            log.info("")

        if config_dict["do_parallel"]:
            # Launch an instance for this tile using qsub for parallelism

            # Setup required paths
            ## TODO Update to obtain these from config file on config_path and match variable names to standardise on those in pyeo.ini)        a config_dict containing `conda_directory` and `conda_env_name`
            ## TODO Change print statments to log.info
            ## (Temporary test paths point to a test function 'apps/automation/_random_duration_test_program.py' that returns after a short random time delay)
            data_directory = config_dict[
                "tile_dir"
            ]  # '/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/automation' #
            sen2cor_path = config_dict[
                "sen2cor_path"
            ]  # '/home/i/ir81/Sen2Cor-02.09.00-Linux64'  #

            conda_directory = config_dict["conda_directory"]
            # conda_environment_directory = "/home/i/ir81/miniconda3/envs"  # config_dict["conda_env_directory"] (NOTE: Doesn't exist in ini file yet)
            conda_environment_name = config_dict["conda_env_name"]  # 'pyeo_env'  #
            conda_environment_path = os.path.join(
                conda_directory, conda_environment_name
            )
            code_directory = config_dict[
                "pyeo_dir"
            ]  # '/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo' #
            python_executable = "pyeo/apps/acd_national/acd_by_tile_raster.py"  # 'apps/automation/_random_duration_test_program.py'  #
            wall_time_hours = config_dict["wall_time_hours"]
            watch_time_hours = config_dict["watch_time_hours"]
            watch_period_seconds = config_dict["watch_period_seconds"]
            qsub_processor_options = config_dict["qsub_processor_options"]

            # qsub_options = f"walltime=00:{wall_time_hours}:00:00,{qsub_processor_options}"

            qsub_options = f"walltime=00:{wall_time_hours}:00:00,{qsub_processor_options}"  # 'walltime=00:00:02:00,nodes=1:ppn=16,vmem=64Gb'
            # config_directory = '/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo' # '/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo'
            # config_filename = 'pyeo.ini'
            # config_path = os.path.join(config_directory, config_filename)
            automation_script_path = os.path.join(
                code_directory, "pyeo/apps/automation/automate_launch.sh"
            )

            tile_name = tile[0]
            new_line = "\n"

            log.info(
                f"automation_test.py: Checking if tile {tile_name} is already being processed and, if so, deleting current process to avoid possible conflicts"
            )
            df = qstat_to_dataframe()
            if not df.empty:
                current_tile_processes_df = df[df["Name"] == tile_name]
                # log.info('current_tile_processes_df')
                # log.info(current_tile_processes_df)
                for index, p in current_tile_processes_df.iterrows():
                    log.info(p)
                    if p["Status"] in ["Q", "R"]:
                        job_id = p["JobID"].split(".")[0]
                        log.info(f"{new_line}Deleting job: {job_id} {new_line}")
                        os.system(f"qdel {job_id}")

            log.info(
                f"automation_test.py: Preparing to launch tile processing of tile {tile_name}"
            )
            # log.info(f'config_path: {type(config_path)}')
            # log.info(f'config_path: {config_path[0]}')

            python_launch_string = f"cd {data_directory}; module load python; source activate {conda_environment_path}; SEN2COR_HOME={sen2cor_path}; export SEN2COR_HOME; python {os.path.join(code_directory, python_executable)} {config_path} {tile_name}"
            qsub_launch_string = f'qsub -N {tile_name} -o {os.path.join(data_directory, tile_name + "_o.txt")} -e {os.path.join(data_directory, tile_name + "_e.txt")} -l {qsub_options}'
            shell_command_string = f"{automation_script_path} '{python_launch_string}' '{qsub_launch_string}'"

            log.info(f"python_launch_string: {python_launch_string}{new_line}")
            log.info(f"qsub_launch_string: {qsub_launch_string}{new_line}")
            log.info(f"shell_command_string: {shell_command_string}{new_line}")

            result = subprocess.run(
                shell_command_string, capture_output=True, text=True, shell=True
            )
            log.info(
                f" Subprocess launched for tile {tile_name}, return value: {result.stdout}"
            )
        # except:
        # log.error(f"Could not complete ACD Raster Processes for Tile: {tile[0]}")

    # if parallel, monitor parallel processes once all have been launched
    if config_dict["do_parallel"]:
        log.info("automation_test.py: subprocess launching completed")
        log.info("automation_test.py: subprocess monitoring started")

        # TODO Move these parameters into the config file and change monitoring loop to a while loop
        # TODO Set maximum_monitoring_period_raster to greater than walltime ( > maximum expected processing time for a tile)
        # monitoring_cycles = 24 * 60  # 24 hours
        # monitoring_period_seconds = 60

        # monitoring_period_seconds = watch_time_hours * 60 * 60
        watch_cycles = int((watch_time_hours * 60 * 60) / watch_period_seconds)

        end_monitoring = False
        for i in range(watch_cycles):
            time.sleep(watch_period_seconds)
            log.info(
                f"automation_test.py: Checking which tiles are still being processed after {i * watch_period_seconds} seconds"
            )

            df = qstat_to_dataframe()
            if not df.empty and end_monitoring == False:
                active_process_count = 0
                for _, tile in tilelist_df.iterrows():
                    current_tile_processes_df = df[df["Name"] == tile[0]]
                    # log.info('current_tile_processes_df')
                    # log.info(current_tile_processes_df)
                    for index, p in current_tile_processes_df.iterrows():
                        if p["Status"] in ["Q", "R"]:  # ['Q', 'R', 'C']):
                            job_id = p["JobID"].split(".")[0]
                            log.info(
                                f'Tile {tile[0]} still running pid: {job_id}, status; {p["Status"]} '
                            )
                            active_process_count += 1
                if active_process_count == 0:
                    end_monitoring = True
            else:
                log.info(
                    "All tiles have been processed.. continuing to next pipeline stage"
                )
                break

        log.info("automation_test.py subprocesses completed")


def qstat_to_dataframe():
    """

    This function:

    - Runs the pbs qstat command as a subprocess,
    - Captures the stdout
    - Parses the output into a dataframe summarising all active processes to allow monitoring of parallel processes launched when in do_parallel mode.

    Parameters
    ----------
    None

    Returns
    -------
    Pandas dataframe with one row for each active process and columns for 'JobID', 'Name', 'User', 'TimeUsed', 'Status', 'Queue'
    """

    # Run qstat command and capture the stdout
    result = subprocess.run(["qstat"], capture_output=True)
    # Decode the byte string into a regular string
    output = result.stdout.decode("utf-8")
    # Split the output into lines and remove any empty lines
    lines = output.split("\n")
    lines = [line.strip() for line in lines if line.strip()]

    if len(output) > 0:
        # Extract the header and data rows
        header = lines[0].split()
        data_rows = [line.split() for line in lines[2:]]
        # Create the pandas DataFrame setting colum names manually to match qstat output
        df = pd.DataFrame(
            data_rows, columns=["JobID", "Name", "User", "TimeUsed", "Status", "Queue"]
        )
        return df
    else:
        return pd.DataFrame()  # Return an empty dataframe is no output from qstat


def acd_integrated_vectorisation(
    log: logging.Logger,
    tilelist_filepath: str,
    config_path: str
) -> None:
    """

    This function:

        - Vectorises the change report raster by calling acd_by_tile_vectorisation for all active tiles

    Parameters
    ----------
    log : logging.Logger
        The logger object

    tilelist_filepath : str
        A filepath of a `.csv` containing the tiles to vectorise, is used for sorting the tiles so they are vectorised in the order reported by `acd_roi_tile_intersection()`.

    config_path : str
        path to pyeo.ini


    Returns
    -------
    None

    """

    import glob
    import os

    config_dict = filesystem_utilities.config_path_to_config_dict(
        config_path=config_path
    )
    
    # changes directory to pyeo_dir, enabling the use of relative paths from the config file
    os.chdir(config_dict["pyeo_dir"])
    
    # check if tilelist_filepath exists, open if it does, exit if it doesn't
    if os.path.exists(tilelist_filepath):
        try:
            tilelist_df = pd.read_csv(tilelist_filepath)
        except:
            log.error(f"Could not open {tilelist_filepath}")
    else:
        log.error(
            f"{tilelist_filepath} does not exist, check that you ran the acd_roi_tile_intersection beforehand"
        )
        log.error("exiting pipeline")
        sys.exit(1)

    log.info(f'Active tiles for vectorisation: {tilelist_df}')

    # get all report.tif that are within the root_dir with search pattern
    tiles_name_pattern = "[0-9][0-9][A-Z][A-Z][A-Z]"
    report_tif_pattern = f"{os.sep}output{os.sep}probabilities{os.sep}report*.tif"
    search_pattern = f"{tiles_name_pattern}{report_tif_pattern}"

    tiles_paths = glob.glob(os.path.join(config_dict["tile_dir"], search_pattern))
    log.info(f'Report files found in tiles folders: {tiles_paths}')

    # only keep filepaths which match tilelist
    matching_filepaths = []

    for filepath in tiles_paths:
        # log.info(f'Filepath Field: {filepath.split(os.sep)[-1].split("_")[2]}')
        # if (tilelist_df["tile"].str.contains(filepath.split({os.sep})[-1].split("_")[2]).any()):
        if (tilelist_df["tile"].str.contains(filepath.split(os.sep)[-1].split("_")[2]).any()):
            matching_filepaths.append(filepath)

    # sort filepaths in ascending order
    sorted_filepaths = sorted(matching_filepaths)
    if len(sorted_filepaths) == 0:
        log.error("There are no change reports to vectorise, here are some pointers:")
        log.error("    Ensure the raster processing pipeline has successfully run and completed ")
        log.error("    Ensure tile_dir has been specified correctly in pyeo.ini")
        log.error("Now exiting the vector pipeline")
        sys.exit(1)

    # log the filepaths to vectorise
    log.info(f"There are {len(sorted_filepaths)} Change Report Rasters to vectorise, these are:")
    for n, tile_path in enumerate(sorted_filepaths):
        log.info(f"{n+1}  :  {tile_path}")
        log.info("---------------------------------------------------------------")

    # vectorise per path logic
    for report in sorted_filepaths:
        if config_dict["do_delete_existing_vector"]:
            # get list of existing report files in report path
            log.info(
                "do_delete_existing_vector flag = True: Deleting existing vectorised change report shapefiles, pkls and csvs"
            )
            directory = os.path.dirname(report)
            report_shp_pattern = f"{os.sep}report*"
            search_shp_pattern = f"{directory}{report_shp_pattern}"
            existing_files = glob.glob(search_shp_pattern)

            # exclude .tif files from the delete list
            files_to_remove = [
                file for file in existing_files if not file.endswith(".tif")
            ]

            for file in files_to_remove:
                try:
                    os.remove(file)
                except:
                    log.error(f"Could not delete : {file}, skipping")
        # find tile string for the report to be vectorised
        # tile = sorted_filepaths[0].split(os.sep)[-1].split("_")[-2]
        tile = report.split(os.sep)[-1].split("_")[-2]

        if not config_dict["do_parallel"]:
            # try:
            acd_by_tile_vectorisation.vector_report_generation(config_path, tile)
            # except:
            #   log.error(f"Sequential Mode: Failed to vectorise {report}, moving on to the next")
        if config_dict["do_parallel"]:
            try:
                subprocess.run()
            except:
                log.error(
                    f"Parallel Mode: Failed to vectorise {report}, moving on to the next"
                )

    log.info("---------------------------------------------------------------")
    log.info("---------------------------------------------------------------")
    log.info("National Vectorisation of the Change Reports Complete")
    log.info("---------------------------------------------------------------")
    log.info("---------------------------------------------------------------")

    return


def acd_national_integration(
    root_dir: str,
    log: logging.Logger,
    epsg: int,
    config_dict: dict,
    write_kml: bool
) -> None:
    """

    This function:

    - globs to find 1 report_xxx.pkl file per tile in output/probabilities,
    - from which to read in a Pandas DataFrame of the vectorised changes
    - then concatenate DataFrame for each tile to form a national change event DataFrame
    - save to disc in /integrated/acd_national_integration.pkl

    Parameters
    ----------
    root_dir : str
        string representing the path to the root directory.
    log : logging.Logger
        a Logger object

    epsg : int
        integer code of the espg appropriate for the study area.

    config_dict : dict
        dictionary of configuration parameters read from the initialisation file.

    write_kml : bool
        writes to `.kml` if True

    Returns
    -------
    None

    """

    # get tile name pattern and report shapefile pattern for glob
    tiles_name_pattern = "[0-9][0-9][A-Z][A-Z][A-Z]"
    report_shp_pattern = f"{os.sep}output{os.sep}probabilities{os.sep}report*.shp"
    search_pattern = f"{tiles_name_pattern}{report_shp_pattern}"

    # glob through passed directory, return files matching the two patterns
    vectorised_paths = glob.glob(os.path.join(root_dir, search_pattern))
    log.info(
        f"Number of change Report Shapefiles to integrate  :  {len(vectorised_paths)}"
    )
    log.info("Paths of shapefiles to integrate are:")
    for number, path in enumerate(sorted(vectorised_paths)):
        log.info(f"{number} : {path}")

    # initialise empty geodataframe
    merged_gdf = gpd.GeoDataFrame()

    # specify roi path
    roi_filepath = os.path.join(config_dict["roi_dir"], config_dict["roi_filename"])

    # logic if roi path does not exist
    if not os.path.exists(roi_filepath):
        log.error("Could not open ROI, filepath does not exist")
        log.error(f"Exiting acd_national(), ensure  {roi_filepath}  exists")
        sys.exit(1)

    # read in ROI, reproject
    log.info("Reading in ROI")
    roi = gpd.read_file(roi_filepath)
    # log.info(f"Ensuring ROI is of EPSG  :  {epsg}")
    roi = roi.to_crs(epsg)

    # for each shapefile in the list of shapefile paths, read, filter and merge
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        for vector in sorted(vectorised_paths):
            try:
                # read in shapefile, reproject
                log.info(f"Reading in change report shapefile   :  {vector}")
                shape = gpd.read_file(vector)
                # log.info(f"Ensuring change report shapefile is of EPSG  :  {epsg}")
                shape = shape.to_crs(epsg)

                # spatial filter intersection of shapefile with ROI
                log.info(f"Intersecting {vector} with {roi_filepath}")
                intersected = shape.overlay(roi, how="intersection")

                # join the two gdfs
                merged_gdf = pd.concat([merged_gdf, intersected], ignore_index=True)
                log.info(f"Intersection: Success")

                # explode to convert any multipolygons created from intersecting to individual polygons
                merged_gdf = merged_gdf.explode(index_parts=False)

                # recompute area
                merged_gdf["area"] = merged_gdf.area

                log.info(
                    f"Integrated geodataframe length is currently  :  {len(merged_gdf['area'])}"
                )
            except:
                log.error(f"failed to merge geodataframe: {vector}")

    # write integrated geodataframe to shapefile
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        try:
            out_path = f"{os.path.join(root_dir, 'national_geodataframe.shp')}"
            log.info(
                f"Merging loop complete, now writing integrated shapefile to {out_path}"
            )
            merged_gdf.to_file(filename=out_path)
        except:
            log.error(f"failed to write output to shapefile at :  {out_path}")

        if write_kml:
            try:
                kml_out_path = f"{os.path.join(root_dir, 'national_geodataframe.kml')}"
                fiona.supported_drivers['KML'] = 'rw'
                merged_gdf.to_file(kml_out_path, driver='KML')
            except:
                log.error(f"failed to write output to kml, at : {kml_out_path}")

    log.info(f"Integrated GeoDataFrame written to : {out_path}")
    if write_kml:
       log.info(f"Integrated GeoDataFrame written to : {kml_out_path}")
    log.info("---------------------------------------------------------------")
    log.info("---------------------------------------------------------------")
    log.info("National Integration of the Vectorised Change Reports Complete")
    log.info("---------------------------------------------------------------")
    log.info("---------------------------------------------------------------")

    return


def acd_national_filtering(log: logging.Logger, config_dict: dict):
    """
    This function:

    - Applies filters to the national vectorised change report, as specified in the pyeo.ini
    - The current filters are Counties and Minimum Area.

    Parameters
    ----------

    log : logging.Logger
        The logger object
    config_dict : dict
        config dictionary containing runtime parameters

    Returns
    -------
    None
    """

    # switch gdal and proj installation to geopandas'
    # gdal_switch(installation="geopandas", config_dict=config_dict)

    # find national_geodataframe
    search_pattern = "national_geodataframe.shp"
    national_change_report_path = glob.glob(
        os.path.join(config_dict["tile_dir"], search_pattern)
    )[0]

    # read in national geodataframe created before by the integrated step
    if os.path.exists(national_change_report_path):
        national_gdf = gpd.read_file(national_change_report_path)
    else:
        log.error(
            f"national geodataframe does not exist, have you set 'do_integrate' to True in pyeo.ini?"
        )
        sys.exit(1)

    # create a query based on the county list provided in pyeo.ini
    query_values = " or ".join(
        f"County == '{county_name}'"
        for county_name in config_dict["counties_of_interest"]
    )

    # apply the county query filter
    filtered = national_gdf.query(query_values)

    # create a query based on minimum area provided in pyeo.ini
    query_values = f"area > {config_dict['minimum_area_to_report_m2']}"

    # apply the minimum area filter
    filtered = filtered.query(query_values)

    # write filtered geodataframe to shapefile
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        try:
            out_path = f"{os.path.join(config_dict['tile_dir'], 'national_geodataframe_filtered.shp')}"
            log.info(
                f"Filtering complete, now writing filtered national shapefile to :"
            )
            log.info(f"        {out_path}")
            filtered.to_file(filename=out_path)
        except:
            log.error(f"failed to write output at :  {out_path}")

    # "reset" gdal and proj installation back to default (which is GDAL's GDAL and PROJ_LIB installation)
    # gdal_switch(installation="gdal_api", config_dict=config_dict)

    return

    #     """

    #     This function:
    #        - Generates a QGIS Project file using pyQGIS
    #        - Generates QGIS Spatial Bookmarks (.xml) from a filtered dataframe for import into QGIS
    #        - Import Vectorised Change Report
    #        - Import ROI (with Names)
    #        - Import Country Boundaries
    #        - Import County Boundaries
    #        - Import Sentinel-2 Tile Boundaries


    #     """

    # def acd_national_manual_validation():
    #     """

    #     This function:
    #         - is a placeholder for manual validation to assess each event, flagging for on the ground observation

    #     """
    #     pass

    # def acd_national_distribution():
    #     """

    #     This function:
    #         - Once the user is happy and has verified the change alerts, Maps.Me and WhatsApp messages are sent from this function.

    #     """
    #     pass

    # ############################

    # def acd_per_tile_raster():
    #     """

    #     This function:

    #         - queries available Sentinel-2 imagery that matches the tilelist and environmental parameters specified in the pyeo.ini file

    #         - for the composite, this downloads Sentinel-2 imagery, applies atmospheric correction (if necessary), converts from .jp2 to .tif and applies cloud masking

    #         - for the change images, this downloads Sentinel-2 imagery, applies atmospheric correction (if necessary), converts from .jp2 to .tif and applies cloud masking

    #         - classifies the composite and change images

    #         - performs change detection between the classified composite and the classified change images, searching for land cover changes specified in the from_classes and to_classes in the pyeo.ini

    #         - outputs a change report raster with dates land cover change

    #     """
    # create tiles folder structure
    # log.info("\nCreating the directory structure if not already present")

    # filesystem_utilities.create_folder_structure_for_tiles(tile_dir)

    # try:


#   # 17.04.23 uncomment the below variables when we have written the code that needs them
#         change_image_dir = os.path.join(tile_dir, r"images")
#         l1_image_dir = os.path.join(tile_dir, r"images/L1C")
#         l2_image_dir = os.path.join(tile_dir, r"images/L2A")
#         l2_masked_image_dir = os.path.join(tile_dir, r"images/cloud_masked")
#         categorised_image_dir = os.path.join(tile_dir, r"output/classified")
#         probability_image_dir = os.path.join(tile_dir, r"output/probabilities")
#         sieved_image_dir = os.path.join(tile_dir, r"output/sieved")
#         composite_dir = os.path.join(tile_dir, r"composite")
#         composite_l1_image_dir = os.path.join(tile_dir, r"composite/L1C")
#         composite_l2_image_dir = os.path.join(tile_dir, r"composite/L2A")
#         composite_l2_masked_image_dir = os.path.join(tile_dir, r"composite/cloud_masked")
#         quicklook_dir = os.path.join(tile_dir, r"output/quicklooks")

# if arg_start_date == "LATEST":
#     report_file_name = [f for f in os.listdir(probability_image_dir) if os.path.isfile(f) and f.startswith("report_") and f.endswith(".tif")][0]
#     report_file_path = os.path.join(probability_image_dir, report_file_name)
#     after_timestamp  = pyeo.filesystem_utilities.get_change_detection_dates(os.path.basename(report_file_path))[-1]
#     after_timestamp.strftime("%Y%m%d") # Returns the yyyymmdd string of the acquisition date from which the latest classified image was derived
# elif arg_start_date:
#     start_date = arg_start_date

# if arg_end_date == "TODAY":
#     end_date = dt.date.today().strftime("%Y%m%d")
# elif arg_end_date:
#     end_date = arg_end_date

# except:
#     log.error("failed to initialise log")

#     pass
