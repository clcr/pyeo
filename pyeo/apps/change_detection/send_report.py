"""
send_report
-------------------------------------
An app for sending out summary information on detected changes between
the vectorised change detection report images to users in various ways.
Vectorisation should be done as part of detect_change.py by setting in the ini file:
  do_vectorise = True
It is intended to support WhatsApp and Email at this stage.
It uses some of the ini file parameters but not the do_x flags.
Shapefiles in the reports_dir will be zipped up to avoid sending the same file twice.
"""

from email.message import EmailMessage
import smtplib
import argparse
import configparser
import cProfile
import datetime
import glob
import pandas as pd
import os
from osgeo import gdal
import sys
from pyeo import filesystem_utilities
from pyeo.acd_national import (
    #acd_initialisation,
    acd_config_to_log,
    acd_roi_tile_intersection,
    )
#from pyeo.apps.acd_national import acd_by_tile_vectorisation
import zipfile

gdal.UseExceptions()

def send_report(config_path, tile_id="None"):
    """
    The main function that sends out change reports created by detect_change.py
        between the median composite and the newly downloaded images with the 
        parameters specified in the ini file found in config_path.

    Args:

        config_path : string with the full path to the ini file or config file containing the
                        processing parameters

        tile_id : string with the Sentinel-2 tile ID. If not "None", this ID is used
                        instead of the region of interest file to define the area to be processed

    """

    # read the ini file contents into a dictionary
    configparser.ConfigParser(allow_no_value=True)
    config_dict = filesystem_utilities.config_path_to_config_dict(config_path)

    ##########################################################
    # Initialisation
    ##########################################################
    
    # changes directory to pyeo_dir, enabling the use of relative paths from 
    #    the config file
    os.chdir(config_dict["pyeo_dir"])

    # check that log directory exists and create if not
    if not os.path.exists(config_dict["log_dir"]):
        os.makedirs(config_dict["log_dir"])

    # initialise log file
    log = filesystem_utilities.init_log_acd(
        log_path=os.path.join(config_dict["log_dir"], config_dict["log_filename"]),
        logger_name="pyeo_log",
    )

    # check conda directory exists
    if config_dict["environment_manager"] == "conda":
        conda_boolean = filesystem_utilities.conda_check(config_dict=config_dict, log=log)
        log.info(conda_boolean)
        if not conda_boolean:
            log.error("Conda Environment Directory does not exist.")
            log.error("Ensure this exists before running pyeo.")
            log.error("Now exiting the pipeline.")
            sys.exit(1)

    log.info(f"Config file that controls the processing run: {config_path}")
    log.info("---------------------------------------------------------------")

    acd_config_to_log(config_dict, log)

    try:
        os.chdir(config_dict["pyeo_dir"]) 
        start_date = config_dict["start_date"]
        end_date = config_dict["end_date"]
        if end_date == "TODAY":
            end_date = datetime.date.today().strftime("%Y%m%d")
        email_alerts = config_dict['email_alerts']
        email_list_file = config_dict['email_list_file']
        whatsapp_alerts = config_dict['whatsapp_alerts']
        #whatsapp_list_file = config_dict['whatsapp_list_file']
        #whatsapp_sender = config_dict['whatsapp_sender']
        #composite_start_date = config_dict["composite_start"]
        #composite_end_date = config_dict["composite_end"]
        #cloud_cover = config_dict["cloud_cover"]
        #cloud_certainty_threshold = config_dict["cloud_certainty_threshold"]
        #model_path = config_dict["model_path"]
        tile_dir = config_dict["tile_dir"]
        #sen2cor_path = config_dict["sen2cor_path"]
        #epsg = config_dict["epsg"]
        #bands = config_dict["bands"]
        #resolution = config_dict["resolution_string"]
        #out_resolution = config_dict["output_resolution"]
        #buffer_size = config_dict["buffer_size_cloud_masking"]
        #buffer_size_composite = config_dict["buffer_size_cloud_masking_composite"]
        #max_image_number = config_dict["download_limit"]
        #faulty_granule_threshold = config_dict["faulty_granule_threshold"]
        #download_limit = config_dict["download_limit"]
        #skip_existing = config_dict["do_skip_existing"]
        #sieve = config_dict["sieve"]
        #from_classes = config_dict["from_classes"]
        #to_classes = config_dict["to_classes"]
        #download_source = config_dict["download_source"]
        #if download_source == "scihub":
        #    log.info("scihub API is the download source")
        #if download_source == "dataspace":
        #    log.info("dataspace API is the download source")
        #log.info("Faulty Granule Threshold is set to   : {}".format(
        #        config_dict['faulty_granule_threshold'])
        #        )
        #log.info("    Files below this threshold will not be downloaded")
        log.info("Successfully read the processing arguments")
    except:
        log.error("Could not read the processing arguments from the ini file.")
        sys.exit(1)

    # check and read in credentials for downloading Sentinel-2 data
    credentials_path = config_dict["credentials_path"]
    if os.path.exists(credentials_path):
        try:
            credentials_conf = configparser.ConfigParser(allow_no_value=True)
            credentials_conf.read(credentials_path)
            if email_alerts:
                log.info(f"Reading your email credentials from {credentials_path}")
                email_sender = credentials_conf["email"]["user"]
                email_password = credentials_conf["email"]["pass"]
            if whatsapp_alerts:
                log.info(f"Reading your WhatsApp credentials from {credentials_path}")
                whatsapp_sender = credentials_conf["whatsapp"]["user"]
                whatsapp_password = credentials_conf["whatsapp"]["pass"]
        except:
            log.error("Could not open " + credentials_path)
            log.error("Create the file with your login credentials.")
            sys.exit(1)
    else:
        log.error(credentials_path + " does not exist.")
        log.error("Did you write the correct filepath in the config file?")
        sys.exit(1)

    # start tile processing
    if tile_id == "None":
        # if no tile ID is given by the call to the function, use the geometry file
        #   to get the tile ID list
        tile_based_processing_override = False
        tilelist_filepath = acd_roi_tile_intersection(config_dict, log)
        log.info("Sentinel-2 tile ID list: " + tilelist_filepath)
        tiles_to_process = list(pd.read_csv(tilelist_filepath)["tile"])
    else:
        # if a tile ID is specified, use that and do not use the tile intersection
        #   method to get the tile ID list
        tile_based_processing_override = True
        tiles_to_process = [tile_id]

    if tile_based_processing_override:
        log.info("Tile based processing selected. Overriding the geometry file intersection method")
        log.info("  to get the list of tile IDs.")

    log.info(str(len(tiles_to_process)) + " Sentinel-2 tile reports to process.")

    # iterate over the tiles
    for tile_to_process in tiles_to_process:
        log.info("Sending out the latest reports for Sentinel-2 tile: " + tile_to_process)
        individual_tile_directory_path = os.path.join(tile_dir, tile_to_process)
        log.info(individual_tile_directory_path)

        try:
            filesystem_utilities.create_folder_structure_for_tiles(individual_tile_directory_path)
            #composite_dir = os.path.join(individual_tile_directory_path, r"composite")
            #composite_l1_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"L1C")
            #composite_l2_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"L2A")
            #composite_l2_masked_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"cloud_masked")
            #change_image_dir = os.path.join(individual_tile_directory_path, r"images")
            #l1_image_dir = os.path.join(individual_tile_directory_path, r"images", r"L1C")
            #l2_image_dir = os.path.join(individual_tile_directory_path, r"images", r"L2A")
            #l2_masked_image_dir = os.path.join(individual_tile_directory_path, r"images", r"cloud_masked")
            #categorised_image_dir = os.path.join(individual_tile_directory_path, r"output", r"classified")
            #probability_image_dir = os.path.join(individual_tile_directory_path, r"output", r"probabilities")
            reports_dir = os.path.join(individual_tile_directory_path, r"output", r"reports")
            #quicklook_dir = os.path.join(individual_tile_directory_path, r"output", r"quicklooks")
        except:
            log.error("ERROR: Tile subdirectory paths could not be created")
            sys.exit(1)

        # initialise tile log file
        tile_log_file = os.path.join(
            individual_tile_directory_path, 
            "log", 
            tile_to_process + ".log"
            )
        log.info(f"Redirecting log output to tile log: {tile_log_file}")
        tile_log = filesystem_utilities.init_log_acd(
            log_path=tile_log_file,
            logger_name="pyeo_"+tile_to_process
        )
        tile_log.info("---------------------------------------------------------------")
        tile_log.info(f"---  TILE PROCESSING START: {tile_to_process}                          ---")
        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Sending out information about vectorised reports."
        )

        '''
        from Ciaran, needs writing a vectorisation.send_email function and can then be tested.

        try:
            # hacky; fix later
            last_class_name = fu.sort_by_timestamp(os.listdir(args.class_raster_path))[0].rsplit('.')[0]+".tif"
            last_class_path = os.path.join(args.class_raster_path, last_class_name)
            last_comparison_name = fu.sort_by_timestamp(os.listdir(args.imagery_path))[0].rsplit('.')[0]+".tif"
            last_comparison_path = os.path.join(args.imagery_path, last_comparison_name)
            last_timestamp = re.findall(r"\d{8}T\d{6}", last_class_name)[0]
            kml_path = os.path.join(args.output_kml_path, "{}.kml".format(last_timestamp))
            json_path = os.path.join(args.output_json_path, "{}.json".format(last_timestamp))
            comparison_dir = os.path.join(args.comparison_dir, last_timestamp)
            try:
                os.mkdir(comparison_dir)
            except FileExistsError:
                pass
            kml_root_name = "{}_{}".format(args.region, last_timestamp)
    
            log.info("Creating report for {}".format(last_class_name))
    
            log.info("Creating change report for {} and {}".format(args.class_raster_path, args.imagery_path))
            report_date = datetime.datetime.now()
            change_report = rt.ChangeReport(
                class_raster_path=last_class_path,
                imagery_path=last_comparison_path,
                class_of_interest=args.class_of_interest,
                region=args.region,
                out_comparison_dir=comparison_dir,
                date=report_date,
                max_changes=args.max_polygons,
                filter_map=args.filter_map,
                filter_classes=args.filter_classes,
                log_path=args.log_path
            )
            log.info("Creating change event list")
            change_report.create_change_event_list()
            log.info("Creating comparison images")
            change_report.create_before_and_after_images()
            log.info("Uploading report to AWS")
            change_report.upload_report_to_backend()
            if args.display_bucket_credentials and args.display_layer_pallette:
                log.info("Uploading map to Ecometrica")
                change_report.upload_to_eolabs(args.display_bucket_credentials, args.display_layer_pallette)
            log.info("Sending email")
            change_report.send_email_report(list(args.email_list))
        except Exception:
            log.exception("Fatal error in report with email module")
       '''


        search_term = "report_*" + tile_to_process + "*.shp"

        tile_log.info(
            f"Searching for vectorised change report shapefiles in {reports_dir}" +
            f"\n  containing: {search_term}."
            )

        vector_files = glob.glob(
            os.path.join(reports_dir, search_term)
        )
        
        # zip up all the shapefiles and ancillary files
        zipped_vector_files = []
        paths = [f for f in vector_files if not f.endswith(".zip")]
        for sf in paths:
            # split off the ".shp" file extension
            file_id = sf.split(sep=".")[0]
            files_to_zip = [f for f in os.listdir(reports_dir) if f.startswith(file_id)]
            files_to_zip = [os.path.join(reports_dir, f) for f in files_to_zip]
            zipped_file = os.path.join(reports_dir, file_id + '.zip')
            with zipfile.ZipFile(
                zipped_file, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for f in files_to_zip:
                    zf.write(f, os.path.basename(f))

            if os.path.exists(zipped_file):
                zipped_vector_files.append(zipped_file)
                for f in files_to_zip:
                    os.remove(f)
            else:
                log.error(f"Zipping failed: {zipped_file}")

        tile_log.info(
            f"{len(zipped_vector_files)} report shapefiles found and zipped up."
            )

        if len(zipped_vector_files) == 0:
            tile_log.info("No new forest alert vector files found.")
            tile_log.info("No message will be sent.")
        else:
            if email_alerts and len(zipped_vector_files)>0:            
                email_list_file = open(email_list_file, 'r')
                recipients = email_list_file.readlines()
                
                for r, recipient in enumerate(recipients):
                    # Remove the newline character
                    recipient_name = recipient.strip().split(",")[0]
                    recipient_email = recipient.strip().split(",")[1]
                    tile_log.info(
                        f"Sending email from {email_sender} to {recipient_name} " +
                        f"at {recipient_email}."
                        )
                    for f in zipped_vector_files:
                        message =  [
                           f"Dear {recipient_name},",
                           "",
                           "New pyeo forest alerts have been detected.",
                           f"Time period: from {start_date} to {end_date}",
                           "",
                           f"Vector file: {f}",
                           "",
                           "Please check the individual alerts and consider action " +
                               "for those you want investigating.",
                           "",
                           "Date of sending this email: " +
                           f"{datetime.date.today().strftime('%Y%m%d')}",
                           "",
                           "Best regards,",
                           "",
                           "The pyeo forest alerts team",
                           "DISCLAIMER: The alerts are providing without any warranty.",
                           "IMPORTANT: Do not reply to this email."
                           ]
        
                        subject_line = "New pyeo forest alerts are ready for you "+\
                            f"(Sentinel-2 tile {tile_to_process})"
        
                        #TODO: Enable sending with file attachment or download location
                        email = EmailMessage()
                        email["From"] = email_sender
                        email["To"] = recipient_email
                        email["Subject"] = subject_line
                        email.set_content("\n".join(message))
                        
                        smtp = smtplib.SMTP("smtp-mail.outlook.com", port=587)
                        smtp.starttls()
                        smtp.login(email_sender, email_password)
                        smtp.sendmail(email_sender, recipient_email, email.as_string())
                        smtp.quit()
                tile_log.info(" ")
                tile_log.info("Info on vectorised reports has been emailed to the contact list.")
                tile_log.info(" ")

            if whatsapp_alerts and len(vector_files)>0:
                tile_log.error("WhatsApp alerts have not been implemented yet.")
                #TODO: WhatsApp
                # run a separate script in a different Python environment using pywhatkit
                # os.script("path to bash file")
                # The bash files needs to do the following:
                #   make sure WhatsApp is open and running
                #   conda activate whatsapp_env
                #   python send_whatsapp.py
		
                '''        
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Info on vectorised reports has been sent via WhatsApp to the contact list.")
                tile_log.info("---------------------------------------------------------------")
                tile_log.info(" ")
                '''        
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("---             TILE PROCESSING END                           ---")
        tile_log.info("---------------------------------------------------------------")

        # process the next tile if more than one tile are specified at this point (for loop)

    # after the for loop ends, log that all tiles are complete:
    log.info(" ")
    log.info("---------------------------------------------------------------")
    log.info("---                  ALL PROCESSING END                      ---")
    log.info("---------------------------------------------------------------")
    return tile_dir


if __name__ == "__main__":
    # save runtime statistics of this code
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Reading in config file
    parser = argparse.ArgumentParser(
        description="Sends information on vectorised reports to a list of "+
        "recipients." +
        "Currently only email is implemented. WhatsApp will be added in future."+
        "Options are set in the .ini file and login details in the credentials file."        
    )
    parser.add_argument(
        dest="config_path",
        action="store",
        default=r"pyeo_linux.ini",
        help="A path to a .ini file containing the specification for the job. See "
        "pyeo/pyeo_linux.ini for an example.",
    )
    parser.add_argument(
        "--tile",
        dest="tile_id",
        type=str,
        default="None",
        help="Overrides the geojson location with a" "Sentinel-2 tile ID location",
    )

    args = parser.parse_args()

    tile_dir = send_report(**vars(args))

    profiler.disable()
    f = os.path.join(tile_dir, "send_report")
    i = 1
    if os.path.exists(f+".prof"):
        while os.path.exists(f+"_"+str(i)+".prof"):
            i = i + 1
        f = f+"_"+str(i)
    f = f + ".prof"
    profiler.dump_stats(f)
    print(f"runtime analysis saved to {f}")
    print("Run snakeviz over the profile file to interact with the profile information:")
    print(f">snakeviz {f}")