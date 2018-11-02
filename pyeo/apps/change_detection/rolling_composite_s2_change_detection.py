"""
 Right, let's think this through.
 Step -1: Create initial composite with last date (stored in filename?)
 Step 0: Load composite
 Step 1: Download images from last date in composite until present last date
 Step 2: Preprocess each image
 Step 3: Generate cloud mask for each image
For each preprocessed image:
    Step 4: Build stack with composite
    Step 5: Classify stack
    Step 6: Update composite with last cloud-free pixel based on cloud mask
    Step 7: Update last_date of composite
Step 8:
 """

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import configparser
import argparse
import os
import gdal
import logging
import datetime as dt


def send_email(address_list, subject, message):
    """Sends an email from my email address to everyone in address_list with the given message and subject"""
    log = logging.getLogger(__name__)
    client = boto3.client('ses', region_name="eu-west-1")  # EU central (frankfurt) doesn't supply ses
    for address in address_list:
        try:
            output = client.send_email(
                Source="jfr10@le.ac.uk",
                Destination={
                    "ToAddresses": [
                        address
                    ]
                },
                Message={
                    'Subject': {
                        "Data": subject
                    },
                    "Body": {
                        "Text": {
                            "Data": message
                        }
                    }
                }
            )
        except client.exceptions.MessageRejected:
            log.warning("{} is not verified".format(address))
            continue


def send_email_report(address_list, image_list, s3_paths):
    """Composes and sends an email report based on a list of classified images"""
    # There's probably a better way to do this with formatted emails, but later.
    report = get_digest_data(image_list, s3_paths)
    message = "+++ Images processed: +++\n"
    for image in report["images"]:
        message = message + "Image ID: {}\n".format(image["index"])
        message = message + "Bounding polygon: {}\n".format(image["area"])
        message = message + "Deforested pixels: {}\n".format(image["deforested_pixels"])
        message = message + "S3 address: https://s3.eu-central-1.amazonaws.com/forestsentinel/{}\n".format(image["s3_bucket"])
        message = message + "+++++++\n"
    subject = "Automated change report for: {}".format(str(dt.date.today()))
    send_email(address_list, subject, message)


def get_digest_data(image_paths, s3_paths):
    """Produces a dictionary of useful data from a list of classified images"""
    out = {
        "images": []
    }
    for index, image_path in enumerate(image_paths):
        image = gdal.Open(image_path)
        image_array = image.GetVirtualMemArray()
        image_report = {
            "index": index,
            "area": pyeo.get_raster_bounds(image),
            "deforested_pixels": (image_array == 3).sum(),
            "s3_bucket": s3_paths[index]
        }
        out["images"].append(image_report)
        image_array = None
        image = None
    return out


def verify_emails(address_list):
    """Sends a verification email to all unverified emails in address list"""
    client = boto3.client('ses', region_name="eu-west-1")
    verified_addresses = client.list_verified_email_addresses()['VerifiedEmailAddresses']
    for address in address_list:
        if not verified_addresses.count(address):
            client.verify_email_identity(EmailAddress=address)


def get_email_list(list_path):
    """Returns a list of emails from a text file"""
    with open(list_path, 'r') as list_file:
        list = [address.strip() for address in list_file.readlines()]
    return list


if __name__ == "__main__":

    do_all = True

    # Reading in config file
    parser = argparse.ArgumentParser(description='Automatically detect and report on change')
    parser.add_argument('--conf', dest='config_path', action='store', default=r'change_detection.ini',
                        help="Path to the .ini file specifying the job.")
    parser.add_argument('-d', '--download', dest='do_download', action='store_true', default=False)
    parser.add_argument('-p', '--preprocess', dest='do_preprocess', action='store_true',  default=False)
    parser.add_argument('-b', '--build_composite', dest='build_composite', action='store_true', default=False)
    parser.add_argument('-m', '--merge', dest='do_merge', action='store_true', default=False)
    parser.add_argument('-a', '--mask', dest='do_mask', action='store_true', default=False)
    parser.add_argument('-s', '--stack', dest='do_stack', action='store_true', default=False)
    parser.add_argument('-c', '--classify', dest='do_classify', action='store_true', default=False)
    parser.add_argument('-u', '--update', dest='do_update', action='store_true', default=False)
    parser.add_argument('-r', '--remove', dest='do_delete', action='store_true', default=False)
    parser.add_argument('-n', '--notify', dest='mail_list', action='store')

    args = parser.parse_args()

    # If any processing step args are present, do not assume that we want to do all steps
    if (args.do_download or args.do_preprocess or args.do_merge or args.do_stack or args.do_classify) == True:
        do_all = False

    conf = configparser.ConfigParser()
    conf.read(args.config_path)
    sen_user = conf['sent_2']['user']
    sen_pass = conf['sent_2']['pass']
    project_root = conf['forest_sentinel']['root_dir']
    aoi_path = conf['forest_sentinel']['aoi_path']
    start_date = conf['forest_sentinel']['start_date']
    end_date = conf['forest_sentinel']['end_date']
    log_path = conf['forest_sentinel']['log_path']
    cloud_cover = conf['forest_sentinel']['cloud_cover']
    cloud_certainty_threshold = int(conf['forest_sentinel']['cloud_certainty_threshold'])
    model_path = conf['forest_sentinel']['model']
    sen2cor_path = conf['sen2cor']['path']
    composite_start_date = conf['forest_sentinel']['composite_start']
    composite_end_date = conf['forest_sentinel']['composite_end']

    pyeo.create_file_structure(project_root)
    log = pyeo.init_log(log_path)

    l1_image_dir = os.path.join(project_root, r"images/L1")
    l2_image_dir = os.path.join(project_root, r"images/L2")
    planet_image_dir = os.path.join(project_root, r"images/planet")
    merged_image_dir = os.path.join(project_root, r"images/merged")
    stacked_image_dir = os.path.join(project_root, r"images/stacked")
    catagorised_image_dir = os.path.join(project_root, r"output/categories")
    probability_image_dir = os.path.join(project_root, r"output/probabilities")
    composite_dir = os.path.join(project_root, r"composite")
    composite_l1_image_dir = os.path.join(project_root, r"composite/L1")
    composite_l2_image_dir = os.path.join(project_root, r"composite/L2")
    composite_merged_dir = os.path.join(project_root, r"composite/merged")

    # Check for boto3 and load mailing list
    if args.mail_list:
        try:
            import boto3

        except ImportError:
            print("boto3 must be installed and configured for email notifications")
            mail_list = None

    # Download and build the initial composite. Does not do by default
    if args.build_composite:
        log.info("Downloading for initial composite between {} and {}".format(composite_start_date, composite_end_date))
        composite_products = pyeo.check_for_s2_data_by_date(aoi_path, composite_start_date, composite_end_date, conf)
        pyeo.download_new_s2_data(composite_products, composite_l1_image_dir)
        log.info("Preprocessing composite products")
        pyeo.atmospheric_correction(composite_l1_image_dir, composite_l2_image_dir, sen2cor_path,
                                    delete_unprocessed_image=True)
        log.info("Aggregating composite layers")
        pyeo.aggregate_and_mask_10m_bands(composite_l2_image_dir, composite_merged_dir, cloud_certainty_threshold)
        log.info("Building initial cloud-free composite")
        pyeo.composite_directory(composite_merged_dir, composite_dir)

    # Query and download all images since last composite
    if args.do_download or do_all:
        products = pyeo.check_for_s2_data_by_date(aoi_path, start_date, end_date, conf)
        log.info("Downloading")
        pyeo.download_new_s2_data(products, l1_image_dir)

    # Atmospheric correction
    if args.do_preprocess or do_all:
        log.info("Applying sen2cor")
        pyeo.atmospheric_correction(l1_image_dir, l2_image_dir, sen2cor_path, delete_unprocessed_image=True)

    # Aggregating layers into single image
    if args.do_merge or do_all:
        log.info("Aggregating layers")
        pyeo.aggregate_and_mask_10m_bands(l2_image_dir, merged_image_dir, cloud_certainty_threshold)

    latest_composite_name = \
        [image for image in pyeo.sort_by_s2_timestamp(os.listdir(composite_dir), recent_first=True)
         if image.endswith(".tif")][0]
    latest_composite_path = os.path.join(composite_dir, latest_composite_name)

    images = [image for image in pyeo.sort_by_s2_timestamp(os.listdir(merged_image_dir), recent_first=False)
              if image.endswith(".tif")]

    for image in images:
        log.info("Detecting change for {}".format(image))
        new_image_path = os.path.join(merged_image_dir, image)

        # Stack with composite
        if args.do_stack or do_all:
            log.info("Stacking images with composite")
            new_stack_path = pyeo.stack_old_and_new_images(latest_composite_path, new_image_path, stacked_image_dir)

        # Classify with composite
        if args.do_classify or do_all:
            log.info("Classifying with composite")
            new_class_image = os.path.join(catagorised_image_dir, "class_{}".format(os.path.basename(new_stack_path)))
            new_prob_image = os.path.join(probability_image_dir, "prob_{}".format(os.path.basename(new_stack_path)))
            pyeo.classify_image(new_stack_path, model_path, new_class_image, new_prob_image, num_chunks=10)

        # Update composite
        if args.do_update or do_all:
            log.info("Updating composite")
            new_composite_path = os.path.join(composite_dir, os.path.basename(image))
            pyeo.composite_images_with_mask((latest_composite_path, new_image_path), new_composite_path)
            latest_composite_path = new_composite_path

    log.info("***PROCESSING END***")
