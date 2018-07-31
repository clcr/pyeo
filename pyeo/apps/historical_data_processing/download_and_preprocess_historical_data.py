import sys
#quick hack for now
sys.path.extend(['/scratch/forest2020/jfr10/ForestSentinel', '/scratch/forest2020/jfr10/ForestSentinel'])
import submodules as pc
import configparser
import os


def parse_date_range_list(date_range_path):
    """Reads a date_range_file and returns a list of lists of date strings"""
    with open(date_range_path, 'r') as date_range_file:
        date_ranges = [date_string.strip().split(" ") for date_string in date_range_file]
    return date_ranges

if __name__ == "__main__":
    """A script designed to be run as an array job on ALICE that downloads and processes bits"""

    conf = configparser.ConfigParser()
    conf.read(r"")
    sen_user = conf['sent_2']['user']
    sen_pass = conf['sent_2']['pass']
    root_dir = conf['forest_sentinel']['root_dir']
    aoi = conf['forest_sentinel']['aoi']
    date_range_path = conf['forest_sentinel']['date_range_file']
    cloud = conf['forest_sentinel']['cloud_cover']
    out_folder = conf['forest_sentinel']['out_folder']
    log_path = conf['forest_sentinel']['log_path']

    pc.create_file_structure(root_dir)
    log = pc.init_log(log_path)

    try:
        array_job_index = int(os.getenv('PBS_ARRAYID'))
    except TypeError:
        print("Bad value for PBS_ARRAYID, setting to 0")
        array_job_index = 0

    date_range_list = parse_date_range_list(date_range_path)

    start_date, end_date = date_range_list[array_job_index]
    pc.sent2_query(sen_user, sen_pass, aoi, start_date, end_date, cloud, out_folder)