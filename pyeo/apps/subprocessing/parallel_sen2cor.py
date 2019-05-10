import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core
import argparse
import shutil

def moveL2(from_path, to_path):
    '''
    removes the S2A directory on a Sentinel-2 scene from the L2 directory
    then copies the S2A directory from L1 to L2
    and finally removes S2A directory from L1
    '''

    #TODO Test

    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)
    shutil.rmtree(from_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel sen2cor')
    parser.add_argument('l1_dir', action='store', help="Path to the directory containing L1 imagery")
    parser.add_argument('l2_dir', action='store', help="Path to directory to contain L2 imagery")
    array_id = int(os.getenv("PBS_ARRAYID"))
    sen_2_cor_home = os.getenv("SEN2COR_HOME")
    args = parser.parse_args()

    log = pyeo.core.init_log("sen2cor_{}.log".format(array_id))

    new_home = os.path.join((sen_2_cor_home), str(array_id))
    log.info("Setting SEN2COR_HOME to {}".format(new_home))
    try:
        os.mkdir(new_home)
    except FileExistsError:
        log.warning("{} already exists, continuing.".format(new_home))
    os.putenv("SEN2COR_HOME", new_home)

    file_list = [os.path.join(args.l1_dir, l1_filename) for l1_filename in sorted(os.listdir(args.l1_dir))]

    l2_name = pyeo.core.apply_sen2cor(file_list[array_id],
                                      r"/scratch/clcr/shared/Sen2Cor-02.05.05-Linux64/bin/L2A_Process")
    from_path = os.path.join(args.l1_dir, os.path.basename(l2_name))
    to_path = os.path.join(args.l2_dir, os.path.basename(l2_name))
    log.info("l2_name  : {}".format(l2_name))
    log.info("Moving   : {}".format(from_path))
    log.info("Moving to: {}".format(to_path))
    moveL2(from_path, to_path)

