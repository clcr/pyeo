import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel sen2cor')
    parser.add_argument('l1_dir', action='store', help="Path to the directory containing L1 imagery")
    parser.add_argument('l2_dir', action='store', help="Path to directory to contain L2 imagery")
    array_id = int(os.getenv("PBS_ARRAYID"))
    sen_2_cor_home = os.getenv("SEN2COR_HOME")
    args = parser.parse_args()

    log = pyeo.core.init_log("sen2cor_{}.log".format(array_id))

    new_home = os.path.join((sen_2_cor_home), array_id)
    os.mkdir(new_home)
    log.info("Setting home to {}".format(new_home))
    os.putenv("SEN2COR_HOME", new_home)

    file_list = [os.path.join(args.l1_dir, l1_filename) for l1_filename in sorted(os.listdir(args.l1_dir))]

    l2_name = pyeo.core.apply_sen2cor(file_list[array_id], r"/scratch/clcr/shared/Sen2Cor-02.05.05-Linux64/bin/L2A_Process")
    shutil.move(os.path.join(args.l1_dir, l2_name), args.l2_dir)

