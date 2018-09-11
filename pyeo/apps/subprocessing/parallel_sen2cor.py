import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel sen2cor')
    parser.add_argument('l1_path', action='store', help="Path to the directory containing L1 imagery")
    parser.add_argument('l2_path', action='store', help="Path to directory to contain L2 imagery")
    array_id = os.getenv("PBS_ARRAYID")
    args = parser.parse_args()

    pyeo.core.init_log("sen2cor_{}.log".format(array_id))

    file_list = os.path.join(args.l1_path, os.listdir(args.l1_path).sort().list)

    pyeo.core.apply_sen2cor(file_list[array_id], args.l2_path)
