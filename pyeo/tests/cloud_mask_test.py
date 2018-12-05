import os, sys
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..')))
import pyeo.core as pyeo

def test_cloud_mask(safe_file, out_file):
    pyeo.create_mask_from_confidence_layer(safe_file, out_file)


def compare_cloud_masks(in_dir, safe_file,  out_range)


if __name__=="__main__":
    os.chdir("some_dir")
    test_cloud_mask("some_file", "some_other_file")