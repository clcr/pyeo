import os, sys
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..','..')))
import pyeo.core as pyeo


def test_composite_images_with_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/composite_test.tif")
    except FileNotFoundError:
        pass
    test_data = [r"test_data/20180103T172709.tif",
                 r"test_data/20180319T172021.tif",
                 r"test_data/20180329T171921.tif"]
    out_file = r"test_outputs/composite_test.tif"
    pyeo.composite_images_with_mask(test_data, out_file)


def test_ml_masking():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pyeo.create_mask_from_model(
        r"test_data/20180103T172709.tif",
        r"test_data/cloud_model_v0.1.pkl",
        r"test_outputs/ml_mask_test.tif"
    )


if __name__ == "__main__":
    print(sys.path)
    pyeo.create_model_from_signatures(r"test_data/cloud_training_outlier_removal.csv",
                                      r"test_data/cloud_model_v0.1.pkl")
    test_ml_masking()
