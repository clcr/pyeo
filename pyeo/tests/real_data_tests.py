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
    )
    shutil.copy(r"test_data/20180103T172709.msk", r"test_outputs/model_mask.tif")


def test_mask_combination():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    masks = [mask for mask in os.listdir("test_data") if mask.endswith(".msk")]
    if os.path.exists("test_outputs/union_or_combination.tif"):
        os.remove("test_outputs/union_or_combination.tif")
    if os.path.exists("test_outputs/intersection_and_combination.tif"):
        os.remove("test_outputs/intersection_and_combination.tif")
    pyeo.combine_masks(masks, "test_outputs/union_or_combination.tif",
                       geometry_func="union", combination_func="or")
    pyeo.combine_masks(masks, "test_outputs/intersection_and_combination.tif",
                       geometry_func="intersection", combination_func="and")


if __name__ == "__main__":
    print(sys.path)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log = pyeo.init_log("test_log.log")
    test_mask_combination()
