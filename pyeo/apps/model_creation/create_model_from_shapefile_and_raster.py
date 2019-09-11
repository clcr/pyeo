"""Produces a trained model from a raster and associated shapefile.

To create a training set, enclose any features you want to classify with polygons. Give the shapefile an field
of something like 'class_id', as an integer, and set that value to the class inside the polygon. Set the paths
to the shapefile and raster in the associated model_creation.ini file, along with a path to where you want to
store the completed model (as a .pkl file) and the name of the field used to identify the polygons ("class_id"
 in this example).

*Do not make class_id = 0 in any training polygon; it will be ignored if you do*

*At present, the raster must be projected in EPSG 4326*

For example: if you want to make a model that classifies forest and water in an image, draw some polygons
inside forested areas and set their 'class_id' to 1. Then draw some more polygons inside water bodies, and set
their class_id to 2.

This code will create and store a pixel classifier from training data and rasters. Each pixel under a polygon
provides a sample of that polygon's class, with every value of that pixel being a feature of that sample.

At present, the model created is a balanced random forest classifier; there are plans to expand the function
to take the model as an augment, but these are not yet implemented.

You can call this script from the command with a .ini file as an argument"""

import pyeo.classification
import pyeo.filesystem_utilities
import configparser
import argparse
import sklearn.ensemble as ens
import joblib

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Produces a trained model from a raster and associated shapefile')
    parser.add_argument('--conf', dest='config_path', action='store', default=r'model_creation.ini',
                        help="Path to the .ini file specifying the job.")
    args = parser.parse_args()

    conf = configparser.ConfigParser()
    conf.read(args.config_path)

    training_shape_path = conf["pyeo"]["shapefile"]
    training_raster_path = conf["pyeo"]["raster"]
    model_out_path = conf["pyeo"]["model"]
    class_field = conf["pyeo"]["class_field"]
    log_path = conf["pyeo"]["log_path"]

    log = pyeo.filesystem_utilities.init_log(log_path)

    # This will be changed in the near future as I'm planning to refactor core soon
    #  to make the ML model building functions more granular
    learning_data, classes = pyeo.classification.get_training_data(training_raster_path, training_shape_path,
                                                                   class_field)
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
                                     min_samples_split=16, n_estimators=100, n_jobs=4, class_weight='balanced')
    model.fit(learning_data, classes)
    joblib.dump(model, model_out_path)
