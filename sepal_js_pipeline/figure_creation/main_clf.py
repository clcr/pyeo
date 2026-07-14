"""
main_clf.py
Author: Matt Payne
This script creates classification visualisation figures from EE Assets.
"""

import argparse
import datetime
import json
from pathlib import Path
import requests

import ee
from ee.image import Image as eeImage
from ee.featurecollection import FeatureCollection
import google.auth
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_map_utils.core.north_arrow import NorthArrow
import matplotlib.ticker as mticker
from PIL import Image
from pyproj import Transformer

def initialise(project_name: str) -> None:
    """
    Initialises programmatic access to Earth Engine via the Python API, once per session.

    Parameters
    ----------
    project_name : str
        name of the Google Cloud Project with the GEE API enabled.

    Returns
    -------
    None
    """
    try:
        ee.Authenticate()
    except Exception as e:
        print(f"Could not authenticate, encountered this error: {e}")
        print("Exiting...")
        exit(1)
    try:
        ee.Initialize(project=project_name)
    except Exception as e:
        print(f"Could not initialise, encountered this error: {e}")
        print("Exiting...")
        exit(1)

    return

def plot_figure(baseline_image_id: str, baseline_clf_id: str, training_points_id: str, epsg_code: str, png_out_path: Path) -> None:
    """_summary_

    Parameters
    ----------
    baseline_image_id : str
        _description_
    baseline_clf_id : str
        _description_
    training_points_id : str
        _description_
    epsg_code : str
        _description_
    png_out_path : Path
        _description_
    """

    # load the assets
    baseline_img = eeImage(baseline_image_id)
    baseline_clf_img = eeImage(baseline_clf_id)
    training_points_fc = FeatureCollection(training_points_id)

    # extract the visual parameters for the images and assert CRS for getThumbURL
    properties_rgb = baseline_img.getInfo().get("properties", {})
    properties_clf = baseline_clf_img.getInfo().get("properties", {})
    properties_points = training_points_fc.getInfo().get("properties", {})

    vis_rgb_dict = json.loads(properties_rgb["visParamsRGB"])
    vis_clf_dict = json.loads(properties_clf["visClassParams"])
    vis_points_dict = json.loads(properties_points["visClassParams"])
    class_name_dict = json.loads(properties_points["classNameMap"])

    # extract the date metadata
    baseline_start_date = properties_rgb["baseline_start"]
    baseline_end_date = properties_rgb["baseline_end"]

    ####################################################
    # AXES EXTENTS
    ####################################################

    # get the image's bounding geometry
    geom = baseline_img.geometry().bounds().getInfo()
    coords = geom['coordinates'][0]

    # these are in degrees because GeoJSON requires all coords to be in EPSG:4326
    lons = [point[0] for point in coords]
    lats = [point[1] for point in coords]

    # reproject to chosen CRS
    transformer = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    extent = [min(xs), max(xs), min(ys), max(ys)]

    ####################################################
    # TRAINING FEATURES GEOMETRIES
    ####################################################

    features = training_points_fc.getInfo().get("features", [])

    # build lists of geometries
    points_lons = [f["geometry"]["coordinates"][0] for f in features if f.get("geometry")]
    points_lats = [f["geometry"]["coordinates"][1] for f in features if f.get("geometry")]
    points_colours = []
    for f in features:
        class_val = str(f.get("properties", {}).get("class"))
        colour = vis_points_dict.get(class_val, {})
        points_colours.append(colour)

    # reproject
    points_xs, points_ys = transformer.transform(points_lons, points_lats)

    ####################################################
    # GRAPH CONSTRUCTION
    ####################################################

    # construct the graph
    fig, (ax1, ax2) = plt.subplots(figsize=(18, 12), nrows=1, ncols=2, dpi=300)

    title_dict = {"fontweight": "bold", "fontsize": 16}

    ####################################################
    # AX1 MEDIAN RGB
    ####################################################

    # get the image as a thumbnail, adequate for plotting
    url = baseline_img.getThumbURL(vis_rgb_dict)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # plot the image
    ax1.imshow(img, extent=extent)

    # plot the training points
    ax1.scatter(points_xs, points_ys, c=points_colours, edgecolor="None")

    # set title
    ax1.set_title(f"Baseline Median:\n{baseline_start_date} - {baseline_end_date}", fontdict=title_dict)

    ####################################################
    # AX2 MEDIAN CLASSIFIED
    ####################################################

    # get the image as a thumbnail, adequate for plotting
    url = baseline_clf_img.getThumbURL(vis_clf_dict)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # plot the image
    ax2.imshow(img, extent=extent)

    # set title
    ax2.set_title(f"Baseline Median\nClassified", fontdict=title_dict)

    ####################################################
    # AXES SHARED PARAMETERS
    ####################################################

    for ax in (ax1, ax2):
        # north arrow
        north = NorthArrow(size="medium", location="upper right", rotation={"degrees": 0})
        ax.add_artist(north)

        # scale bar
        scalebar = ScaleBar(dx=1, units="m", location="lower left", 
                            scale_formatter=lambda value, unit: f"{value} {unit}",
                            box_alpha=0.6)
        ax.add_artist(scalebar)

    ####################################################
    # AXES COLOUR LEGEND
    ####################################################

    legend_patches = [
        mpatches.Patch(color=colour_name, label=class_name_dict.get(class_id))
        for class_id, colour_name in vis_points_dict.items()
    ]

    fig.legend(
            title="Classes",
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=4,
            frameon=False,
            title_fontsize=16,
            fontsize=14)
    
    # save
    try:
        # plt.subplots_adjust(hspace=0.1, wspace=0.3, bottom=0.15)
        plt.tight_layout()
        plt.savefig(png_out_path, dpi=300)
        plt.close(fig)
    except (OSError, IOError) as e:
        print(f"Could not save plot, encountered: {e}") 

if __name__ == "__main__":
    
    ####### parsing arguments
    parser = argparse.ArgumentParser(prog="This script creates a figure for visualising a classification against the training imagery.")

    parser.add_argument("baseline_asset_path", help="The string corresponding to the path of the baseline image asset.", type=str)

    parser.add_argument("baseline_clf_asset_path", help="The string corresponding to the path of the classified baseline image asset.", type=str)

    parser.add_argument("training_points_asset_path", help="The string corresponding to the path of the training points feature collection asset.", type=str)

    parser.add_argument("png_out_path", help="The string of the output path to write the .png to.", type=str)

    parser.add_argument("epsg", help="The EPSG string to use.", type=str)

    args = parser.parse_args()
    png_out_path = Path(args.png_out_path)
    epsg = args.epsg
    #######

    # define the scopes required for Earth Engine and Google Cloud
    scopes = [
        'https://www.googleapis.com/auth/earthengine',
        'https://www.googleapis.com/auth/cloud-platform'
    ]

    # get the credentials, passing in the required scopes
    credentials, project = google.auth.default(scopes=scopes)

    # initialise the Earth Engine API
    ee.Initialize(credentials=credentials)

    print("Earth Engine authenticated and initialised successfully via environment variable.")

    plot_figure(baseline_image_id=args.baseline_asset_path,
                baseline_clf_id=args.baseline_clf_asset_path,
                training_points_id=args.training_points_asset_path,
                epsg_code=epsg,
                png_out_path=png_out_path)