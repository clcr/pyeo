"""
main_fcd.py
Author: Matt Payne
This script creates First Change Date (FCD) comparison figures from EE Assets.
"""

import argparse
import datetime
import json
from pathlib import Path
import requests

import ee
from ee.image import Image as eeImage
import google.auth
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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

def plot_figure(baseline_image_id: str, first_image_id: str, last_image_id: str, change_report_id: str, epsg_code: str, png_out_path: Path):
    """_summary_

    Parameters
    ----------
    baseline_image_id : str
        _description_
    first_image_id : str

    last_image_id : str
        _description_
    change_report_id : str
        _description_
    epsg_code : str
        _description_
    png_out_path : Path
        _description_
    """

    # load the assets
    baseline_img = eeImage(baseline_image_id)
    first_img = eeImage(first_image_id)
    last_img = eeImage(last_image_id)
    change_img = eeImage(change_report_id)

    # extract the visual parameters for the images and assert CRS for getThumbURL
    properties = baseline_img.getInfo().get("properties", {})
    vis_params = json.loads(properties["visParamsRGB"])

    # extract the date metadata
    first_img_date = first_img.getInfo().get("properties", {})["date"]
    last_img_date = last_img.getInfo().get("properties", {})["date"]
    from_date = baseline_image_id.split("_")[-2]
    end_date = baseline_image_id.split("_")[-1]

    properties = change_img.getInfo().get("properties", {})
    total_changes_params = json.loads(properties["totalChangesVisParams"])
    repeatability_params = json.loads(properties["repeatabilityVisParams"])
    fcd_decision_params = json.loads(properties["fcdDecisionVisParams"])


    # define things to iterate through
    images = [baseline_img, first_img, last_img]
    titles = [f"Baseline Median:\n{from_date} - {end_date}", f"First Monitoring Image:\n{first_img_date}", f"Last Monitoring Image:\n{last_img_date}"]
    change_params = [total_changes_params, repeatability_params, fcd_decision_params]
    change_titles = ["Total Changes", "Post FCD Change Repeatability (%)", "FCD Decision"]

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
    
    # construct the graph
    fig, axes = plt.subplots(figsize=(18, 12), nrows=2, ncols=3, dpi=300)

    title_dict = {"fontweight": "bold", "fontsize": 16}

    # iterate through the top three plots
    for ax, img, title in zip(axes[0], images, titles):

        # get the image as a thumbnail, adequate for plotting
        url = img.getThumbURL(vis_params)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # plot the image
        ax.imshow(img, extent=extent)

        # set title
        ax.set_title(title, fontdict=title_dict)

        # north arrow
        north = NorthArrow(size="medium", location="upper right", rotation={"degrees": 0})
        ax.add_artist(north)

        # scale bar
        scalebar = ScaleBar(dx=1, units="m", location="lower left", 
                            scale_formatter=lambda value, unit: f"{value} {unit}",
                            box_alpha=0.6)
        ax.add_artist(scalebar)

    for ax, vis, title in zip(axes[1], change_params, change_titles):

        # get the image as a thumbnail, adequate for plotting
        url = change_img.getThumbURL(vis)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        ax.imshow(img, extent=extent)

        ax.set_title(title, fontdict=title_dict)

        # north arrow
        north = NorthArrow(size="medium", location="upper right", rotation={"degrees": 0})
        ax.add_artist(north)

        # scale bar
        scalebar = ScaleBar(dx=1, units="m", location="lower left", 
                            scale_formatter=lambda value, unit: f"{value} {unit}",
                            box_alpha=0.6)
        ax.add_artist(scalebar)

        # construct palette legend
        vmin = vis.get("min")
        vmax = vis.get("max")
        palette = vis.get("palette")

        # create colour map
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", palette)
        # ensure values are normalised
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        # create scalar mappable linking normlisation and colour map
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

        # add to the subplot
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

        # format the dates of fcd decision map colour map
        if title == "FCD Decision":
            cbar.set_label("Date")

            date_formatter = mticker.FuncFormatter(
                lambda x, _: datetime.datetime.fromtimestamp(x / 1000.0).strftime("%d %B %Y"))
            cbar.ax.yaxis.set_major_formatter(date_formatter)

    # save
    try:
        #plt.tight_layout()
        plt.subplots_adjust(hspace=0.1, wspace=0.3)
        plt.savefig(png_out_path, dpi=300)
        plt.close(fig)
    except (OSError, IOError) as e:
        print(f"Could not save plot, encountered: {e}")

if __name__ == "__main__":
    
    ####### parsing arguments
    parser = argparse.ArgumentParser(prog="This script creates a figure for comparing the change report.")
    parser.add_argument("baseline_asset_path", help="The string corresponding to the path of a baseline image asset.", type=str)
    parser.add_argument("first_monitoring_image_asset_path", help="The string corresponding to the path of the first monitoring image asset.", type=str)
    parser.add_argument("last_monitoring_image_asset_path", help="The string corresponding to the path of the final monitoring image asset.", type=str)
    parser.add_argument("change_report_asset_path", help="The string corresponding to the path of the change report asset.", type=str)
    parser.add_argument("png_out_path", help="The string of the output path to write the .png to.", type=str)
    parser.add_argument("epsg", help="A string of the EPSG to use.", type=str)

    args = parser.parse_args()
    png_out_path = Path(args.png_out_path)
    epsg = args.epsg

    # TODO put into a function
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
                first_image_id=args.first_monitoring_image_asset_path,
                last_image_id=args.last_monitoring_image_asset_path,
                change_report_id=args.change_report_asset_path,
                png_out_path=png_out_path,
                epsg_code=epsg)