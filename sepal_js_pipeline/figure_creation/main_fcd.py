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

import contextily as ctx
import ee
from ee.image import Image as eeImage
import geopandas as gpd
import google.auth
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_map_utils.core.north_arrow import NorthArrow
import matplotlib.ticker as mticker
import osmnx as ox
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box

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

def plot_figure(baseline_image_id: str, first_image_id: str, last_image_id: str, change_report_id: str, epsg_code: str, png_out_path: Path, country_string: str):
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
    country_string : str
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

    ############
    #  construct the graph
    ############

    fig = plt.figure(figsize=(18, 16), layout="constrained")
    gs = gridspec.GridSpec(nrows=3, ncols=3, figure=fig)

    # create the axes
    axes_top_rgb = [fig.add_subplot(gs[0, i]) for i in range(3)] # 3 cols
    axes_middle_report = [fig.add_subplot(gs[1, i]) for i in range(3)] # 3 cols
    ax_bottom_extent = [fig.add_subplot(gs[2, 1])][0] # 1 plot in the 2nd col

    title_dict = {"fontweight": "bold", "fontsize": 16}

    ###################
    # PLOT THE RGB IMAGERY
    ###################

    for ax, img, title in zip(axes_top_rgb, images, titles):

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
        scalebar = ScaleBar(dx=1, units="deg", dimension="angle", location="lower left", 
                            scale_formatter=lambda value, unit: f"{value} {unit}",
                            box_alpha=0.6)
        ax.add_artist(scalebar)

    ###################
    # PLOT THREE CHANGE REPORT LAYERS
    ###################

    for ax, vis, title in zip(axes_middle_report, change_params, change_titles):

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
        scalebar = ScaleBar(dx=1, units="deg", dimension="angle", location="lower left", 
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

    ###################
    # PLOT EXTENT
    ###################

    region_gdf = ox.geocode_to_gdf(country_string)
    region_gdf = region_gdf.to_crs(epsg_code)

    # plot state boundary
    region_gdf.plot(ax=ax_bottom_extent, facecolor="none", edgecolor="black", linewidth=2, zorder=2)

    # plot the image AOI as an extent indicator onto the state boundary
    aoi_box = box(min(lons), min(lats), max(lons), max(lats))
    aoi_gdf = gpd.GeoDataFrame({"geometry": [aoi_box]}, crs="EPSG:4326")

    aoi_gdf = aoi_gdf.to_crs(epsg_code)    
    aoi_gdf.plot(ax=ax_bottom_extent, facecolor="none", edgecolor="red", alpha=0.6, zorder=3, linewidth=10)

    ####### basemap
    ctx.add_basemap(ax_bottom_extent, crs=region_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron, zorder=1)

    ####### format extent map
    # north arrow
    north = NorthArrow(size="medium", location="upper right", rotation={"degrees": 0})
    ax_bottom_extent.add_artist(north)
    
    # scale bar
    ax_bottom_extent.set_aspect("equal")
    scalebar = ScaleBar(dx=1, units="deg", dimension="angle", location="lower left", 
                        scale_formatter=lambda value, unit: f"{value} {unit}",
                        box_alpha=0.8)
    
    ax_bottom_extent.add_artist(scalebar)
    ax_bottom_extent.set_title(f"Location of pilot site within {country_string}", fontdict=title_dict)
    ax_bottom_extent.set_xlabel("Longitude")
    ax_bottom_extent.set_ylabel("Latitude")

    # legend time
    aoi_patch = mpatches.Patch(facecolor="none", edgecolor="red", alpha=1, label="AOI Boundary")
    state_patch = mpatches.Patch(facecolor="none", edgecolor="black", alpha=1, label="Country Boundary")
    ax_bottom_extent.legend(handles=[aoi_patch, state_patch], loc="lower right")

    ######### plotting bounds
    bounds = region_gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    
    # calculate the geographic width and height
    dx = maxx - minx
    dy = maxy - miny

    # calculate centre coords
    cx = (maxx + minx) / 2.0
    cy = (maxy + miny) / 2.0

    # find the max dimension to force a square plot window
    max_dim = max(dx, dy)

    buffer = 0.05
    padded_dim = max_dim * (1 + buffer)

    # set the extents
    ax_bottom_extent.set_xlim(cx - (padded_dim / 2.0), cx + (padded_dim / 2.0))
    ax_bottom_extent.set_ylim(cy - (padded_dim / 2.0), cy + (padded_dim / 2.0))
    #######

    # save
    try:
        #plt.tight_layout()
        # plt.subplots_adjust(hspace=0.1, wspace=0.3)
        plt.savefig(png_out_path, dpi=300, bbox_inches="tight")
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
    parser.add_argument("country_string", help="A string where the AOI is located, taken in the <COUNTRY> format.", type=str)

    args = parser.parse_args()
    png_out_path = Path(args.png_out_path)
    epsg = args.epsg
    country_string = args.country_string

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
                epsg_code=epsg,
                country_string=country_string)