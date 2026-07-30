"""
main.py
Output Validation
Author: Matt Payne
This script checks whether the output of a change report with ms since epoch, is the same as that with fractional calendar years.
"""

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import rasterio

def plot_change_date_distributions(ms_array: np.ndarray, frac_array: np.ndarray, out_png_path: str, 
                                   array_ms_name:str, array_frac_name: str, bins: int=50):
    """
    Plots side-by-side histograms to compare the distributions of change dates.
    Converts both milliseconds since epoch and fractional years into readable dates for the x-axes.
    
    Parameters
    ----------
    ms_array :np.ndarray
        1D array of dates in milliseconds since epoch.
    frac_array :np.ndarray
        1D array of dates in fractional calendar years.
    array_ms_name : str
        Friendly name for the ms since epoch array.
    array_frac_name : str
        Friendly name for fractional calendar year array.
    out_png_path : str
        Path to write the graph png to.
    bins: int, optional:
        Number of histogram bins. Defaults to 50.
        
    Returns
    -------
    None
    """

    # flatten arrays and remove any NaNs/NoData values
    ms_clean = ms_array[~np.isnan(ms_array)].ravel()
    frac_clean = frac_array[~np.isnan(frac_array)].ravel()
    
    # vectorised conversion for milliseconds
    dt_ms_array = ms_clean.astype('datetime64[ms]')
    
    # vectorised conversion for fractional years
    years = np.floor(frac_clean).astype(int)
    fracs = frac_clean - years
    
    # create string arrays for the start of the current year and the start of the next year
    start_year_str = np.char.add(years.astype(str), '-01-01')
    next_year_str = np.char.add((years + 1).astype(str), '-01-01')
    
    # convert string dates to datetime64[ms] arrays
    start_of_year = start_year_str.astype('datetime64[ms]')
    start_of_next_year = next_year_str.astype('datetime64[ms]')
    
    # calculate the exact duration of each year in milliseconds (handles leap years)
    ms_in_year = start_of_next_year - start_of_year
    
    # multiply the fraction by the year's duration and add it to the start of the year
    added_ms = (fracs * ms_in_year.astype(float)).astype('timedelta64[ms]')
    dt_frac_array = start_of_year + added_ms
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # plot 1: 
    ax1.hist(dt_ms_array, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_title(array_ms_name)
    ax1.set_ylabel('Pixel Count')
    
    # plot 2: fractional years
    ax2.hist(dt_frac_array, bins=bins, color='seagreen', edgecolor='black', alpha=0.7)
    ax2.set_title(array_frac_name)
    ax2.set_ylabel('Pixel Count')
    
    # format both x-axes with datetime format string
    date_fmt = mdates.DateFormatter("%d %B %Y")
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(date_fmt)
        # rotate labels so they don't overlap
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # save
    try:
        plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except (OSError, IOError) as e:
        print(f"Could not save plot, encountered: {e}") 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("change_report_ms_path", help="The string corresponing to the path of first change report, encoded in ms since epoch.", type=str)
    parser.add_argument("change_report_frac_path", help="The string corresponding to the path of the second change report, encoded in fractional calendar years.", type=str)
    parser.add_argument("change_report_ms_name", help="Friendly name to distinguish change_report_a with", type=str)
    parser.add_argument("change_report_frac_name", help="Friendly name to distinguish change_report_b with", type=str)
    parser.add_argument("out_png_path", help="The string corresponding to the output path to write the analysis graphs to", type=str)

    # parse them arguments
    args = parser.parse_args()

    # read rasters
    with rasterio.open(args.change_report_ms_path) as src_ms, rasterio.open(args.change_report_frac_path) as src_frac:

        change_report_ms = src_ms.read()
        change_report_frac = src_frac.read()

        # plot histograms
        plot_change_date_distributions(ms_array=change_report_ms[10, :, :], frac_array=change_report_frac[10, :, :],
                                       array_ms_name=args.change_report_ms_name,
                                       array_frac_name=args.change_report_frac_name,
                                       out_png_path=args.out_png_path,
                                       bins=50)