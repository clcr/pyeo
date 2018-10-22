'''
We can use the Example given in the mpl_toolkits documentation, but the axes_class needs to
be set explicitly, it has to be set as axes_class = plt.Axes, else it attempts to create a
GeoAxes as colorbar
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs


def sample_data_3d(shape):
    """Returns `lons`, `lats`, and fake `data`

    adapted from:
    http://scitools.org.uk/cartopy/docs/v0.15/examples/axes_grid_basic.html
    """

    nlons, nlats = shape
    lats = np.linspace(-np.pi / 2, np.pi / 2, nlats)
    lons = np.linspace(0, 2 * np.pi, nlons)
    lons, lats = np.meshgrid(lons, lats)
    wave = 0.75 * (np.sin(2 * lats) ** 8) * np.cos(4 * lons)
    mean = 0.5 * np.cos(2 * lats) * ((np.sin(2 * lats)) ** 2 + 2)

    lats = np.rad2deg(lats)
    lons = np.rad2deg(lons)
    data = wave + mean

    return lons, lats, data


# get data
lons, lats, data = sample_data_3d((180, 90))

# set up the plot
proj = ccrs.PlateCarree()

f, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj))
h = ax.pcolormesh(lons, lats, data, transform=proj, cmap='RdBu')
ax.coastlines()

# following https://matplotlib.org/2.0.2/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes
# we need to set axes_class=plt.Axes, else it attempts to create
# a GeoAxes as colorbar

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)

f.add_axes(ax_cb)
plt.colorbar(h, cax=ax_cb)
plt.show()
