# Spatial map of north indian ocean
from glob import glob
from pathlib import Path
import numpy as np, xarray as xr, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cmocean.cm as cmo


# data loading
FILES_GLOB = "/home/deepak/Desktop/CAS_deepak/Noah_data_1982-2024_SST_daily_mean/sst.day.mean.*.nc"
VAR = "sst"
YEARS = (1982, 2024)
ROI = {"lat_min": 0.0, "lat_max": 25.0, "lon_min": 40.0, "lon_max": 100.0}


# Min -max color levels & discrete steps
V_MIN, V_MAX, V_STEP = 26, 30, 0.25
LEVELS = np.arange(V_MIN, V_MAX + V_STEP, V_STEP)

# Color Palette: light cream to deep red 
PALETTE = [
    "#f9f5e7", "#f2ebd2", "#eedeb7", "#e8cea1","#e5c592", "#e3bd83",
          "#e0ad6a", "#dc9c55", "#d88943", "#d47635", "#cf652b",
          "#c94f23", "#c03b1c", "#b62c18", "#a92014", "#99150f"
]

CMAP = ListedColormap(PALETTE)
NORM = BoundaryNorm(LEVELS, ncolors=CMAP.N, clip=False)  

# Open data & subset region/time of interest (ROI)
def open_roi(files_glob, roi, years, engine="netcdf4"):
    paths = sorted(glob(files_glob))
    if not paths:
        raise FileNotFoundError("No files found.")
    ds = xr.open_mfdataset(paths, combine="by_coords", parallel=True,
                           chunks={"time": 120}, engine=engine)
    # define coords names
    latn = "lat" if "lat" in ds.coords else "latitude"
    lonn = "lon" if "lon" in ds.coords else "longitude"
    # spatial subset
    ds = ds.sel({latn: slice(roi["lat_min"], roi["lat_max"]),
                 lonn: slice(roi["lon_min"], roi["lon_max"])})
    # temporal subset
    ds = ds.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
    return ds, latn, lonn

# plotting function
def plot_map(data2d: xr.DataArray, latn: str, lonn: str, title: str):
    plt.figure(figsize=(7.8, 4.6))
    #assiging the land color (NAN values)
    cmap_land = CMAP.copy()
    cmap_land.set_bad(color= "#d3d3d3") # light gray for land
    im = plt.pcolormesh(data2d[lonn], data2d[latn], data2d,
                        shading="auto", cmap=cmap_land, norm=NORM)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title(title)
    cbar = plt.colorbar(im, orientation="vertical", extend="both", pad=0.02)
    cbar.set_label("Temp (°C)")
    cbar.set_ticks(LEVELS)
    cbar.ax.set_yticklabels([f"{lv:.1f}" for lv in LEVELS])
    ax= plt.gca()
    #x and y lables formatting
    xticks= np.arange(40, 101, 10)
    yticks= np.arange(0, 26, 5)
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    xlabels= [f"{lon}°E" for lon in xticks]
    ylabels= []
    for lat in yticks:
        if lat == 0:
            ylabels.append("0°")
        else:
            ylabels.append(f"{lat}°N")
    ax.set_xticklabels(xlabels); ax.set_yticklabels(ylabels)
    plt.tight_layout(); plt.show()

