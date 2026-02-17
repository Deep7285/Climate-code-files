# Annual climatology spatial map animation for the North Indian Ocean 
# This animation shows the daily mean SST climatology (1982-2024) for each day of the year from 1st January to 31st December
# This animation has 366 total frames representing the daily mean SST climatology for each day of the year

from glob import glob
from pathlib import Path
import numpy as np 
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import animation

# data path & region of interest (ROI)
FILES_GLOB = "/home/Desktop/Noah_data_1982-2024_SST_daily_mean/sst.day.mean.*.nc"
VAR = "sst"
CLIM_YEARS = (1982, 2024)
ROI = {"lat_min": 0.0, "lat_max": 25.0, "lon_min": 20.0, "lon_max": 100.0}

#saving output directory
OUTDIR = Path("outputs_anim"); OUTDIR.mkdir(exist_ok=True, parents=True)
OUT_MP4 = OUTDIR / "NIO_DailyClimatology_1982-2024.mp4"
OUT_GIF = OUTDIR / "NIO_DailyClimatology_1982-2024.gif"

# Setting temperature scale and colour map
V_MIN, V_MAX, V_STEP = 20, 34, 1
LEVELS = np.arange(V_MIN, V_MAX + V_STEP, V_STEP)

# Colour Palette: light cream to deep red 
PALETTE = [
    "#f9f5e7", "#f2ebd2", "#eedeb7", "#e8cea1", "#e3bd83",
          "#e0ad6a", "#dc9c55", "#d88943", "#d47635", "#cf652b",
          "#c94f23", "#c03b1c", "#b62c18", "#a92014", "#99150f"
]

CMAP = ListedColormap(PALETTE)
norm = BoundaryNorm(LEVELS, ncolors=CMAP.N, clip=False) 

# load the data & subset ROI
def open_roi(files_glob, roi, engine="netcdf4"):
    paths = sorted(glob(files_glob))
    if not paths:
        raise FileNotFoundError("No input files found.")
    ds = xr.open_mfdataset(paths, combine="by_coords", chunks={"time": 365}, engine=engine)
    latn = "lat" if "lat" in ds.coords else "latitude"
    lonn = "lon" if "lon" in ds.coords else "longitude"
    ds = ds.sel({latn: slice(roi["lat_min"], roi["lat_max"]),
                 lonn: slice(roi["lon_min"], roi["lon_max"])})
    return ds, latn, lonn

ds, latn, lonn = open_roi(FILES_GLOB, ROI, engine="netcdf4")

# Clip to baseline years defensively
years = pd.to_datetime(ds.time.values).year
ds = ds.sel(time=((years >= CLIM_YEARS[0]) & (years <= CLIM_YEARS[1])))

# Daily climatolgy per grid point (366 days)
daily_clim = ds[VAR].groupby("time.dayofyear").mean("time", skipna=True).compute()
daily_clim = daily_clim.rename({"dayofyear": "doy"}).assign_coords(doy=np.arange(1, 367))

# Include the leap year day (29 Feb)
ref_dates = pd.date_range("2000-01-01", "2000-12-31", freq="D")  
doy_to_label = [d.strftime("%d %b") for d in ref_dates]          

# Create the animation
Lon, Lat = np.meshgrid(daily_clim[lonn].values, daily_clim[latn].values)

fig, ax = plt.subplots(figsize=(8.5, 5))
# First frame
im = ax.pcolormesh(Lon, Lat, daily_clim.isel(doy=0).values,
                   cmap=cmap, norm=norm, shading="nearest")
cb = fig.colorbar(im, ax=ax, orientation="vertical", extend='max')
cb.set_label("Temp (°C)")

ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
title = ax.set_title(f"Annual Climatology of North Indian Ocean (1982–2024)\nDay 001 — {doy_to_label[0]}")

# Update function for FuncAnimation
def update(frame_idx):
    arr = daily_clim.isel(doy=frame_idx).values
    im.set_array(arr.ravel())
    # Day label with leading zeros
    day_str = f"{frame_idx+1:03d}"
    title.set_text(
        f"Annual Climatology of North Indian Ocean(1982–2024)\nDay {day_str} — {doy_to_label[frame_idx]}"
    )
    return [im, title]

anim = animation.FuncAnimation(fig, update, frames=366, interval=80, blit=False)

# Try saving to MP4 If not available, fall back to GIF.
try:
    anim.save(OUT_MP4.as_posix(), dpi=300, writer=animation.FFMpegWriter(fps=10, bitrate=10000))
    print(f"Saved MP4: {OUT_MP4}")
except Exception as e:
    print(f"[warn] MP4 save failed ({e}); writing GIF instead …")
    anim.save(OUT_GIF.as_posix(), dpi=300, writer=animation.PillowWriter(fps=10))
    print(f"Saved GIF: {OUT_GIF}")

plt.close(fig)


