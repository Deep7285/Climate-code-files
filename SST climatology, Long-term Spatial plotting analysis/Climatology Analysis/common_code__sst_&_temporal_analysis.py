# Common code for Climatologoy, Threshold, Mean STT ,SST Anomaly, SST Trend and MHW days correlation analysis
#Run this code first before running other analysis codes

from glob import glob
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import marineHeatWaves as mhw  

# convert np.NAN to np.nan to avoid dependency issues
import numpy as _np
if not hasattr(_np, "NaN"):
    _np.NaN = _np.nan

# File path
FILES_GLOB = "/home/deepak/Desktop/CAS_deepak/Noah_data_1982-2024_SST_daily_mean/sst.day.mean.*.nc"
VAR = "sst"
CLIM_YEARS: Tuple[int,int] = (1982, 2024)   
REGIONS: Dict[str, Dict[str, float]] = {   # modify the lat-lon bounds as per your region
    "Arabian Sea":    {"lon_min": 40.0, "lon_max": 78.0,  "lat_min": 0.0, "lat_max": 30.0},
    "Bay Of Bengal":   {"lon_min": 78.0, "lon_max": 110.0, "lat_min": 0.0, "lat_max": 30.0},
    "North Indian Ocean": {"lon_min": 40.0, "lon_max": 110.0, "lat_min": 0.0, "lat_max": 30.0},
}

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
season_order = ["DJF", "MAM", "JJA", "SON"] 
OUTDIR = Path("outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)

# leap-aware reference year 
_ref = pd.date_range("2000-01-01", "2000-12-31", freq="D") 
_doy_to_month = pd.Series(_ref.month.values, index=np.arange(1, 367))
_month_to_doy0 = {m: (_doy_to_month[_doy_to_month == m].index.values - 1) for m in range(1, 13)}


def open_mfdataset(paths_glob: str, chunks={"time": 120}, engine: str = "netcdf4") -> xr.Dataset:
    paths = sorted(glob(paths_glob))
    if not paths:
        raise FileNotFoundError(f"No files match: {paths_glob}")
    print(f"[open] {len(paths)} files")
    return xr.open_mfdataset(paths, combine="by_coords", parallel=True, chunks=chunks, engine=engine)

def subset_box(ds: xr.Dataset, box: Dict[str, float]) -> xr.Dataset:
    latn = "lat" if "lat" in ds.coords else "latitude"
    lonn = "lon" if "lon" in ds.coords else "longitude"
    return ds.sel({latn: slice(box["lat_min"], box["lat_max"]),lonn: slice(box["lon_min"], box["lon_max"])})

def area_weighted_boxmean(da: xr.DataArray) -> xr.DataArray:
    latn = "lat" if "lat" in da.coords else "latitude"
    w = np.cos(np.deg2rad(da[latn]))
    return da.weighted(w).mean(dim=[latn, "lon" if "lon" in da.coords else "longitude"])
