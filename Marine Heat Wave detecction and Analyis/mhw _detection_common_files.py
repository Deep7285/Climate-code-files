# common script to calculate the marine heat wave 
# Imports the libraries and dependencies
from glob import glob
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

# Oliver’s package (Hobday method implementation)
import marineHeatWaves as mhw

# files path
FILES_GLOB = "/home/Noah_data_1982-2024_SST_daily_mean/sst.day.mean.*.nc" 
SST_VAR    = "sst"

# define the baseline year or analysis time-frame
BASELINE = (1982, 2024)

# Event definition based on Hobday et al. (2016)
MIN_DUR  = 5     # minimum event duration (days)
MAX_GAP  = 2     # join across gaps up to this many days

# Dask-friendly chunks
CHTIME, CHXY = 160, 40

# Set the regions
ROI_DICT = {
    "Arabian Sea": {"lon_min": 40.0, "lon_max": 80.0, "lat_min":  0.0, "lat_max": 30.0, "slug": "arabian_sea"},
    "Bay Of Bengal": {"lon_min": 80.0, "lon_max": 110.0,"lat_min":  0.0, "lat_max":  30.0, "slug": "bay_of_bengal"},
    "North Indian Ocean": {"lon_min": 40.0, "lon_max": 110.0,"lat_min":  0.0, "lat_max":  30.0,"slug": "north_indian_ocean"},
}

OUTROOT = Path("outputs_mhw"); OUTROOT.mkdir(parents=True, exist_ok=True)

# open SST data using xarray
def open_sst(files_glob: str, roi: dict) -> tuple[xr.Dataset, str, str]:
    """Open OISST, subset ROI, return dataset and coordinate names."""
    paths = sorted(glob(files_glob))
    if not paths:
        raise FileNotFoundError(f"No files match: {files_glob}")
    ds = xr.open_mfdataset(paths, combine="by_coords",chunks={"time": CHTIME}, engine="netcdf4")
    latn = "lat" if "lat" in ds.coords else "latitude"
    lonn = "lon" if "lon" in ds.coords else "longitude"
    ds = ds.sel({latn: slice(roi["lat_min"], roi["lat_max"]),lonn: slice(roi["lon_min"], roi["lon_max"])})
    return ds, latn, lonn
# calculate the area of the grids
def area_weights_1d(ds: xr.Dataset, latn: str) -> xr.DataArray:
    """1-D cos(lat) weights (finite) broadcastable across lon."""
    return np.cos(np.deg2rad(ds[latn]))

# use the hobdey's defination on region
def _detect_mask_1d(sst_1d: np.ndarray, thresh_1d: np.ndarray, min_dur: int = MIN_DUR, max_gap: int = MAX_GAP) -> np.ndarray:
    """
    Hobday event mask on a single time series: exceed >= threshold; join gaps <= max_gap; drop runs < min_dur.
    """
    ok = np.isfinite(sst_1d) & np.isfinite(thresh_1d)
    exc = ok & (sst_1d >= thresh_1d)
    if not exc.any():
        return exc
    x = exc.astype(np.int8)
    n = x.size

    # join short gaps
    i = 0
    while i < n:
        if x[i] == 1:
            j = i + 1
            while j < n and x[j] == 1:
                j += 1
            g0 = j
            while g0 < n and x[g0] == 0:
                g0 += 1
            gap_len = g0 - j
            if gap_len > 0 and gap_len <= max_gap:
                x[j:g0] = 1
                j = g0
            i = j
        else:
            i += 1

    # enforce min duration
    y = x.copy()
    i = 0
    while i < n:
        if y[i] == 1:
            j = i + 1
            while j < n and y[j] == 1:
                j += 1
            if (j - i) < min_dur:
                y[i:j] = 0
            i = j
        else:
            i += 1

    return y.astype(bool)

def detect_mask_time(sst: xr.DataArray, thresh: xr.DataArray) -> xr.DataArray:
    """ Vectorized Hobday event mask over (time, lat, lon). Requires single time-chunk."""
    return xr.apply_ufunc(_detect_mask_1d, sst, thresh, input_core_dims=[["time"], ["time"]], output_core_dims=[["time"]],
                          vectorize=True, dask="parallelized",output_dtypes=[bool],)

# per grid climatology & threshold 
def _clim_thresh_time_1d(ords_1d: np.ndarray, temp_1d: np.ndarray,y0: int, y1: int, pct: int,
                         min_dur: int, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """
    For one grid time series: run mhw.detect to get time-aligned climatology (seas) and threshold.
    """
    # If mostly missing, return NaNs to avoid unstable fits
    if np.isfinite(temp_1d).sum() < 30:
        n = temp_1d.size
        return np.full(n, np.nan, float), np.full(n, np.nan, float)

    # oliver's detection
    _, clim = mhw.detect(ords_1d.astype(int), temp_1d.astype(float),climatologyPeriod=[int(y0), int(y1)],pctile=int(pct),minDuration=int(min_dur),
        joinAcrossGaps=True,maxGap=int(max_gap),)
    seas   = np.asarray(clim["seas"],   float)
    thresh = np.asarray(clim["thresh"], float)
    return seas, thresh
                             
# compute time-aligned climatology and threshold for every grid
def build_grid_baseline(ds: xr.Dataset, latn: str, lonn: str,pctile: int = 90) -> tuple[xr.DataArray, xr.DataArray]:
    
    t_index = pd.to_datetime(ds["time"].values)
    y0 = max(BASELINE[0], t_index.year.min())
    y1 = min(BASELINE[1], t_index.year.max())
    ords_da = xr.DataArray(np.array([d.toordinal() for d in t_index], dtype=int),coords={"time": ds["time"]},dims=["time"],).chunk({"time": -1})  # single time chunk for gufunc
    sst = ds[SST_VAR].chunk({"time": -1, latn: CHXY, lonn: CHXY})

    seas_t, thresh_t = xr.apply_ufunc(_clim_thresh_time_1d, ords_da, sst,input_core_dims=[["time"], ["time"]],output_core_dims=[["time"], ["time"]],
        vectorize=True, dask="parallelized",output_dtypes=[float, float],kwargs=dict(y0=int(y0), y1=int(y1), pct=int(pctile),min_dur=MIN_DUR, max_gap=MAX_GAP),)
    seas_t   = seas_t.rename("seas_t")
    thresh_t = thresh_t.rename("thresh_t")
    return seas_t, thresh_t

