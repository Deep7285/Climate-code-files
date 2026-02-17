# Monthly Climatology, 90th & 80th percentile thresholds for Arabian Sea, Bay Of Bengal, North Indian Ocean

from glob import glob
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import marineHeatWaves as mhw  


FILES_GLOB = "/home/Desktop/Noah_data_1982-2024_SST_daily_mean/sst.day.mean.*.nc"
VAR = "sst"
CLIM_YEARS: Tuple[int, int] = (1982, 2024)
REGIONS: Dict[str, Dict[str, float]] = {
    "Arabian Sea":      {"lon_min": 20.0, "lon_max": 78.0,  "lat_min": 0.0, "lat_max": 25.0},
    "Bay Of Bengal":    {"lon_min": 78.0, "lon_max": 100.0, "lat_min": 0.0, "lat_max": 25.0},
    "North Indian Ocean": {"lon_min": 20.0, "lon_max": 100.0, "lat_min": 0.0, "lat_max": 25.0},
}
OUTDIR = Path("outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)

# leap-aware reference year 
_ref = pd.date_range("2000-01-01", "2000-12-31", freq="D") 
_doy_to_month = pd.Series(_ref.month.values, index=np.arange(1, 367))

_month_to_doy0 = {m: (_doy_to_month[_doy_to_month == m].index.values - 1) for m in range(1, 13)}


def open_mfdataset(paths_glob: str, chunks={"time": 120}, engine: str = "netcdf4") -> xr.Dataset:
    paths = sorted(glob(paths_glob))
    if not paths:
        raise FileNotFoundError(f"No files match: {paths_glob}")
    return xr.open_mfdataset(paths, combine="by_coords", parallel=True, chunks=chunks, engine=engine)

def subset_box(ds: xr.Dataset, box: Dict[str, float]) -> xr.Dataset:
    latn = "lat" if "lat" in ds.coords else "latitude"
    lonn = "lon" if "lon" in ds.coords else "longitude"
    return ds.sel({latn: slice(box["lat_min"], box["lat_max"]),
                   lonn: slice(box["lon_min"], box["lon_max"])})

def area_weighted_boxmean(da: xr.DataArray) -> xr.DataArray:
    latn = "lat" if "lat" in da.coords else "latitude"
    w = np.cos(np.deg2rad(da[latn]))
    return da.weighted(w).mean(dim=[latn, "lon" if "lon" in da.coords else "longitude"])

# doad dataset
ds = open_mfdataset(FILES_GLOB, chunks={"time": 120}, engine="netcdf4")
assert VAR in ds, f"{VAR} not found in dataset"

# curve for each region
results = {}

for name, box in REGIONS.items():
    da = subset_box(ds[[VAR]], box)[VAR]
    da_box = area_weighted_boxmean(da).rename(f"sst_boxmean_{name}")
    da_box = da_box.compute()
    time_np = pd.to_datetime(da_box.time.values)          
    temp_np = np.asarray(da_box.values, dtype=float)     
    assert temp_np.shape[0] == time_np.shape[0], "time/temperature length mismatch"

    y0 = max(CLIM_YEARS[0], int(time_np.year.min()))
    y1 = min(CLIM_YEARS[1], int(time_np.year.max()))
    ords = np.array([d.toordinal() for d in time_np], dtype=int)

    # clim['seas'] and clim['thresh'] for the 90th percentile 
    res90, clim90 = mhw.detect(ords, temp_np, climatologyPeriod=[y0, y1],pctile=90, minDuration=5, joinAcrossGaps=True)
    seas_full     = np.asarray(clim90["seas"],   dtype=float)  # daily, same length as time_np
    thresh90_full = np.asarray(clim90["thresh"], dtype=float)
    assert seas_full.shape[0] == temp_np.shape[0] == time_np.shape[0]

    # -clim['seas'] and clim['thresh'] for the 80th percentile 
    res80, clim80 = mhw.detect(
        ords, temp_np, climatologyPeriod=[y0, y1],
        pctile=80, minDuration=5, joinAcrossGaps=True
    )
    thresh80_full = np.asarray(clim80["thresh"], dtype=float)
    assert thresh80_full.shape[0] == temp_np.shape[0]

    df = pd.DataFrame({"seas": seas_full, "p90": thresh90_full, "p80": thresh80_full}, index=time_np)
    g = df.groupby(df.index.dayofyear).mean(numeric_only=True)
    days = np.arange(1, 367)
    g = g.reindex(days)

    seas_366        = g["seas"].to_numpy()
    thresh_90th_366 = g["p90"].to_numpy()
    thresh_80th_366 = g["p80"].to_numpy()

    months = np.arange(1, 13)
    monthly_seas = np.array([np.nanmean(seas_366[_month_to_doy0[m]])        for m in months])
    monthly_90   = np.array([np.nanmean(thresh_90th_366[_month_to_doy0[m]]) for m in months])
    monthly_80   = np.array([np.nanmean(thresh_80th_366[_month_to_doy0[m]]) for m in months])

    results[name] = {
        "monthly_seas": monthly_seas,
        "monthly_90_thresh": monthly_90,
        "monthly_80_thresh": monthly_80,
    }

# plot the combined axis graphs
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
x = np.arange(1, 13)
region_order = list(REGIONS.keys())

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11, 8), sharex=True, constrained_layout=True)

# Common y-limits for comparability
all_vals = []
for nm in region_order:
    all_vals.extend([
        results[nm]["monthly_seas"],
        results[nm]["monthly_90_thresh"],
        results[nm]["monthly_80_thresh"],
    ])
all_vals = np.concatenate(all_vals)
ymin = float(np.nanmin(all_vals)) - 0.2
ymax = float(np.nanmax(all_vals)) + 0.2

for ax, nm in zip(axes, region_order):
    ms   = results[nm]["monthly_seas"]
    mt90 = results[nm]["monthly_90_thresh"]
    mt80 = results[nm]["monthly_80_thresh"]

    ax.plot(x, ms,   marker="o", label="Climatology (Monthly)", color="C0",lw=1.6)
    ax.plot(x, mt90, marker="s", label="90th perc Threshold",   ls="--", color="#ff7f0e", lw=1.6)
    ax.plot(x, mt80, marker="^", label="80th perc Threshold",   ls=":",  color="C2",lw=1.6)

    ax.set_ylabel("Temp (°C)")
    ax.set_title(f"{nm}: Monthly Climatology, 90th & 80th Thresholds ({CLIM_YEARS[0]}–{CLIM_YEARS[1]})")
    ax.legend(loc="upper right")
    ax.set_ylim(ymin, ymax)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(month_labels)
axes[-1].set_xlabel("Month")

plt.show()

