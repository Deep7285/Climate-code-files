# Seasonal Climatology, 90th and 80th percentile Threshold of Arabian Sea, Bay Of Bengal, North Indian Ocean

from glob import glob
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt

CLIM_YEARS: Tuple[int,int] = (1982, 2024) 
SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

season_order = ["DJF", "MAM", "JJA", "SON"]
OUTDIR = Path("outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)

# Leap-aware DOY→month mapping (using leap year 2000)
ref = pd.date_range("2000-01-01", "2000-12-31", freq="D")
doy_month = ref.month.values              
doy_index = np.arange(1, 367)             
# Precompute mask for each season over DOY
season_to_doymask0 = {
    s: np.isin(doy_month, months)         
    for s, months in SEASONS.items()
}

def open_ds(globpat, chunks={"time": 120}, engine="netcdf4"):
    return xr.open_mfdataset(sorted(glob(globpat)), combine="by_coords",
                             chunks=chunks, engine=engine)

def subset_box(ds: xr.Dataset, box: Dict[str, float]) -> xr.Dataset:
    latn = "lat" if "lat" in ds.coords else "latitude"
    lonn = "lon" if "lon" in ds.coords else "longitude"
    return ds.sel({latn: slice(box["lat_min"], box["lat_max"]),
                   lonn: slice(box["lon_min"], box["lon_max"])})

def area_weighted_boxmean(da: xr.DataArray) -> xr.DataArray:
    latn = "lat" if "lat" in da.coords else "latitude"
    w = np.cos(np.deg2rad(da[latn]))
    return da.weighted(w).mean(dim=[latn, "lon" if "lon" in da.coords else "longitude"])

def boxmean_doy_curves_seas(da_box: xr.DataArray, baseline=(1982,2024)):

    da_box = da_box.compute()

    t = pd.to_datetime(da_box.time.values)        
    temp = np.asarray(da_box.values, dtype=float)
    assert len(t) == len(temp), "time/temperature length mismatch"

    # Clip baseline to available data
    y0 = max(baseline[0], int(t.year.min()))
    y1 = min(baseline[1], int(t.year.max()))

    ords = np.array([d.toordinal() for d in t], dtype=int)

    # clim['seas'] and clim['thresh'] for the 90th percentile 
    res90, clim90 = mhw.detect(
        ords, temp, climatologyPeriod=[y0, y1],
        pctile=90, minDuration=5, joinAcrossGaps=True
    )
    seas_full     = np.asarray(clim90["seas"],   dtype=float)  # same seas for any pctile
    thresh90_full = np.asarray(clim90["thresh"], dtype=float)
    assert seas_full.shape[0] == len(t) and thresh90_full.shape[0] == len(t)

    # clim['seas'] and clim['thresh'] for the 80th percentile 
    res80, clim80 = mhw.detect(
        ords, temp, climatologyPeriod=[y0, y1],
        pctile=80, minDuration=5, joinAcrossGaps=True
    )
    thresh80_full = np.asarray(clim80["thresh"], dtype=float)
    assert thresh80_full.shape[0] == len(t)

    # Build DataFrame aligned on SAME index, then DOY groupby
    df = pd.DataFrame(
        {"seas": seas_full, "p90": thresh90_full, "p80": thresh80_full},
        index=t
    )
    g = df.groupby(df.index.dayofyear).mean(numeric_only=True)   
    g = g.reindex(doy_index)                                    

    return g["seas"].to_numpy(), g["p90"].to_numpy(), g["p80"].to_numpy()

#load dataset
ds0 = open_ds(FILES_GLOB, engine="netcdf4")

results = {}
for name, box in REGIONS.items():
    ds_r   = subset_box(ds0[[VAR]], box)
    da_box = area_weighted_boxmean(ds_r[VAR]).rename("sst_boxmean")

    seas_doy, p90_doy, p80_doy = boxmean_doy_curves_seas(da_box, CLIM_YEARS)

    # Seasonal aggregation from DOY
    seas_vals   = []
    p90_vals    = []
    p80_vals    = []
    for s in season_order:
        mask0 = season_to_doymask0[s]     
        seas_vals.append(  np.nanmean(seas_doy[mask0]) )
        p90_vals.append(   np.nanmean(p90_doy[mask0]) )
        p80_vals.append(   np.nanmean(p80_doy[mask0]) )
    results[name] = {
        "seas": np.array(seas_vals),
        "p90":  np.array(p90_vals),
        "p80":  np.array(p80_vals),
    }

# plot the combined axis graphs
x = np.arange(4)  # 4 seasons
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11, 8), sharex=True, constrained_layout=True)

# common y-limits across regions and curves
all_vals = np.concatenate([np.r_[v["seas"], v["p90"], v["p80"]] for v in results.values()])
ymin = float(np.nanmin(all_vals)) - 0.2
ymax = float(np.nanmax(all_vals)) + 0.2

for ax, nm in zip(axes, results.keys()):
    ms   = results[nm]["seas"]
    mt90 = results[nm]["p90"]
    mt80 = results[nm]["p80"]

    ax.plot(x, ms,   marker="o",  lw=1.6, color="C0",     label="Climatology (Seasonal)")
    ax.plot(x, mt90, marker="s",  lw=1.6, ls="--", color="#ff7f0e", label="90th percentile threshold")
    ax.plot(x, mt80, marker="^",  lw=1.6, ls=":",  color="C2",      label="80th percentile threshold")
    ax.set_ylabel("Temp (°C)")
    ax.set_title(f"{nm}: Seasonal Climatology, 90th & 80th thresholds ({CLIM_YEARS[0]}–{CLIM_YEARS[1]})")
    ax.legend(loc="upper right")
    ax.set_ylim(ymin, ymax)

axes[-1].set_xticks(x, season_order)
axes[-1].set_xlabel("Season")
plt.show()
