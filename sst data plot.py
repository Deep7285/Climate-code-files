# --- CONFIG (edit only the base_dir if your folder differs) ---
from pathlib import Path
base_dir = Path("/home/deepak/Desktop/CAS_deepak/Noah_data_1982-2024_daily_mean")  # <-- update if needed
pattern  = "sst.day.mean.*.nc"   # all yearly SST files

# --- WHAT we're using ---
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import marineHeatWaves as mhw
import matplotlib.pyplot as plt

# --- HOW: Collect files and open lazily with Dask (safe for large sets) ---
files = sorted(str(p) for p in base_dir.glob(pattern))
print(f"Found {len(files)} files")
if not files:
    raise FileNotFoundError(f"No files matched {base_dir}/{pattern}")

ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    parallel=True,
    engine="netcdf4",
    chunks={"time": 365}   # WHY: chunk by ~1 year for smoother memory use
)

# quick sanity
print("coords:", list(ds.coords))
print("data_vars:", list(ds.data_vars))
assert "sst" in ds.data_vars, "Expected variable 'sst' not found."

# --- HOW: Subset your box (lat 0..30, lon 40..100). NOTE: lon is 0..360 already. ---
ds_box = ds.sel(lat=slice(0, 30), lon=slice(40, 100))

# --- WHY: ensure units are degC (convert if Kelvin) ---
sst = ds_box["sst"]
units = (sst.attrs.get("units", "") or "").lower()
if units in ("k", "kelvin"):
    sst = sst - 273.15
    sst.attrs["units"] = "degC"
    print("Converted SST from Kelvin → degC")

# --- HOW: Make a single time series by spatially averaging the box ---
ts = sst.mean(dim=("lat", "lon"), skipna=True)

# --- HOW: Convert time coordinate to Python datetime (robust to cftime), then to ORDINAL DAYS ---
def to_datetimeindex(dataarray):
    # try native conversion first
    try:
        return dataarray.indexes["time"].to_datetimeindex()
    except Exception:
        # calendars like noleap → convert to a standard proleptic gregorian timeline
        da2 = dataarray.convert_calendar("proleptic_gregorian")
        return da2.indexes["time"].to_datetimeindex()

time_dt = to_datetimeindex(ts)
time_ord = np.array([d.date().toordinal() for d in time_dt])  # WHAT MHW expects (ints)

temp = np.asarray(ts.values, dtype=float)

print("Series length:", temp.size, "range:", str(time_dt[0].date()), "→", str(time_dt[-1].date()))
print("Temp units:", sst.attrs.get("units", "unknown"))

# --- RUN marineHeatWaves detection on the box-mean series ---
res, clim = mhw.detect(time_ord, temp)

print("\n=== Detection summary (box-mean) ===")
print("Events found:", len(res.get("index_start", [])))
if len(res.get("index_start", [])) > 0:
    i0 = 0
    print("First event: duration (days) =", int(res["duration"][i0]),
          "| max intensity (°C) =", float(res["intensity_max"][i0]))

# --- QUICK LOOK PLOT (WHAT: raw SST series; WHY: sanity check) ---
plt.figure()
plt.plot(time_dt, temp)
plt.title("Box-mean SST (lat 0–30, lon 40–100)")
plt.xlabel("Date")
plt.ylabel(f"SST ({sst.attrs.get('units', 'degC')})")
plt.tight_layout()
plt.show()
