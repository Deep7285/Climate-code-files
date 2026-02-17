from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import marineHeatWaves as mhw
import matplotlib.pyplot as plt

base_dir = Path("/home/Desktop/Noah_data_1982-2024_daily_mean/sst.day.mean.*.nc")  # update the path 

# collect files and open lazily with Dask
files = sorted(str(p) for p in glob(base_dir))
print(f"Found {len(files)} files")
if not files:
    raise FileNotFoundError(f"No files matched {base_dir}")

ds = xr.open_mfdataset(files,combine="by_coords", parallel=True, engine="netcdf4", chunks={"time": 365}) # adjust the chunk size for better memory usage 

# check the data
print("coords:", list(ds.coords))
print("data_vars:", list(ds.data_vars))
assert "sst" in ds.data_vars, "Expected variable 'sst' not found."

# subset the regional box 
ds_box = ds.sel(lat=slice(0, 30), lon=slice(40, 100))

# keep the sst unit in degC 
sst = ds_box["sst"]
units = (sst.attrs.get("units", "") or "").lower()
if units in ("k", "kelvin"):
    sst = sst - 273.15
    sst.attrs["units"] = "degC"
    print("Converted SST from Kelvin to degC")

# make a single time series by spatially averaging the regions
ts = sst.mean(dim=("lat", "lon"), skipna=True)

# convert time coordinate to Python datetime then to ORDINAL DAYS
def to_datetimeindex(dataarray):
    # try native conversion first
    try:
        return dataarray.indexes["time"].to_datetimeindex()
    except Exception:
        # calendars like noleap to convert to a standard proleptic gregorian timeline
        da2 = dataarray.convert_calendar("proleptic_gregorian")
        return da2.indexes["time"].to_datetimeindex()

time_dt = to_datetimeindex(ts)
time_ord = np.array([d.date().toordinal() for d in time_dt])  # WHAT MHW expects (ints)

temp = np.asarray(ts.values, dtype=float)

print("Series length:", temp.size, "range:", str(time_dt[0].date()), "→", str(time_dt[-1].date()))
print("Temp units:", sst.attrs.get("units", "unknown"))

# run marineHeatWaves detection on the box-mean series
res, clim = mhw.detect(time_ord, temp)

print("\n MHW detection summary of box-mean")
print("Events found:", len(res.get("index_start", [])))
if len(res.get("index_start", [])) > 0:
    i0 = 0
    print("First event: duration (days) =", int(res["duration"][i0]),
          "| max intensity (°C) =", float(res["intensity_max"][i0]))

# quick SST plotting
plt.figure()
plt.plot(time_dt, temp)
plt.title("Box-mean SST (lat 0–30, lon 40–100)")
plt.xlabel("Date")
plt.ylabel(f"SST ({sst.attrs.get('units', 'degC')})")
plt.tight_layout()
plt.show()
