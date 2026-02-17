import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
# base_dir = Path("/home/CAS_deepak/Noah_data_1982-2024_daily_mea")
# files = sorted(str(p) for p in base_dir.glob("sst.day.mean.*.nc"))
# ds = xr.open_mfdataset(files, combine="by_coords", parallel=True, engine="netcdf4", chunks={"time": 365})

# single-day map (first day of dataset)
day0 = ds.isel(time=0).sel(lat=slice(0, 30), lon=slice(40, 100))
plt.figure()
day0["sst"].plot()
plt.title(f"SST map on {str(ds['time'].isel(time=0).values)[:10]} (0–30N, 40–100E)")
plt.tight_layout()
plt.show()

# annual mean map for a chosen year
year_sel = 2019
annual = (ds.sel(time=str(year_sel)).sel(lat=slice(0, 30), lon=slice(40, 100))["sst"].mean("time", skipna=True))
plt.figure()
annual.plot()
plt.title(f"Annual mean SST in {year_sel} (0–30N, 40–100E)")
plt.tight_layout()
plt.show()
