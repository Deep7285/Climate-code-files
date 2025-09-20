import numpy as np, pandas as pd, matplotlib.pyplot as plt

# --- this cell assumes you still have from Step 6:
# time_dt (DatetimeIndex), temp (box-mean SST 1D), res, clim
# if the kernel was restarted, re-run Step 6 first.

# 1) extract seasonal climatology & threshold returned by MHW
seas   = np.asarray(clim["seas"], dtype=float)      # seasonal climatology (degC), length ~366
thresh = np.asarray(clim["thresh"], dtype=float)    # 90th percentile threshold (degC)

# build day-of-year (1..366) for plotting
doy_full = np.arange(1, len(seas) + 1)

# 2) pick a sample year to overlay (change if you want another, e.g., 2019, 2023)
year_sel = 2019
mask_y   = pd.DatetimeIndex(time_dt).year == year_sel
doy_y    = np.array([d.timetuple().tm_yday for d in time_dt[mask_y]])
temp_y   = temp[mask_y]

# 3) plot: climatology + threshold + selected year SST
plt.figure()
plt.plot(doy_full, seas, label="Climatology (seas)")
plt.plot(doy_full, thresh, label="90th percentile threshold")
plt.scatter(doy_y, temp_y, s=6, label=f"SST in {year_sel} (box-mean)")
plt.title("Seasonal climatology & threshold vs SST")
plt.xlabel("Day of year"); plt.ylabel("°C"); plt.legend(); plt.tight_layout(); plt.show()

# 4) (optional) show the first detected event on the full series
starts = np.asarray(res["index_start"], dtype=int)
ends   = np.asarray(res["index_end"],   dtype=int)
if starts.size:
    s0, e0 = starts[0], ends[0]
    plt.figure()
    plt.plot(time_dt, temp, lw=0.8)
    plt.axvspan(time_dt[s0], time_dt[e0], color="orange", alpha=0.3, label="First event")
    plt.title("Box-mean SST with first detected event highlighted")
    plt.xlabel("Date"); plt.ylabel("°C"); plt.legend(); plt.tight_layout(); plt.show()
