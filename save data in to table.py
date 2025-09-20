import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- WHAT: build a tidy table from res + time_dt ---
n_events = len(res.get("index_start", []))
if n_events == 0:
    raise SystemExit("No events to tabulate; re-run detection or change region.")

idx_start = np.array(res["index_start"], dtype=int)
idx_end   = np.array(res["index_end"],   dtype=int)

event_df = pd.DataFrame({
    "start_date": [pd.to_datetime(time_dt[i]).date() for i in idx_start],
    "end_date":   [pd.to_datetime(time_dt[i]).date() for i in idx_end],
    "duration_days": res["duration"].astype(int),
    "intensity_max_degC": res["intensity_max"].astype(float),
    "intensity_mean_degC": res["intensity_mean"].astype(float),
    "cumulative_intensity_degC": res["intensity_cumulative"].astype(float)
})

# Add helper columns
event_df["start_year"] = pd.DatetimeIndex(event_df["start_date"]).year
event_df["end_year"]   = pd.DatetimeIndex(event_df["end_date"]).year

# --- WHY: save for later analysis/sharing ---
out_dir = Path.cwd() / "mhw_outputs"
out_dir.mkdir(exist_ok=True)
csv_path = out_dir / "mhw_events_boxmean_lat0-30_lon40-100.csv"
event_df.to_csv(csv_path, index=False)
print(f"Saved events table → {csv_path}")

# --- HOW: quick yearly summaries ---
yearly = (
    event_df
    .groupby("start_year")
    .agg(events=("duration_days", "size"),
         total_hw_days=("duration_days", "sum"),
         mean_max_intensity=("intensity_max_degC", "mean"))
    .reset_index()
    .rename(columns={"start_year": "year"})
)

print("First 5 rows of yearly summary:")
print(yearly.head())

# --- PLOTS: events/year and total HW days/year (matplotlib only) ---
plt.figure()
plt.plot(yearly["year"], yearly["events"], marker="o")
plt.title("Marine heatwave events per year (box-mean 0–30N, 40–100E)")
plt.xlabel("Year"); plt.ylabel("Events")
plt.tight_layout(); plt.show()

plt.figure()
plt.plot(yearly["year"], yearly["total_hw_days"], marker="o")
plt.title("Total marine heatwave days per year (box-mean 0–30N, 40–100E)")
plt.xlabel("Year"); plt.ylabel("Days")
plt.tight_layout(); plt.show()
