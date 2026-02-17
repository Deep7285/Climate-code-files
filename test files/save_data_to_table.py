import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# convert res fields in to numpy arrays with explicit dtypes
idx_start = np.array(res["index_start"], dtype=int)
idx_end   = np.array(res["index_end"],   dtype=int)
duration  = np.array(res["duration"], dtype=int)
i_max     = np.array(res["intensity_max"], dtype=float)
i_mean    = np.array(res["intensity_mean"], dtype=float)
i_cum     = np.array(res["intensity_cumulative"], dtype=float)

n_events = idx_start.size
if n_events == 0:
    raise SystemExit("No events to tabulate; re-run detection or change region.")

# map indices back to real dates using time_dt 
start_dates = [pd.to_datetime(time_dt[i]).date() for i in idx_start]
end_dates   = [pd.to_datetime(time_dt[i]).date() for i in idx_end]

event_df = pd.DataFrame({"start_date": start_dates,"end_date":   end_dates,"duration_days": duration,
    "intensity_max_degC": i_max,"intensity_mean_degC": i_mean,"cumulative_intensity_degC": i_cum,})

#  read the dataframe
event_df["start_year"] = pd.to_datetime(event_df["start_date"]).dt.year
event_df["end_year"]   = pd.to_datetime(event_df["end_date"]).dt.year

# save the csv file
out_dir = Path.cwd() / "mhw_outputs"
out_dir.mkdir(exist_ok=True)
csv_path = out_dir / "mhw_events_boxmean_lat0-30_lon40-100.csv"
event_df.to_csv(csv_path, index=False)
print(f"Saved events table → {csv_path}")

# yearly summaries
yearly = (event_df.groupby("start_year").agg(events=("duration_days", "size"),total_hw_days=("duration_days", "sum"),
         mean_max_intensity=("intensity_max_degC", "mean")).reset_index().rename(columns={"start_year": "year"}))
print("First 5 rows of yearly summary:")
print(yearly.head())

# plot the figure
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

