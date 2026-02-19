# The Script compute and plot the  combined bar graph of AS, BoB and NIO regions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# CSV Files path for event analysis
as_event_file  = "/home/Desktop/arabian_sea/mhw_region_mean_events_per_year.csv"
bob_event_file = "/home/Desktop/bay_of_bengal/mhw_region_mean_events_per_year.csv"
nio_event_file = "/home/Desktop/north_indian_ocean/mhw_region_mean_events_per_year.csv"   #provide the proper path here

# Load the csv file
as_event_df  = pd.read_csv(as_event_file)
bob_event_df = pd.read_csv(bob_event_file)
nio_event_df = pd.read_csv(nio_event_file)

# Rename columns
as_event_df  = as_event_df.rename(columns={"events_per_year": "AS"})
bob_event_df = bob_event_df.rename(columns={"events_per_year": "BoB"})
nio_event_df = nio_event_df.rename(columns={"events_per_year": "NIO"})

# merge into single dataframe
event_df = as_event_df.merge(bob_event_df, on="year").merge(nio_event_df, on="year")
event_years = event_df["year"].values

# Combine the regions bar into one graph
x = np.arange(len(event_years))
w = 0.25   # adjust the bar width

# Plot the figure
fig, ax = plt.subplots(figsize=(14,6))
ax.bar(x - w, event_df["AS"],  width=w, color="#0072B2",  label="Arabian Sea")   # blue 
ax.bar(x,     event_df["BoB"], width=w, color= "#D55E00", label="Bay of Bengal")  # vermillion
ax.bar(x + w, event_df["NIO"], width=w, color= "#009E73", label="North Indian Ocean")  # bluish green

ax.set_title("MHW Events Comparison of AS, BoB and NIO (1982–2024)")
ax.set_ylabel("Events / year")
ax.set_xlabel("Year")

# 5-year ticks
tick_idx = np.arange(0, len(event_years), 3)
ax.set_xticks(tick_idx)
ax.set_xticklabels(event_years[tick_idx])
ax.legend()
ax.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.show()


