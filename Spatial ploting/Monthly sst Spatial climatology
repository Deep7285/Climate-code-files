# Monthly mean spatial maps
ds, latn, lonn = open_roi(FILES_GLOB, ROI, YEARS, engine="netcdf4")
da = ds[VAR]
# mean for each calendar month across all years  
monthly = da.groupby("time.month").mean("time", skipna=True)

# Plot a 3x4 grid
fig, axes = plt.subplots(3, 4, figsize=(12, 7), constrained_layout=True)
for m in range(1, 13):
    ax = axes[(m-1)//4, (m-1)%4]
    cmap_land = CMAP.copy()
    cmap_land.set_bad(color= "#d3d3d3")
    im = ax.pcolormesh(monthly[lonn], monthly[latn], monthly.sel(month=m), shading="auto", cmap=cmap_land, norm=NORM)
    ax.set_title(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    #x and y lables formatting
    xticks= np.arange(40, 101, 20)
    yticks= np.arange(0, 26, 5)
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    xlabels= [f"{lon}°E" for lon in xticks]
    ylabels= []
    for lat in yticks:
        if lat == 0:
            ylabels.append("0°")
        else:
            ylabels.append(f"{lat}°N")
    ax.set_xticklabels(xlabels); ax.set_yticklabels(ylabels)

# shared colorbar 
cbar = fig.colorbar(im, ax=axes, orientation="vertical", extend="both", fraction=0.025, pad=0.02)
cbar.set_label("Temp (°C)")
cbar.set_ticks(LEVELS)
cbar.ax.set_yticklabels([f"{lv:.1f}" for lv in LEVELS])
fig.suptitle("Monthly Mean SST of North Indian Ocean (1982–2024)", y=1.02)
plt.show()
