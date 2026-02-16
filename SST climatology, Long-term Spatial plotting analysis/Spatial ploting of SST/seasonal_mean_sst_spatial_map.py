# Seasonal mean spatial maps of Dec-Jan-Feb (DJF), March-April-May (MAM), June-Jul-Aug (JJA), Sept-Oct-Nov (SON)
da = ds[VAR]
# mean for each season across all years
seasonal = da.groupby("time.season").mean("time", skipna=True) 

# Seasons order DJF, MAM, JJA, SON
season_order = ["DJF", "MAM", "JJA", "SON"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for i, s in enumerate(season_order):
    ax = axes[i//2, i%2]
    cmap_land = CMAP.copy()
    cmap_land.set_bad(color= "#d3d3d3")
    im = ax.pcolormesh(seasonal[lonn], seasonal[latn], seasonal.sel(season=s),
                       shading="auto", cmap=cmap_land, norm=NORM)
    ax.set_title(s)
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    #x and y lables formatting
    xticks= np.arange(40, 101, 10)
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

cbar = fig.colorbar(im, ax=axes, orientation="vertical", extend="both", fraction=0.03, pad=0.02)
cbar.set_label("Temp (°C)")
cbar.set_ticks(LEVELS)
cbar.ax.set_yticklabels([f"{lv:.1f}" for lv in LEVELS])
fig.suptitle("Seasonal Mean SST of North Indian Ocean (1982–2024)", y=1.02)
plt.show()

