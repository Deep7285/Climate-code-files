#primary requirements: # python3, numpy, pandas, xarray, dask, bottleneck, netCDF4, cftime, matplotlib, marineHeatWaves
# run this script to check if all dependencies are installed and working
# also fixes np.NaN to np.nan in marineHeatWaves module

import sys, platform
import numpy as np
import pandas as pd
import xarray as xr, dask, bottleneck, netCDF4, cftime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import marineHeatWaves as mhw, re, pathlib
from importlib.metadata import version, PackageNotFoundError

# convert np.NAN to np.nan to avoid dependency issues
if not hasattr(np, "NaN"):
    np.NaN = np.nan  

#Check the package is working
try:
    mhw_ver = version('marineHeatWaves')
except PackageNotFoundError:
    mhw_ver = getattr(mhw, '__version__', 'unknown')
print('OK:', pd.__version__, xr.__version__, dask.__version__, 'mhw:', mhw_ver)
print('mhw module file:', getattr(mhw, '__file__', 'unknown'))

#checkk the marineHeatWaves module file and replace np.NaN with np.nan
p = pathlib.Path(mhw.__file__)
p.write_text(re.sub(r'\bnp\.NaN\b', 'np.nan', p.read_text()))

print(sys.executable)
print(platform.python_version())
