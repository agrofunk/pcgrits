# %%
print("""
      Vegetation Indices series extraction 
        from Landsat series
      
        ---
        created by Denis Mariano 
            denis@seca.space
                www.seca.space
                    2024-02-09
    ToDo's
      - verificar porque EVI e LAI não estão displaying no valuetool
        - TEM QUE DAR UM TRATO NOS VALUES 
      - agregar no tempo, zscores
      - plots
      - extraction   
    
        """)

# %%
import time

start = time.time()

import sys
import subprocess
import pkg_resources

required = {"rasterstats", "odc-ui"}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, "-m", "pip", "install", *missing], stdout=subprocess.DEVNULL)
else:
    print(f"Required packages {required} already installed.")

import geopandas as gpd
import stackstac
import xarray as xr
import numpy as np
import rioxarray
from scipy.signal import savgol_filter
import zipfile
from xrspatial import zonal_stats
import pandas as pd
import numpy as np

sys.path.append("/home/jovyan/PlanetaryComputerExamples/CODE/pcgrits/")
from grits import *

print("all good!")
# %% DEFINE AREA OF INTEREST
path_vector = "/home/jovyan/PlanetaryComputerExamples/vetorial/FAZENDAS/"
file = path_vector + "iaca_r400.shp"
field_ = gpd.read_file(file)
print(field_.grid.unique())
# %% # some parameters to filter scenes
indices = ["NDVI", "MSAVI"]  # EVI, LAI,"NDMI","BSI",
# assets = ['blue','green','red','nir08','swir16','swir22']
assets = ["red", "nir08"]
path_nc = "/home/jovyan/PlanetaryComputerExamples/OUT/nc/iacanga/"
max_cloud = 30

datetime = "2022-05-02/2024-02-08"

#  -> Iacanga
# '2019-06-20/2022-04-01' done
# '2022-04-02/2024-02-08' done
# '2022-05-02/2024-02-08' done
# '2013-04-02/2019-06-19'


for grid in sorted(field_.grid.unique()):
    name = f"iacanga_{grid}"
    field = field_[field_["grid"] == grid]
    bbox, lat_range, lon_range = get_lims(field)
    ha = field.area_ha.sum()
    print(f"{name} de {ha} ha - periodo {datetime}")

    items89 = query_Landsat_items(
        datetime=datetime, bbox=bbox, max_cloud=max_cloud, landsats=["landsat-8", "landsat-9"]
    )
    # get the data the lazy way
    data89 = stackstac.stack(
        items89,
        assets=assets,
        bounds_latlon=bbox,
        epsg=4326,
    )
    del data89.attrs["spec"]

    ds89 = data89.to_dataset(dim="band")
    ds = ds89.rio.write_crs("4326")

    ds_ = xr.where(ds > 60000, np.nan, ds)

    # INTERPOLATE NANs
    print("interpolating NaNs")
    ds_ = ds_.chunk(dict(time=-1))
    ds_ = ds_.interpolate_na(
        dim="time",
        method="pchip",
        # limit = 7,
        use_coordinate=True,
    )

    smooth = True
    w = 4
    sm = "pchip_smW" + str(w)
    if smooth:
        print("smoothening...")
        ds_ = ds_.chunk(dict(time=-1))
        ds_ = ds_.rolling(time=w, center=True).mean(savgol_filter, window=w, polyorder=2)

    # CALCULATE INDICES
    ds_ = ds_.rename({"nir08": "nir"})
    dsi = calculate_indices(
        ds_,
        index=indices,
        satellite_mission="ls",
        # normalise=True,
        drop=True,
    )
    #  REPROJECT
    print("reprojecting")
    dsi = dsi.rio.write_crs("4326")
    dsi = dsi.rio.reproject("EPSG:4326")
    dsi = dsi.rename({"x": "longitude", "y": "latitude"})

    # DROPPING STUFF
    drops = [
        "landsat:correction",
        "landsat:wrs_path",
        "landsat:wrs_row",
        "landsat:collection_number",
        "landsat:wrs_type",
        "instruments",
        "raster:bands",
        "sci:doi",
    ]
    dsi = dsi.drop_vars(drops)
    dsi = dsi.astype("float32")

    # Saving
    dt1 = datetime.split("/")[0]
    dt2 = datetime.split("/")[1]
    dsi.to_netcdf(f"{path_nc}/{dt1}_{dt2}_{name}.nc")
    print(f"SAVED ___ {path_nc}/{dt1}_{dt2}_{name}.nc ___SAVED ")

    del dsi, ds, ds_, data89, items89, ds89
    # %%


def fun_grits_vis():
    return

from grits_lst89_p1 import fun_grits_lst