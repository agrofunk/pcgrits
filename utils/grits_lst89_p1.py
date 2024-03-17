"""

Extraindo LST por region (grid) em uma grande propriedade

    Por enquanto, somente Landsat 8 e 9

        Feb, 9, 2024

"""

# %%
import time
from datetime import date
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
import pylab as plt
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

# %% Area Of Interest
# name embrapa_sc , layer talhoes, fazenda_embrapa.gpkg, column 'tid'
path_vector = "/home/jovyan/PlanetaryComputerExamples/vetorial/FAZENDAS/"
file = path_vector + "iaca_r400.shp"
layer = None
column = "grid"
field = gpd.read_file(file, layer=layer)
print(field.dtypes)


bbox, lat_range, lon_range = get_lims(field)
print(field.head())
field.plot(column=column, legend=True)


savenc = True
zscores = True


# parameters for extracting data
savecsv = True
path_csv = "/home/jovyan/PlanetaryComputerExamples/OUT/csv/"


# %%
# ### THE FUCKING FOR
#

name = "iacanga"
path_nc = "/home/jovyan/PlanetaryComputerExamples/OUT/nc/iacanga/"
max_cloud = 50
# DATETIME CONTROL
a0 = 2020
a1 = 2024
p = 1  # ano a ano
d0 = "-06-20"
d1 = "-06-20"

# for record, Iacanga
# grid 100 -> 2013 - 2019[ ok
# pulando 2019-2020

####
for i in sorted(field.grid.unique()):
    gridstart = time.time()
    print(i)
    name_ = f"{name}_{i}"
    field_ = field[field[column] == i]
    bbox, lat_range, lon_range = get_lims(field_)
    print(bbox, lat_range, lon_range)
    ha = field_.area_ha.sum()
    print(f"{name_} de {ha} ha")
    field_.plot()
    plt.show()
    plt.close()
    #
    #
    #

    for ano in range(a0, a1, p):
        pstart = time.time()
        dt0 = str(ano) + d0
        dt1 = str(ano + p) + d1
        datetime = dt0 + "/" + dt1
        print(f"periodo {datetime}, {column} = {i}")

        # items57 = query_Landsat_items(datetime=datetime,
        #                         bbox=bbox,
        #                         max_cloud=max_cloud,
        #                         landsats = [
        #                             "landsat-5", "landsat-7",
        #                                     ])
        # print('items57 created')
        items89 = query_Landsat_items(
            datetime=datetime,
            bbox=bbox,
            max_cloud=max_cloud,
            landsats=[  # "landsat-5", "landsat-7",
                "landsat-8",
                "landsat-9",
            ],
        )
        scale = items89[0].assets["lwir11"].extra_fields["raster:bands"][0]["scale"]
        offset = items89[0].assets["lwir11"].extra_fields["raster:bands"][0]["offset"]
        print(f"items89 created, scale {scale} and offset {offset}")

        # get the data the lazy way
        data89 = stackstac.stack(
            items89,
            assets=["lwir11"],
            bounds_latlon=bbox,
            epsg=4326,
            # resolution=100
        )
        data89 = data89.rename("lwir").squeeze()
        print("data89 ok!")
        print(humanbytes(data89.nbytes))
        # data57 = (
        #         stackstac.stack(
        #         items57,
        #         assets=['lwir'],
        #         bounds_latlon=bbox,
        #         epsg=4326,
        #         resolution=100
        #     ))
        # data89
        # print('data57 ok!')
        ## %% The CONCAT Way

        # MATCH REPROJECTION using rioxarray
        # print(f'matching DataArrays spatially for _{datetime}')
        # data57 = data57.rio.reproject_match(data89)

        # CONCATENATE DATAARRAYS
        # da = xr.concat([data89, data57], dim="time", join='outer')

        # RESCALE AND FILTER FOR LAND SURFACE TEMPERATURE

        da = data89.copy()

        print("reescaling LST")
        da = da * scale + offset - 273.15
        da = da.astype("float32")
        da = xr.where((da < -5) | (da > 65), np.nan, da)

        # REPROJECT
        # print(f'reprojecting_{datetime}')
        print("reprojecting...")
        da = da.rio.write_crs("4326")
        da = da.rio.reproject("EPSG:4326")
        da = da.rename({"x": "longitude", "y": "latitude"})
        print("reprojecting... done")

        # REORDER
        da = da.rename("lst")
        da = da.sortby("time")

        # INTERPOLATE NANs
        print("interpolating NaNs")
        da = da.chunk(dict(time=-1))
        da = da.interpolate_na(dim="time", method="pchip", limit=7, use_coordinate=True)
        print("interpolating NaNs... done")

        # XXX SMOOTHENING WOULD BE COOL
        smooth = True
        w = 3
        sm = "pchip_" + str(w)
        if smooth:
            print("smoothening...")
            da = da.chunk(dict(time=-1))
            da = da.rolling(time=w, center=True).mean(savgol_filter, window=w, polyorder=2)
            print("smoothing... done.")

        # DROPPING STUFF
        drops = [
            "landsat:correction",
            "landsat:wrs_path",
            "landsat:wrs_row",
            "landsat:collection_number",
            "landsat:wrs_type",
            "instruments",
            "raster:bands",
        ]
        da = da.drop_vars(drops)

        # SAVE
        print("saving...")
        da.to_netcdf(f"{path_nc}/{dt0}_{dt1}_{name}_{i}_LST_{sm}.nc")
        print(f"saving... {path_nc}/{dt0}_{dt1}_{name}_{i}_LST_{sm}.nc DONE!")
        del (
            da,
            data89,
            items89,
        )
        pend = time.time()
        print(f"{dt0}_{dt1}_{name}_{i} took {pend - pstart} seconds to complete.")

    gridend = time.time()
    print(f" Grid {i} took {(gridend - gridstart)} seconds")

def fun_grits_lst():
    return


from grits_vis_p1 import fun_grits_vis