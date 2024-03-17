# %%
import xarray as xr
import pandas as pd
import pylab as plt
import geopandas as gpd
import os, sys
from glob import glob

sys.path.append("/home/jovyan/PlanetaryComputerExamples/CODE/pcgrits/")
from grits import *
from xrspatial import zonal_stats

# %% Area Of Interest
# name embrapa_sc , layer talhoes, fazenda_embrapa.gpkg, column 'tid'
path_nc = "/home/jovyan/PlanetaryComputerExamples/OUT/nc/iacanga/"
path_csv = "/home/jovyan/PlanetaryComputerExamples/OUT/csv/iacanga/"
if not os.path.exists(path_csv):
    os.makedirs(path_csv)


path_vector = "/home/jovyan/PlanetaryComputerExamples/vetorial/FAZENDAS/"
file = path_vector + "iacanga_r400.shp"
layer = None
column = "grid"
attribute_col = "TID"
field = gpd.read_file(file, layer=layer)
print(field.dtypes)

# %%


# XXX XXX XXX
# ano, precisa ir por ANOOOO


r = 200
band = "lst"

field_ = field[field[column] == r]

bbox, lat_range, lon_range = get_lims(field_)
print(field_.head())
field_.plot(column=attribute_col, legend=True)


# %%
files = sorted(glob(path_nc + "*_" + str(r) + "_LST_*"))
print(files)

# %% Starting stats extraction
dst = xr.open_dataset(files[0], chunks=64)
fz = xr_rasterize(field_, dst, attribute_col="TID", verbose=True)
tempo = pd.to_datetime(dst.time.values)
fz = fz.chunk(64)

# %% first date
data_ = dst.sel(time=tempo[0]).squeeze()
print(f"computing stats for {tempo[0]}")
outst = zonal_stats(zones=fz, values=data_[band]).compute()

# %% the whole thing
data_.close()
for t in tempo[1:]:
    data_ = dst[band].sel(time=t).squeeze()
    print(f"computing stats for {t}")

    outst1 = zonal_stats(zones=fz, values=data_).compute()
    outst1["date"] = str(t)
    outst = pd.concat([outst, outst1])
    data_.close()
    del outst1

outst.to_csv(f"{path_csv}/{r}_2020-2024_{band}.csv")
print(f"{path_csv}/{r}_2020-2024_{band}.csv \n")

# %%
