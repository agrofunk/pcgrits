# %%
import time
start = time.time()

from datetime import date
import sys
import subprocess
import pkg_resources

required = {'rasterstats','odc-ui'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
else:
    print(f'Required packages {required} already installed.')

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

sys.path.append('/home/jovyan/PlanetaryComputerExamples/CODE/pcgrits/')
from grits import *

print('all good!')

#%%
def query_Landsat_items(datetime,
                     bbox, 
                     max_cloud = 30, 
                     landsats = ["landsat-4", "landsat-5","landsat-7",
                                 "landsat-8", "landsat-9"],
                     tiers = ['T1']
):
    '''
        query Landsat 8 and 9
    '''

    # stac object from Planetary Computer
    stac = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
    )

    # some parameters
    query_params = {
        "eo:cloud_cover": {"lt": max_cloud},
        "platform": {"in": landsats},
        "landsat:collection_category": { "in": tiers}
                }

    # search
    search = stac.search(
        bbox=bbox,
        datetime=datetime, 
        collections='landsat-c2-l2',
        query=query_params,  
    )

    # sign items
    items = planetary_computer.sign(search)

    items = search.item_collection()
    print(f'\n found {len(items)} items \n first: {items[-1]} \n last: {items[0]} \n')
    print(items[0].assets.keys())
    return items

# %% DEFINE AREA OF INTEREST
# =========================
# Name for reference
name = 'Uniguiri_farm_unifyXXX'

# AOI file and layer (for GPKG)
path_vector = '/home/jovyan/PlanetaryComputerExamples/vetorial/FAZENDAS/'
file = path_vector + 'fazenda_uniguiri.gpkg'
layer = 'piquetes_tid'

# Get FIELD
field = gpd.read_file(file, layer=layer)
#field = field[field['Re'] == 80000]

bbox, lat_range, lon_range = get_lims(field)
print(field.head())
field.plot(column='TID')

# %% Define period and output path
# Landsat 4,5,7 have 'lwir' and 8 and 9 have 'lwir11'
#datetime = '1985-05-01/'+str(date.today())
datetime='2017-07-07/2022-07-05'
print(datetime)

# Parameters to save raster data?
savenc = True
zscores = True
path_nc = '/home/jovyan/PlanetaryComputerExamples/OUT/nc/'

# parameters for extracting data
savecsv = True
column = 'TID'
path_csv = '/home/jovyan/PlanetaryComputerExamples/OUT/csv/'

# some parameters to filter scenes
max_cloud = 50
# %% QUERY LANDSAT
items57 = query_Landsat_items(datetime=datetime,
                         bbox=bbox,
                         max_cloud=max_cloud,
                         landsats = [
                             "landsat-5", "landsat-7",
                                     ])

items89 = query_Landsat_items(datetime=datetime,
                         bbox=bbox,
                         max_cloud=max_cloud,
                        landsats = [
                                "landsat-8", "landsat-9"
                                     ])

# get the data the lazy way
data89 = (
        stackstac.stack(
        items89,
        assets=['lwir11'],
        bounds_latlon=bbox,
        epsg=4326, 
    ))

data57 = (
        stackstac.stack(
        items57,
        assets=['lwir'],
        bounds_latlon=bbox,
        epsg=4326, 
    ))
# %% The CONCAT Way
# SQUEEZE monoBAND
data89 = data89.rename('lwir').squeeze()
data57 = data57.rename('lwir').squeeze()

# MATCH REPROJECTION using rioxarray
print('matching DataArrays spatially')
data57 = data57.rio.reproject_match(data89)

# CONCATENATE DATAARRAYS
da = xr.concat([data89, data57], dim="time", join='outer')

# RESCALE AND FILTER FOR LAND SURFACE TEMPERATURE
print('reescaling LST')
scale = items89[0].assets['lwir11'].extra_fields["raster:bands"][0]['scale']
offset = items89[0].assets['lwir11'].extra_fields["raster:bands"][0]['offset']
da = da*scale + offset - 273.15
da = da.astype('float32')
da = xr.where((da < -5) | (da > 65), np.nan, da)

# REPROJECT
print('reprojecting')
da = da.rio.write_crs('4326')
da = da.rio.reproject('EPSG:4326')
da = da.rename({'x': 'longitude','y': 'latitude'})

# REORDER
da = da.rename('lst')
da = da.sortby('time')

# INTERPOLATE NANs
print('interpolating NaNs')
da = da.interpolate_na(dim='time',
                       method='pchip', 
                       limit = 7,
                       use_coordinate=True)

# %% XXX SMOOTHENING WOULD BE COOL
smooth = True
w = 5
sm = 'pchip_smW'+str(w)
if smooth:
    print('smoothening...')
    da = da.chunk(dict(time=-1))
    da = da.rolling(time=w, 
                    center=True).mean(savgol_filter, 
                                              window = w, 
                                              polyorder=2)

# DROPPING STUFF
drops = ['landsat:correction','landsat:wrs_path','landsat:wrs_row',
        'landsat:collection_number','landsat:wrs_type','instruments',
        'raster:bands']
da = da.drop_vars(drops)

# Save NC
da.to_netcdf(f'{path_nc}/{name}_LST{sm}.nc')
print(f'SAVED {path_nc}/{name}_LST{sm}.nc')
# %%
if zscores:
    print('calculating zscores')
    da_mean = da.groupby('time.month').mean(dim='time')
    da_std = da.groupby('time.month').std(dim='time')

    da_anom = da.groupby('time.month') - da_mean
    da_z = da_anom.groupby('time.month') / da_std

    da_z.to_netcdf(f'{path_nc}/{name}_Z-LST{sm}.nc')
    print('zscores saved')

# %%
print(f'{time.time()-start} seconds')