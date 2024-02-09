
#%%
import pylab as plt
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




#%% Embrapa Sao Carlos
path_vector = '/home/jovyan/PlanetaryComputerExamples/vetorial/FAZENDAS/'
file = path_vector + 'fazenda_embrapa.gpkg'
layer = 'talhoes'

# Get FIELD
field = gpd.read_file(file, layer=layer)
#field = field[field['Re'] == 80000]

bbox, lat_range, lon_range = get_lims(field)
print(field.head())
field.plot(column='tid')

# to save
savenc = True
zscores = True
path_nc = '/home/jovyan/PlanetaryComputerExamples/OUT/nc/'
# parameters for extracting data
savecsv = True
column = 'TID'
path_csv = '/home/jovyan/PlanetaryComputerExamples/OUT/csv/'

# some parameters to filter scenes
max_cloud = 70

name = 'embrapa_sc_testeyearly' 



indices = ["NDVI","MSAVI","NDMI","BSI"] # EVI, LAI
assets = ['blue','green','red','nir08','swir16','swir22']

# %%
###    THE FUCKING for
###

for ano in range(2022,2023):
    dt1 = str(ano)+'-11-15'
    dt2 = str(ano+1)+'-06-20'

    datetime = dt1 + '/' + dt2
    print(datetime)
    # get items
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

    # get Data
    data89 = (
            stackstac.stack(
            items89,
            assets=assets,
            bounds_latlon=bbox,
            epsg=4326, 
            ))
    del data89.attrs['spec']

    data57 = (
            stackstac.stack(
            items57,
            assets=assets,
            bounds_latlon=bbox,
            epsg=4326, 
        ))
    del data57.attrs['spec']

    # Match, Repro, Concat
    print(f'matching datasets ... ')
    ds57 = data57.to_dataset(dim='band')
    ds57 = ds57.rio.write_crs('4326')
    ds89 = data89.to_dataset(dim='band')
    ds89 = ds89.rio.write_crs('4326')

    ds57 = ds57.rio.reproject_match(ds89)

    ds = xr.concat([ds57, ds89 ], dim="time", join='outer')
    ds = ds.sortby('time')
    ds = ds.chunk(dict(time=-1))

    # data wrangling
    ds = xr.where(ds > 65000, np.nan, ds)

    print('interpolating NaNs')
    ds = ds.interpolate_na(dim='time',
                       method='pchip',  #pchip
                       #limit = 7,
                       use_coordinate=True)

    smooth = True
    w = 4
    sm = 'pchip_w'+str(w)
    if smooth:
        print('smoothening...')
        ds = ds.rolling(time=w, 
                        center=True).mean(savgol_filter, 
                                                window = w, 
                                                polyorder=2)

    # CALCULATE INDICES
    ds = ds.rename({'nir08':'nir'})
    dsi = calculate_indices(ds, 
                            index= indices, 
                            satellite_mission='ls',
                            drop=True)
    
    print('reprojecting')
    dsi = dsi.rio.write_crs('4326')
    dsi = dsi.rio.reproject('EPSG:4326')
    dsi = dsi.rename({'x': 'longitude','y': 'latitude'})

    # DROPPING STUFF
    dsi = dsi.astype('float32')

    drops = ['landsat:correction','landsat:wrs_path','landsat:wrs_row',
            'landsat:collection_number','landsat:wrs_type','instruments',
            'raster:bands','sci:doi']
    dsi = dsi.drop_vars(drops)

    #SAVE
    print('saving...')
    dsi.to_netcdf(f'{path_nc}/{dt1}_{dt2}_{name}.nc')
    print(f'{path_nc}/{dt1}_{dt2}_{name}.nc saved')

    del dsi, ds, ds57, data57, items57, ds89, data89, items89

# %%
