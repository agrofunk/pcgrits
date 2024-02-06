#%%
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

sys.path.append('/home/jovyan/PlanetaryComputerExamples/CODE/pcgrits/')
from grits import *



name = 'embrapa_sc'
path_vector = '/home/jovyan/PlanetaryComputerExamples/vetorial/FAZENDAS/'
file = path_vector + 'fazenda_embrapa.gpkg'
layer = 'talhoes'
field = gpd.read_file(file, layer=layer)


bbox, lat_range, lon_range = get_lims(field)
print(field.head())
field.plot()#column='tid'
plt.title(name)


savenc = True
zscores = True
path_nc = '/home/jovyan/PlanetaryComputerExamples/OUT/nc/'

# parameters for extracting data
savecsv = True
column = 'TID'
path_csv = '/home/jovyan/PlanetaryComputerExamples/OUT/csv/'






# %% QUERY LANDSAT

max_cloud = 50
for ano in range(1985,2025,1):
    dt1 = str(ano)+'-06-20'
    dt2 = str(ano+1)+'-06-20'
    datetime = dt1 + '/' + dt2
    print(datetime)

    items57 = query_Landsat_items(datetime=datetime,
                            bbox=bbox,
                            max_cloud=max_cloud,
                            landsats = [
                                "landsat-5", "landsat-7",
                                        ])
    print('items57 created')
    items89 = query_Landsat_items(datetime=datetime,
                            bbox=bbox,
                            max_cloud=max_cloud,
                            landsats = [
                                    "landsat-8", "landsat-9"
                                        ])
    print('items89 created')
    # get the data the lazy way
    data89 = (
            stackstac.stack(
            items89,
            assets=['lwir11'],
            bounds_latlon=bbox,
            epsg=4326, 
            resolution=100
        ))
    print('data89 ok!')
    data57 = (
            stackstac.stack(
            items57,
            assets=['lwir'],
            bounds_latlon=bbox,
            epsg=4326, 
            resolution=100
        ))
    print('data57 ok!')
    # %% The CONCAT Way
    # SQUEEZE monoBAND
    data89 = data89.rename('lwir').squeeze()
    data57 = data57.rename('lwir').squeeze()

    # MATCH REPROJECTION using rioxarray
    print(f'matching DataArrays spatially for _{datetime}')
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
    print(f'reprojecting_{datetime}')
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
    w = 7
    sm = 'pchip_'+str(w)
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

    #SAVE
    print('saving...')
    da.to_netcdf(f'{path_nc}/{dt1}_{dt2}_{name}_LST_{sm}.nc')
    print(f'{path_nc}/{dt1}_{dt2}_{name}_LST_{sm}.nc')