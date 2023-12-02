# %%
print('''
      Vegetation Indices series extractor
        created by Denis Mariano 
        www.seca.space
        denis@seca.space
        ''')

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
# %% DEFINE AREA OF INTEREST
# =========================
# Name for reference
name = 'Uniguiri_full_'

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

# %% THE CAR WAY
'''
a CAR MT-5103601-948E6FB555E3445CB7E0538F61483371
 XXX ler o gpkg do MT leva 30 segundos, não está bom
 
'''
car = 'MT-5103601-948E6FB555E3445CB7E0538F61483371'
if car:
    name = car
    gdf = gpd.read_file('/home/jovyan/PlanetaryComputerExamples/vetorial/CAR/MT_CAR_AREA_IMOVEL_.gpkg')
    field = gdf[gdf['cod_imovel'] == name]

    bbox, lat_range, lon_range = get_lims(field)
    print(field.head())
    del gdf
    field.plot()



# %% Define period and output path
# Landsat 4,5,7 have 'lwir' and 8 and 9 have 'lwir11'
datetime='1985-01-01/'+str(date.today())
#datetime='2015-01-01/2017-01-01'
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

# %% LOAD BANDS
indices = ["NDVI","LAI", "EVI", "BSI"]
assets = ['blue','green','red','nir08','swir16']
#assets = ['lwir11']


# get the data the lazy way
data89 = (
        stackstac.stack(
        items89,
        assets=assets,
        bounds_latlon=bbox,
        epsg=4326, 
    ))

data57 = (
        stackstac.stack(
        items57,
        assets=assets,
        bounds_latlon=bbox,
        epsg=4326, 
    ))
# %% The CONCAT Way
# %% SQUEEZE monoBAND
# data89 = data89.rename('lwir').squeeze()
# data57 = data57.rename('lwir').squeeze()

# %%MATCH REPROJECTION using rioxarray
# print('matching DataArrays spatially')
ds57 = data57.to_dataset(dim='band')
ds57 = ds57.rio.write_crs('4326')
ds89 = data89.to_dataset(dim='band')
ds89 = ds89.rio.write_crs('4326')

ds57 = ds57.rio.reproject_match(ds89)

#%% CONCATENATE DATAARRAYS
# ds = xr.concat([ds89, ds57], dim="time", 
#                join='outer',
#                compat='override')

ds = xr.merge([ds89,ds57], compat='override', join='outer')
ds = ds.sortby('time')

# %% RESCALE AND FILTER FOR LAND SURFACE TEMPERATURE
# print('reescaling LST')
# scale = items89[0].assets['lwir11'].extra_fields["raster:bands"][0]['scale']
# offset = items89[0].assets['lwir11'].extra_fields["raster:bands"][0]['offset']
# da = da*scale + offset - 273.15
# da = da.astype('float32')
# da = xr.where((da < -5) | (da > 65), np.nan, da)

# REPROJECT
# print('reprojecting')
# da = da.rio.write_crs('4326')
# da = da.rio.reproject('EPSG:4326')
# da = da.rename({'x': 'longitude','y': 'latitude'})

# %% REORDER
#da = da.rename('lst')
# da = da.sortby('time')

#%%
#ds = da.to_dataset(dim='band')
ds = ds.rename({'nir08':'nir'})

#%%
dsi = calculate_indices(ds, 
                       index= indices, 
                       satellite_mission='ls', 
                       drop=True);

# XXX OS INDICES SAO GERADOS APARENTEMENTE OK

import pylab as plt
lat, lon = -15.80757, -54.82775

#%%
dsi['BSI'].sel(y=lat, x=lon, method='nearest').plot()


 #%%
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

print(f'{time.time()-start} seconds')



# XXX XXX XXX XXX ... ,,, XXX XXX
# %% THE EXTRACTION MISSION
def mask_farm(field,dst):
    
    mask = xr_rasterize(field,dst) 
    # #mask data
    dst = dst.where(mask)
    # #convert to float 32 to conserve memory
    #ds = ds.astype(np.int16) * 1000
    dst = dst.astype(np.float32)
    print('Farm masked outside of boundaries!')
    return dst

if savecsv:
    print('Masking farm')
    dam = mask_farm(field,da)

# %% Create zones for paddocks 
def farm_zones(field,data,column,ochunk=64):
        
    fz = xr_rasterize(field,data,attribute_col=column,verbose=True)
    fz = fz.chunk(ochunk)
    fz.astype('int16')
    return fz

start = time.time()

def extract_fz_timeseries(dst, data, field, column, path_csv, name, suffix, band, ochunk=64, verbose=False):
    '''
        Extract time-series for farm zones for one variable
        band is, for example in LST 89, 'lwir11'
    '''
    fz = farm_zones(field,data,column,ochunk)
    tozip = []
    dstrc = dst.chunk(ochunk)

    #
    tempo = pd.to_datetime(dam.time.values)
    anos = np.unique([str(x) for x in tempo.year])

    for ano in anos[:-1]:

    # get stats for the first dataframe
        print(f'working on {ano}')
        data_ = dstrc[band].sel(time=dstrc[band].time.values[0]).squeeze()
        data_ = data_.sel(time=slice(ano+'-01-01',str(int(ano)+1)+'12-31'))
        print(f'computing stats for the first date of year {ano}')
        outst = zonal_stats(zones=fz, values=data_).compute()
        outst['date'] = str(dstrc[band].time.values[0])
        data_.close()

        # and through the loop
        for t in dstrc.time.values[1:]:
            data_ = dstrc[band].sel(time=t).squeeze()
            if verbose: print(f'computing stats for {t}')
            
            outst1 = zonal_stats(zones=fz, values=data_).compute()

            outst1['date'] = str(t)
            outst = pd.concat([outst,outst1])
            data_.close()
            del outst1
        namestr = f'{path_csv}/{name}_{band}_{ano}_{suffix}.csv'
        #tozip.append(namestr)
        outst.to_csv(namestr)
        print(f'{namestr} SAVED \n \n')
        del outst, dstrc, data_

    # if zip:
    #     with zipfile.ZipFile(f'{path_csv}/{name}_{band}.zip', 'w') as zipMe:        
    #         for file in tozip:
    #             zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

# %%
ds = da.to_dataset()
extract_fz_timeseries(dams, 
                      da, 
                      field, 
                      'TID', 
                      path_csv, 
                      name, 
                      'allLandsat', 
                      'lst', 
                      ochunk=64, verbose=False)
# %%
