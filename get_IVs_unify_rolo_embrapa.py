# %%
print('''
      Vegetation Indices series extraction 
        from Landsat series
      
        ---
        created by Denis Mariano 
            denis@seca.space
                www.seca.space
                    2023-12
    ToDo's
      - verificar porque EVI e LAI não estão displaying no valuetool
        - TEM QUE DAR UM TRATO NOS VALUES 
      - agregar no tempo, zscores
      - plots
      - extraction
      - o pior de todos - datas faltantes   
    
        ''')

# %%
import time
start = time.time()

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
# %% DEFINE AREA OF INTEREST
# =========================

# %% THE CAR WAY

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

# %% Define period and output path
# 2022-04-02, '2022-11-10' - embrapa sanca
# '2022-07-01' - tabapora mt
name = 'embrapa_saocarlos' 
#dt1 = '1985-01-01'
dt1 = '2020-11-10'
dt2 = '2022-04-01'

#dt1 = '2022-11-15'
#dt2 = '2022-10-28'

#dt2 = str(date.today())

datetime = dt1 + '/' + dt2

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
max_cloud = 70
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
indices = ["NDVI","MSAVI","NDMI","BSI","LAI"] # EVI, LAI
assets = ['blue','green','red','nir08','swir16','swir22']
# get the data the lazy way
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

# %% MATCH REPROJECTION using rioxarray
print(f'matching datasets ... ')
ds57 = data57.to_dataset(dim='band')
ds57 = ds57.rio.write_crs('4326')
ds89 = data89.to_dataset(dim='band')
ds89 = ds89.rio.write_crs('4326')

ds57 = ds57.rio.reproject_match(ds89)

#%% CONCAT DATASETS
ds = xr.concat([ds57, ds89 ], dim="time", join='outer')
ds = ds.sortby('time')
ds = ds.chunk(dict(time=-1))


#%%
# if pau:
#     ds2 = ds.copy()
#     #%%
#     dss = xr.concat([ds2,ds], dim="time", join='outer')
#     dss = dss.sortby('time')
#     if dss:
#         print('reprojecting')
#         dss = dss.rio.write_crs('4326')
#         dss = dss.rio.reproject('EPSG:4326')
#         dss = dss.rename({'x': 'longitude','y': 'latitude'})
#         ds_ = xr.where(dss > 50000, np.nan, dss)
    
# #%% REPROJECT
# print('reprojecting')
# ds = ds.rio.write_crs('4326')
# ds = ds.rio.reproject('EPSG:4326')
# ds = ds.rename({'x': 'longitude','y': 'latitude'})

#%% clip nodata and run simple diagnostic 
ds = xr.where(ds > 65000, np.nan, ds)

for var in list(ds.data_vars):
    print(var, ds[var].quantile([.01,.1,.5,.9,.99], skipna=True).values, '\n')

 #%%
%%time
# INTERPOLATE NANs
print('interpolating NaNs')
ds = ds.interpolate_na(dim='time',
                       method='pchip',  #pchip
                       #limit = 7,
                       use_coordinate=True)

# %% XXX SMOOTHENING WOULD BE COOL
%%time
smooth = True
w = 4
sm = 'pchip_w'+str(w)
if smooth:
    print('smoothening...')
    ds = ds.rolling(time=w, 
                    center=True).mean(savgol_filter, 
                                              window = w, 
                                              polyorder=2)



#%% CALCULATE INDICES
ds = ds.rename({'nir08':'nir'})
dsi = calculate_indices(ds, 
                        index= indices, 
                        satellite_mission='ls',
                        #normalise=True, 
                        drop=True);
#%% REPROJECT
print('reprojecting')
dsi = dsi.rio.write_crs('4326')
dsi = dsi.rio.reproject('EPSG:4326')
dsi = dsi.rename({'x': 'longitude','y': 'latitude'})
#%%
# DROPPING STUFF
dsi = dsi.astype('float32')

drops = ['landsat:correction','landsat:wrs_path','landsat:wrs_row',
        'landsat:collection_number','landsat:wrs_type','instruments',
        'raster:bands','sci:doi']
dsi = dsi.drop_vars(drops)

#%%
dsi.to_netcdf(f'{path_nc}/{dt1}_{dt2}_{iv}_{name}_.nc')
#XXX BSI e NDVI ok, LAI e EVI weird

# #%%
# for iv in indices:
#     dsi[iv].to_netcdf(f'{path_nc}/{name}_{iv}.nc')

# %% XXX OS INDICES SAO GERADOS APARENTEMENTE OK
lat = field.geometry.centroid.y.values[0]
lon = field.geometry.centroid.x.values[0]

for iv in indices:
    dsi[iv].sel(latitude=lat, longitude=lon, 
                 method='nearest').plot();plt.grid();plt.show();plt.close()



#%%
# #%% IVS Climatology
# Cdsi = dsi.groupby('time.month').mean(skipna=True)

# Cdsi.load()

# #%%
# Cdsi.to_netcdf(f'{path_nc}/{name}_IVs_cli.nc')

# #%%
# for iv in indices:
#     Cdsi[iv].sel(latitude=lat, longitude=lon, method='nearest').plot()
#     plt.grid();plt.plot(); plt.show()

# #%%
# for iv in indices:
#     dsi[iv].sel(latitude=lat, longitude=lon, method='nearest').plot()
#     plt.grid();plt.plot(); plt.show()


# # %%
# if zscores:
#     print('calculating zscores')
#     dsi_mean = dsi.groupby('time.month').mean(dim='time')
#     dsi_std = dsi.groupby('time.month').std(dim='time')

#     dsi_anom = dsi.groupby('time.month') - dsi_mean
#     dsi_z = dsi_anom.groupby('time.month') / dsi_std

#     dsi_z.to_netcdf(f'{path_nc}/{name}_Z-{sm}.nc')
#     print('zscores saved')

# print(f'{time.time()-start} seconds')


# # XXX XXX XXX XXX ... ,,, XXX XXX
# # %% THE EXTRACTION MISSION
# def mask_farm(field,dst):
    
#     mask = xr_rasterize(field,dst) 
#     # #mask data
#     dst = dst.where(mask)
#     # #convert to float 32 to conserve memory
#     #ds = ds.astype(np.int16) * 1000
#     dst = dst.astype(np.float32)
#     print('Farm masked outside of boundaries!')
#     return dst

# if savecsv:
#     print('Masking farm')
#     dam = mask_farm(field,dsi)

# # %% Create zones for paddocks 
# def farm_zones(field,data,column,ochunk=64):
        
#     fz = xr_rasterize(field,data,attribute_col=column,verbose=True)
#     fz = fz.chunk(ochunk)
#     fz.astype('int16')
#     return fz

# start = time.time()

# def extract_fz_timeseries(dst, data, field, column, path_csv, name, suffix, band, ochunk=64, verbose=False):
#     '''
#         Extract time-series for farm zones for one variable
#         band is, for example in LST 89, 'lwir11'
#     '''
#     fz = farm_zones(field,data,column,ochunk)
#     tozip = []
#     dstrc = dst.chunk(ochunk)

#     #
#     tempo = pd.to_datetime(dam.time.values)
#     anos = np.unique([str(x) for x in tempo.year])

#     for ano in anos[:-1]:

#     # get stats for the first dataframe
#         print(f'working on {ano}')
#         data_ = dstrc[band].sel(time=dstrc[band].time.values[0]).squeeze()
#         data_ = data_.sel(time=slice(ano+'-01-01',str(int(ano)+1)+'12-31'))
#         print(f'computing stats for the first date of year {ano}')
#         outst = zonal_stats(zones=fz, values=data_).compute()
#         outst['date'] = str(dstrc[band].time.values[0])
#         data_.close()

#         # and through the loop
#         for t in dstrc.time.values[1:]:
#             data_ = dstrc[band].sel(time=t).squeeze()
#             if verbose: print(f'computing stats for {t}')
            
#             outst1 = zonal_stats(zones=fz, values=data_).compute()

#             outst1['date'] = str(t)
#             outst = pd.concat([outst,outst1])
#             data_.close()
#             del outst1
#         namestr = f'{path_csv}/{name}_{band}_{ano}_{suffix}.csv'
#         #tozip.append(namestr)
#         outst.to_csv(namestr)
#         print(f'{namestr} SAVED \n \n')
#         del outst, dstrc, data_

#     # if zip:
#     #     with zipfile.ZipFile(f'{path_csv}/{name}_{band}.zip', 'w') as zipMe:        
#     #         for file in tozip:
#     #             zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

# # %%
# #ds = da.to_dataset()
# extract_fz_timeseries(dsi, 
#                       ds, 
#                       field, 
#                       'mod_fiscal', 
#                       path_csv, 
#                       name, 
#                       '_-_', 
#                       'NDVI', 
#                       ochunk=64, verbose=False)

# %%
