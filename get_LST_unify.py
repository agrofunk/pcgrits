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

#%% DEFINE AREA OF INTEREST

# Name for reference
name = 'Uniguiri_farm_unify'

# AOI file and layer (for GPKG)
path_vector = '/home/jovyan/PlanetaryComputerExamples/vetorial/FAZENDAS/'
file = path_vector + 'fazenda_uniguiri.gpkg'
layer = 'piquetes_tid'

# %%# Get FIELD
field = gpd.read_file(file, layer=layer)
#field = field[field['Re'] == 80000]

bbox, lat_range, lon_range = get_lims(field)
print(field.head())
field.plot(column='TID')

#%% Define DATE TIME
# Landsat 4,5,7 have 'lwir' and 8 and 9 have 'lwir11'
# datetime89 = '2013-05-01/'+str(date.today())
# datetime457 = '1985-01-01/2013-05-01'
datetime = '2012-05-01/2015-11-22'



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

#%% You can exclude some Landsats from the list
items = query_Landsat_items(datetime=datetime,
                         bbox=bbox,
                         max_cloud=max_cloud,
                         landsats = [
                             "landsat-4", "landsat-5", "landsat-7",
                                    "landsat-8", "landsat-9"
                                     ]
                        )
print(len(items))

#%%

#%%
assets = ['lwir11','lwir']
data= (
        stackstac.stack(
        items,
        assets=assets,
        bounds_latlon=bbox,
        epsg=4326, # o xarray de imagens será retornado no EPSG:4326
        #resolution = 0.000281612818071153, # cuidado se for mexer na resolucao, tente algo como 0.001 para começar, pois é graus (não metros)
    ))
data = data.rename({'x': 'longitude','y': 'latitude'})
dst = data.to_dataset(dim='band')
del dst.attrs['spec']

# %%
def get_lst(lwirband, items, dst, w=5):
    '''
        Convert lwir to Celcius and prepare dataset for further processing
        lwirband (str): 'lwir' for 457 and lwirband for 89
        da (DataArray loaded from items__) 
        w (int): rolling mean window size, default is 5
    '''
    # get lwir11 band info
    band_info = items[0].assets[lwirband].extra_fields["raster:bands"][0]
    print(band_info)

    dst[lwirband] = dst[lwirband].astype(float)
    dst[lwirband] *= band_info['scale']
    dst[lwirband] += band_info['offset']
    dst[lwirband] -= 273.15

    # variables to drop so I can save the .nc later on
    drops = ['landsat:correction','landsat:wrs_path','landsat:wrs_row',
            'landsat:collection_number','landsat:wrs_type','instruments',
            'raster:bands','instruments']
    dst = dst.drop_vars(drops)
    # interpolate NaNs (rechunk it first)
    dst = dst.chunk(dict(time=-1))
    dst[lwirband] = xr.where(dst[lwirband] < 1, np.nan, dst[lwirband]) # 
    dst[lwirband] = xr.where(dst[lwirband] > 65, np.nan, dst[lwirband])
    dst[lwirband] = dst[lwirband].interpolate_na(dim='time',method='linear')

    # I`m overwriting the raw data
    dst[lwirband] = dst[lwirband].rolling(time=w, center=True).mean(savgol_filter, window = w, polyorder=2)
    del band_info
    return dst

# %% finally, get the Land Surface Temperature in Celcius
lst89 = get_lst('lwir11',items89, dst89,5)
lst457 = get_lst('lwir',items457, dst457,5)

#%%
def lst2nc(dst,path_nc,name,Landsats):
    '''
        save LST data to netcdf
    '''
    Landsats = str(Landsats)

    print('Reprojecting and saving ... \n')
    dst = dst.rio.write_crs('4326')
    dst = dst.rio.reproject('EPSG:4326')
    dst = dst.rename({'x': 'longitude','y': 'latitude'})
    print('... saving ...')
    
    try:
        dst.to_netcdf(f'{path_nc}lst_{name}_{Landsats}.nc', mode='w')
    except:
        print('trying to remove some weird shit')
        dst = dst.drop_vars(['raster:bands','instruments'])
        dst.to_netcdf(f'{path_nc}lst_{name}_{Landsats}.nc', mode='w')

    print(f'lst_{name}_{Landsats}.nc saved!')

if savenc:
    lst2nc(dst89,path_nc,name,89)
    lst2nc(dst457,path_nc,name,457)


#%% EXTRACTING data

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
    lst89m = mask_farm(field,lst89)
    lst457m = mask_farm(field,lst457)



print(f'Tempo total de processamento salvando os netcdfs no final: {time.time() - start} segundos')




if savecsv:
    start = time.time()

#%% Create zones for paddocks 
def farm_zones(field,data,column,ochunk=64):
        
    fz = xr_rasterize(field,data,attribute_col=column,verbose=True)
    fz = fz.chunk(ochunk)
    fz.astype('int16')
    return fz

#%% and finally, the extraction
def extract_fz_timeseries(dst, data, field, column, path_csv, name, suffix, band, ochunk=64, zip=False, verbose=False):
    '''
        Extract time-series for farm zones for one variable
        band is, for example in LST 89, 'lwir11'
    '''
    fz = farm_zones(field,data,column,ochunk)
    tozip = []
    dstrc = dst.chunk(ochunk)
    # get stats for the first dataframe
    data_ = dstrc[band].sel(time=dstrc[band].time.values[0]).squeeze()
    print('computing stats for the first date')
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
    namestr = f'{path_csv}/{name}_{band}_{suffix}.csv'
    tozip.append(namestr)
    outst.to_csv(namestr)
    print(f'{namestr} SAVED \n \n')
    del outst, dstrc, data_

    if zip:
        with zipfile.ZipFile(f'{path_csv}/{name}_{band}.zip', 'w') as zipMe:        
            for file in tozip:
                zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

#%%
extract_fz_timeseries(lst89, 
                      data89, 
                      field, 
                      'TID', 
                      path_csv, 
                      name, 
                      '89', 
                      'lwir11', 
                      ochunk=64, zip=False, verbose=False)
#%%
extract_fz_timeseries(lst457, 
                      data457, 
                      field, 
                      'TID', 
                      path_csv, 
                      name, 
                      '457', 
                      'lwir', 
                      ochunk=64, zip=False, verbose=True)
#%%

print(f'Tempo total de processamento das extractions: {time.time() - start} segundos')
