
#
print('CHECK THIS OUT AND MAKE IT WORK FOR ME \n ')
print('https://github.com/flowers-huang/cs325b-wildfire/tree/665fdff830de37e84f717af8797996d04215b492') 



#%%
# LOAD PACKAGES
import numpy as np
import xarray as xr
import pylab as plt

import rasterio.features
import stackstac
import pystac_client
import planetary_computer


from dask_gateway import GatewayCluster
import rasterio
import geopandas as gpd
import rioxarray

from pyproj import Proj, transform


import sys
sys.path.append('/home/jovyan/PlanetaryComputerExamples/CODE/grits')
from grits import humanbytes, get_field, get_lims, get_mms, xr_rasterize, calculate_indices

# from DEA
sys.path.append('/home/jovyan/PlanetaryComputerExamples/CODE/grits/deafrica_tools/')
from plotting import display_map, rgb, map_shapefile


#%%
#_____ perhaps _______ CREATE DASK CLUSTER 
cluster = GatewayCluster()  

client = cluster.get_client()

cluster.adapt(minimum=4, maximum=24)
print(cluster.dashboard_link)

#%%
# GET FARM and BBOX
# path = '/home/jovyan/PlanetaryComputerExamples/vetorial/FAZENDAS/'
# field = gpd.read_file( path + 'fazenda_uniguiri.gpkg')
# field.plot()
# fieldgeo = field.geometry.to_dict()[0]
# bbox = rasterio.features.bounds(fieldgeo)
# print(bbox)


# para um grupo dentro de uma farm
path = '/home/jovyan/PlanetaryComputerExamples/vetorial/FAZENDAS/'

layer = 'piquetes_tid'
column = 'Re'
val = 80000

field = gpd.read_file( path + 'fazenda_uniguiri.gpkg' , layer=layer)
field = field[field[column] == val]

# FURTHER REDUCING AREA FOR TESTS
field = field[field['R'] == 'R8_']


bbox, lat_range, lon_range = get_lims(field)

print(field.head())
field.plot(column='TID')



#%%
# RUN THE SEARCH for Landsat 8 and 9 
datetime="2013-03-10/2021-10-30" # tentar pegar 2022 inteiro, tem erro de imagem faltando e travando os calculos
print(datetime)

stac = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# only Landsat 8 and 9
query_params = {
    "eo:cloud_cover": {"lt": 20},
    "platform": {"in": ["landsat-8", "landsat-9"]},
            }

# Aqui definimos o intervalo de datas onde faremos a procura
search = stac.search(
    bbox=bbox,
    datetime=datetime, # Landsat 8 é de abril 2013 pra frente
    collections='landsat-c2-l2',
    query=query_params, # e aqui alteramos o limite máximo de nuvens 
)

# o resultado da procura está guardado aqui no 'items'
items = search.item_collection()
print(f'found {len(items)} items \n first: {items[-1]} \n last: {items[0]}')
print(items[0].assets.keys())
#%%

#%%
# CREATE STACK
assets = ['blue','green','red','nir08','lwir11','swir16','qa']

data = (
    stackstac.stack(
        items,
        assets=assets,
        bounds_latlon=bbox,
        epsg=4326, # o xarray de imagens será retornado no EPSG:4326
      # resolution = 0.0003, # cuidado se for mexer na resolucao, tente algo como 0.001 para começar, pois é graus (não metros)
    )
 
)
data = data.rename({'x': 'longitude','y': 'latitude'})
data
#data

#%%
# CONVERT TO DATASET
ds_ = data.to_dataset(dim='band') #TO KEEP
ds = data.to_dataset(dim='band') 
#ds
# %%
# GET LST
if 'lwir11' in assets:
    print('Land Surface Temperature requested. \n -> Converting to Celcius')

    # get lwir11 band info
    band_info = items[0].assets["lwir11"].extra_fields["raster:bands"][0]
    print(band_info)

    ds['lwir11'] = ds['lwir11'].astype(float)
    ds['lwir11'] *=band_info['scale']
    ds['lwir11'] +=band_info['offset']
    ds['lwir11'] -= 273.15
    lst = ds['lwir11'].copy()
    ds = ds.drop(['lwir11'])


#%%
# RENAME BANDS
ds['nir'] = ds['nir08']
ds = ds.drop(['nir08'])

#%%
# CALCULATE INDICES

# calcula os indices
indices = ['NDVI','BSI','LAI','EVI']
ds = calculate_indices(ds, 
                       index= indices, 
                       satellite_mission='ls', 
                       drop=True);

humanbytes(ds.nbytes)

# %%
# MASK TO FARM
masked = True

if masked:
    mask = xr_rasterize(field,ds,
                        # x_dim='x',
                        # y_dim='y',
                       #export_tiff='masked2.tiff',
                       ) 

    # #mask data
    ds = ds.where(mask)

    # #convert to float 32 to conserve memory
    #ds = ds.astype(np.int16) * 1000
    ds = ds.astype(np.float32)
    humanbytes(ds.nbytes)
ds

# %% levou 5 minutos
%%time
# NDVI ok
# EVI com maximo muito alto
# LAI com maximo muito alto
# BSI tem valor negativo, mas pode estar ok

# para os IVs
qmin_qmax=[0.02, 0.98]
mms = get_mms(ds, indices, qmin_qmax)
# mms = {'LAI': array([ 1.43231112, 13.73826774]),
#         'NDVI': array([0.11159951, 0.45451676]),
#         'EVI': array([0.42849948, 3.82981419]),
#         'BSI': array([-0.1559449 ,  0.11614301])}


# %%
# VISUALIZANDO!!!
i1,i2,i3 = 'LAI','BSI','NDVI'

timeslice = False

if timeslice:
    ds_ = ds.sel(time=slice("2023-06-15", "2023-07-31")).copy(deep=True)
    step = 1
    
if not timeslice:
    #zscores_ = zscores.copy()
    step = 1


for t in range(0, len(ds.time),step):

    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6))  = plt.subplots(2, 3,figsize=(20,7))

    # RGB
    rgb(ds_,index=[t],percentile_stretch=(0.02,0.98), ax=ax1, aspect=1)
    ax1.set_title('RGB')
    ax1.grid()

    # i1 plot
    ds[i1].isel(time=[t]).plot(cmap = 'RdYlGn', vmin = mms[i1][0], vmax = mms[i1][1], ax=ax2)
    ax2.grid()

    # i1 histogram
    flat = ds[i1].isel(time=[t]).values.flatten()
    flat = np.where((flat > mms[i1][1]) | (flat < mms[i1][0]), np.nan, flat)
    ax5.hist(flat,50)
    ax5.grid()
    ax5.set_title(f'{i1}, sum = {np.nansum(flat):.0f} quantiles .05, .5, .95 = {np.round(np.nanquantile(flat,[.05,.5,.95]),2)}')

    # i2 plot
    # por ser 'BSI', uso a escala invertida
    ds[i2].isel(time=[t]).plot(cmap = 'RdBu_r', vmin = mms[i2][0], vmax = mms[i2][1], ax=ax3)
    ax3.set_title(None);
    ax3.grid()

    # i2 histogram
    flat = ds[i2].isel(time=[t]).values.flatten()
    flat = np.where((flat > mms[i2][1]) | (flat < mms[i2][0]), np.nan, flat)
    ax6.hist(flat, bins=50)
    ax6.grid()
    ax6.set_title(f'{i2}, sum = {np.nansum(flat):.0f} quantiles .05, .5, .95 = {np.round(np.nanquantile(flat,[.05,.5,.95]),2)}')

    # i3 plot
    ds[i3].isel(time=[t]).plot(cmap = 'RdYlGn', vmin = mms[i3][0], vmax = mms[i3][1], ax=ax4)
    ax4.set_title(i3);
    ax4.grid()

    fig.tight_layout();   
    plt.show();plt.close()
# %%
