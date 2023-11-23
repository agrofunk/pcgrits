import numpy as np
import warnings
from geopandas import read_file
import pystac_client
import planetary_computer

"""
    Estou juntando aqui em 'grits' as funções. 
    Há temas diferentes, em algum momento elas terão de ser separadas
    de forma que faça sentido. 
    
    denismeia@icloud.com 


    ## from me
    - get_field
    - get_lims
    - get_mms
    - query_l2a_items
    - query_modis_items
    - zscore_dataset
    
    
    - humanbytes
    
    
    ## from DEA plotting.py
    - display_map
    - rgb
    
    ## from DEA spatial.py
    - xr_rasterize
    - calculate_indices
    - dualpol_indices # ainda não testada

"""


def zscore_dataset(ds):
    '''
        calculate zscores based on monthly mean and std values
        It is recommended the data to be already filtered (maxs, mins, nodata), 
            but it is ok if the dataset is not yet masked (by geometry)
    '''
    def calculate_zscore(da):
        '''
            zscore calculation
        '''
        mean = da.mean(skipna=True)
        std = da.std(skipna=True)
        zscore = (da - mean) / std
        return zscore

    # firstly turn the dataset into monthly data
    grouped_ds = ds.groupby('time.month').mean()

    # then, calculate zscore using the predefined function
    zscore = grouped_ds.apply(calculate_zscore)

    return zscore


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


def get_field(file,  column, ID, layer=None, multi_IDs=False, IDs=None):
    '''
        file.:str: vector file containing group of farms (CARs, SIGEFs) or in the case of a farm, the farm itself
        column.:str: 'column' to find farms or in the case of a farm file, the column to identify fields (padocks, plots, talhoes, piquetes)
        ID.:str,int: farm ID in the 'column'
        multi_IDs.:bool: default 'False'. If true, one must provide the list of IDs to aggregate as a single field. 

        return 'field': a farm, a field or a group of fields combined as one.
    '''

#     if column == None:
#         gdf_ = read_file( file, layer=layer )

    if file[-4:] == 'gpkg':
        layer = layer
        gdf_ = read_file(file, layer=layer)
    else:
        gdf_ = read_file(file)

    if column == None:
        field = gdf_

    if multi_IDs == False:

        field = gdf_[gdf_[column] == ID]
    else:
        field = gdf_[gdf_[column].isin(IDs)]

    return field


def get_lims(gdf):
    '''
        get bbox, lat_range and lon_range from a geodataframe, let's say, a farm or field.
        It returns the info as three tuples.

        gdf.:GeoDataframe:

        return bbox, lat_range and lon_range
    '''

    limites = gdf
    bbox = (limites.bounds.minx.min(),
            limites.bounds.miny.min(),
            limites.bounds.maxx.max(),
            limites.bounds.maxy.max()
            )

    lat_range = (bbox[1], bbox[3])
    lon_range = (bbox[0], bbox[2])
    print('got bbox, lat_range, lon_range')
    return bbox, lat_range, lon_range


def get_mms(ds, indices, qmin_qmax=[.01, .99]):
    '''
        Return a dictionary of minimuns and maximuns for each variable in a xarray dataset
        ds.:xarray dataset: the xarray dataset
        qmin_qmax.:2 float list: the minimum and maximum quantile, default is [.01,.99] 1% and 99%

    '''

    import numpy as np

    mms = {}
    keys = indices
    for i in keys:
        line = np.nanquantile(ds[i].values, qmin_qmax)
        mms[i] = line
    print(mms)
    return mms


def query_l2a_items(bbox,
                    datetime,
                    max_cloud_cover):
    '''
        Query Sentinel 2 L2A items for a given bounding box withing a 
        datetime range 
        bbox.:tuple with coordinates of the 2 corners of a bounding box: it is retrieved by the 
                get_lims function
        max_cloud_cover.:int: percentage of max cloud allowed.

    '''

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    query_params = {"eo:cloud_cover": {"lt": max_cloud_cover}}

    search = catalog.search(bbox=bbox,
                            collections=["sentinel-2-l2a"],
                            datetime=datetime,
                            query=query_params)

    items = search.item_collection()
    print(f' found {len(items)} items')

    return items


def query_modis_items(bbox,
                      datetime,
                      collection):
    '''
        Query MODIS items for a given bounding box withing a 
        datetime range 
        bbox.:tuple with coordinates of the 2 corners of a bounding box: it is retrieved by the 
                get_lims function
        collection.:str: collection.
        ... product? band?
    '''

    # import pystac_client
    # import planetary_computer

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # query_params = {"eo:cloud_cover": {"lt": max_cloud_cover}}

    search = catalog.search(bbox=bbox,
                            collections=collections,
                            datetime=datetime
                            )

    items = search.item_collection()
    print(f' found {len(items)} items')

    return items


def query_L457_items(datetime,
                     bbox, 
                     max_cloud = 30, 
                     landsats = ["landsat-4", "landsat-5", "landsat-7"],
                     tiers = ['T1']
):
    '''
        query Landsat 5 and 7 (and maybe 4)
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


def query_L89_items(datetime,
                     bbox, 
                     max_cloud = 30, 
                     landsats = ["landsat-8", "landsat-9"],
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




#==========================
# from DEA plotting.py 
#==========================

def display_map(x, y, crs="EPSG:4326", margin=-0.5, zoom_bias=0):
    """
    Given a set of x and y coordinates, this function generates an
    interactive map with a bounded rectangle overlayed on Google Maps
    imagery.

    Last modified: September 2019

    Modified from function written by Otto Wagner available here:
    https://github.com/ceos-seo/data_cube_utilities/tree/master/data_cube_utilities

    Parameters
    ----------
    x : (float, float)
        A tuple of x coordinates in (min, max) format.
    y : (float, float)
        A tuple of y coordinates in (min, max) format.
    crs : string, optional
        A string giving the EPSG CRS code of the supplied coordinates.
        The default is 'EPSG:4326'.
    margin : float
        A numeric value giving the number of degrees lat-long to pad
        the edges of the rectangular overlay polygon. A larger value
        results more space between the edge of the plot and the sides
        of the polygon. Defaults to -0.5.
    zoom_bias : float or int
        A numeric value allowing you to increase or decrease the zoom
        level by one step. Defaults to 0; set to greater than 0 to zoom
        in, and less than 0 to zoom out.
    Returns
    -------
    folium.Map : A map centered on the supplied coordinate bounds. A
    rectangle is drawn on this map detailing the perimeter of the x, y
    bounds.  A zoom level is calculated such that the resulting
    viewport is the closest it can possibly get to the centered
    bounding rectangle without clipping it.
    """

    from pyproj import Transformer
    import folium
    import numpy as np

    # Convert each corner coordinates to lat-lon
    all_x = (x[0], x[1], x[0], x[1])
    all_y = (y[0], y[0], y[1], y[1])
    transformer = Transformer.from_crs(crs, "EPSG:4326")
    all_longitude, all_latitude = transformer.transform(all_x, all_y)

    # Calculate zoom level based on coordinates
    lat_zoom_level = (
        _degree_to_zoom_level(min(all_latitude), max(
            all_latitude), margin=margin)
        + zoom_bias
    )
    lon_zoom_level = (
        _degree_to_zoom_level(min(all_longitude), max(
            all_longitude), margin=margin)
        + zoom_bias
    )
    zoom_level = min(lat_zoom_level, lon_zoom_level)

    # Identify centre point for plotting
    center = [np.mean(all_latitude), np.mean(all_longitude)]

    # Create map
    interactive_map = folium.Map(
        location=center,
        zoom_start=zoom_level,
        tiles="http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google",
    )

    # Create bounding box coordinates to overlay on map
    line_segments = [
        (all_latitude[0], all_longitude[0]),
        (all_latitude[1], all_longitude[1]),
        (all_latitude[3], all_longitude[3]),
        (all_latitude[2], all_longitude[2]),
        (all_latitude[0], all_longitude[0]),
    ]

    # Add bounding box as an overlay
    interactive_map.add_child(
        folium.features.PolyLine(
            locations=line_segments, color="red", opacity=0.8)
    )

    # Add clickable lat-lon popup box
    interactive_map.add_child(folium.features.LatLngPopup())

    return interactive_map


def rgb(
    ds,
    bands=["red", "green", "blue"],
    index=None,
    index_dim="time",
    robust=True,
    percentile_stretch=None,
    col_wrap=4,
    size=6,
    aspect=None,
    savefig_path=None,
    savefig_kwargs={},
    **kwargs,
):
    """
    Takes an xarray dataset and plots RGB images using three imagery
    bands (e.g ['red', 'green', 'blue']). The `index`
    parameter allows easily selecting individual or multiple images for
    RGB plotting. Images can be saved to file by specifying an output
    path using `savefig_path`.
    This function was designed to work as an easier-to-use wrapper
    around xarray's `.plot.imshow()` functionality.

    Last modified: April 2021

    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array to plot as an RGB
        image. If the array has more than two dimensions (e.g. multiple
        observations along a 'time' dimension), either use `index` to
        select one (`index=0`) or multiple observations
        (`index=[0, 1]`), or create a custom faceted plot using e.g.
        `col="time"`.
    bands : list of strings, optional
        A list of three strings giving the band names to plot. Defaults
        to '['red', 'green', 'blue']'. If the dataset does not contain
        bands named `'red', 'green', 'blue'`, then `bands` must be
        specified.
    index : integer or list of integers, optional
        `index` can be used to select one (`index=0`) or multiple
        observations (`index=[0, 1]`) from the input dataset for
        plotting. If multiple images are requested these will be plotted
        as a faceted plot.
    index_dim : string, optional
        The dimension along which observations should be plotted if
        multiple observations are requested using `index`. Defaults to
        `time`.
    robust : bool, optional
        Produces an enhanced image where the colormap range is computed
        with 2nd and 98th percentiles instead of the extreme values.
        Defaults to True.
    percentile_stretch : tuple of floats
        An tuple of two floats (between 0.00 and 1.00) that can be used
        to clip the colormap range to manually specified percentiles to
        get more control over the brightness and contrast of the image.
        The default is None; '(0.02, 0.98)' is equivelent to
        `robust=True`. If this parameter is used, `robust` will have no
        effect.
    col_wrap : integer, optional
        The number of columns allowed in faceted plots. Defaults to 4.
    size : integer, optional
        The height (in inches) of each plot. Defaults to 6.
    aspect : integer, optional
        Aspect ratio of each facet in the plot, so that aspect * size
        gives width of each facet in inches. Defaults to None, which
        will calculate the aspect based on the x and y dimensions of
        the input data.
    savefig_path : string, optional
        Path to export image file for the RGB plot. Defaults to None,
        which does not export an image file.
    savefig_kwargs : dict, optional
        A dict of keyword arguments to pass to
        `matplotlib.pyplot.savefig` when exporting an image file. For
        all available options, see:
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    **kwargs : optional
        Additional keyword arguments to pass to `xarray.plot.imshow()`.
        For example, the function can be used to plot into an existing
        matplotlib axes object by passing an `ax` keyword argument.
        For more options, see:
        http://xarray.pydata.org/en/stable/generated/xarray.plot.imshow.html
    Returns
    -------
    An RGB plot of one or multiple observations, and optionally an image
    file written to file.
    """

    # If bands are not in the dataset
    ds_vars = list(ds.data_vars)
    if set(bands).issubset(ds_vars) == False:
        raise ValueError(
            "rgb() bands do not match band names in dataset. "
            "Note the default rgb() bands are ['red', 'green', 'blue']."
        )

    # If ax is supplied via kwargs, ignore aspect and size
    if "ax" in kwargs:

        # Create empty aspect size kwarg that will be passed to imshow
        aspect_size_kwarg = {}
    else:
        # Compute image aspect
        if not aspect:
            aspect = image_aspect(ds)

        # Populate aspect size kwarg with aspect and size data
        aspect_size_kwarg = {"aspect": aspect, "size": size}

    # If no value is supplied for `index` (the default), plot using default
    # values and arguments passed via `**kwargs`
    if index is None:

        # Select bands and convert to DataArray
        da = ds[bands].to_array()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax
        if percentile_stretch:
            vmin, vmax = da.compute().quantile(percentile_stretch).values
            kwargs.update({"vmin": vmin, "vmax": vmax})

        # If there are more than three dimensions and the index dimension == 1,
        # squeeze this dimension out to remove it
        if (len(ds.dims) > 2) and ("col" not in kwargs) and (len(da[index_dim]) == 1):

            da = da.squeeze(dim=index_dim)

        # If there are more than three dimensions and the index dimension
        # is longer than 1, raise exception to tell user to use 'col'/`index`
        elif (len(ds.dims) > 2) and ("col" not in kwargs) and (len(da[index_dim]) > 1):

            raise Exception(
                f"The input dataset `ds` has more than two dimensions: "
                "{list(ds.dims.keys())}. Please select a single observation "
                "using e.g. `index=0`, or enable faceted plotting by adding "
                'the arguments e.g. `col="time", col_wrap=4` to the function '
                "call"
            )
        da = da.compute()
        img = da.plot.imshow(
            robust=robust, col_wrap=col_wrap, **aspect_size_kwarg, **kwargs
        )

    # If values provided for `index`, extract corresponding observations and
    # plot as either single image or facet plot
    else:

        # If a float is supplied instead of an integer index, raise exception
        if isinstance(index, float):
            raise Exception(
                f"Please supply `index` as either an integer or a list of " "integers"
            )

        # If col argument is supplied as well as `index`, raise exception
        if "col" in kwargs:
            raise Exception(
                f"Cannot supply both `index` and `col`; please remove one and "
                "try again"
            )

        # Convert index to generic type list so that number of indices supplied
        # can be computed
        index = index if isinstance(index, list) else [index]

        # Select bands and observations and convert to DataArray
        da = ds[bands].isel(**{index_dim: index}).to_array().compute()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax
        if percentile_stretch:
            vmin, vmax = da.compute().quantile(percentile_stretch).values
            kwargs.update({"vmin": vmin, "vmax": vmax})

        # If multiple index values are supplied, plot as a faceted plot
        if len(index) > 1:

            img = da.plot.imshow(
                robust=robust,
                col=index_dim,
                col_wrap=col_wrap,
                **aspect_size_kwarg,
                **kwargs,
            )

        # If only one index is supplied, squeeze out index_dim and plot as a
        # single panel
        else:

            img = da.squeeze(dim=index_dim).plot.imshow(
                robust=robust, **aspect_size_kwarg, **kwargs
            )

    # If an export path is provided, save image to file. Individual and
    # faceted plots have a different API (figure vs fig) so we get around this
    # using a try statement:
    if savefig_path:

        print(f"Exporting image to {savefig_path}")

        try:
            img.fig.savefig(savefig_path, **savefig_kwargs)
        except:
            img.figure.savefig(savefig_path, **savefig_kwargs)


#=========================
# from DEA bandindices.py
#=========================

def calculate_indices(
    ds,
    index=None,
    collection=None,
    satellite_mission=None,
    custom_varname=None,
    normalise=True,
    drop=False,
    deep_copy=True,
):
    """
    Takes an xarray dataset containing spectral bands, calculates one of
    a set of remote sensing indices, and adds the resulting array as a
    new variable in the original dataset.

    Last modified: July 2022

    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array with containing the
        spectral bands required to calculate the index. These bands are
        used as inputs to calculate the selected water index.

    index : str or list of strs
        A string giving the name of the index to calculate or a list of
        strings giving the names of the indices to calculate:

        * ``'ASI'``  (Artificial Surface Index, Yongquan Zhao & Zhe Zhu 2022)
        * ``'AWEI_ns'`` (Automated Water Extraction Index, no shadows, Feyisa 2014)
        * ``'AWEI_sh'`` (Automated Water Extraction Index, shadows, Feyisa 2014)
        * ``'BAEI'`` (Built-Up Area Extraction Index, Bouzekri et al. 2015)
        * ``'BAI'`` (Burn Area Index, Martin 1998)
        * ``'BSI'`` (Bare Soil Index, Rikimaru et al. 2002)
        * ``'BUI'`` (Built-Up Index, He et al. 2010)
        * ``'CMR'`` (Clay Minerals Ratio, Drury 1987)
        * ``'ENDISI'`` (Enhanced Normalised Difference for Impervious Surfaces Index, Chen et al. 2019)
        * ``'EVI'`` (Enhanced Vegetation Index, Huete 2002)
        * ``'FMR'`` (Ferrous Minerals Ratio, Segal 1982)
        * ``'IOR'`` (Iron Oxide Ratio, Segal 1982)
        * ``'LAI'`` (Leaf Area Index, Boegh 2002)
        * ``'MBI'`` (Modified Bare Soil Index, Nguyen et al. 2021)
        * ``'MNDWI'`` (Modified Normalised Difference Water Index, Xu 1996)
        * ``'MSAVI'`` (Modified Soil Adjusted Vegetation Index, Qi et al. 1994)
        * ``'NBI'`` (New Built-Up Index, Jieli et al. 2010)
        * ``'NBR'`` (Normalised Burn Ratio, Lopez Garcia 1991)
        * ``'NDBI'`` (Normalised Difference Built-Up Index, Zha 2003)
        * ``'NDCI'`` (Normalised Difference Chlorophyll Index, Mishra & Mishra, 2012)
        * ``'NDMI'`` (Normalised Difference Moisture Index, Gao 1996)
        * ``'NDSI'`` (Normalised Difference Snow Index, Hall 1995)
        * ``'NDTI'`` (Normalised Difference Turbidity Index, Lacaux et al. 2007)
        * ``'NDVI'`` (Normalised Difference Vegetation Index, Rouse 1973)
        * ``'NDWI'`` (Normalised Difference Water Index, McFeeters 1996)
        * ``'SAVI'`` (Soil Adjusted Vegetation Index, Huete 1988)
        * ``'TCB'`` (Tasseled Cap Brightness, Crist 1985)
        * ``'TCG'`` (Tasseled Cap Greeness, Crist 1985)
        * ``'TCW'`` (Tasseled Cap Wetness, Crist 1985)
        * ``'WI'`` (Water Index, Fisher 2016)

    collection : str
        Deprecated in version 0.1.7. Use `satellite_mission` instead. 

        Valid options are: 
        * ``'c2'`` (for USGS Landsat Collection 2)
            If 'c2', then `satellite_mission='ls'`.
        * ``'s2'`` (for Sentinel-2)
            If 's2', then `satellite_mission='s2'`.

    satellite_mission : str
        An string that tells the function which satellite mission's data is
        being used to calculate the index. This is necessary because
        different satellite missions use different names for bands covering
        a similar spectra.

        Valid options are:

         * ``'ls'`` (for USGS Landsat)
         * ``'s2'`` (for Copernicus Sentinel-2)

    custom_varname : str, optional
        By default, the original dataset will be returned with
        a new index variable named after `index` (e.g. 'NDVI'). To
        specify a custom name instead, you can supply e.g.
        `custom_varname='custom_name'`. Defaults to None, which uses
        `index` to name the variable.

    normalise : bool, optional
        Some coefficient-based indices (e.g. ``'WI'``, ``'BAEI'``,
        ``'AWEI_ns'``, ``'AWEI_sh'``, ``'TCW'``, ``'TCG'``, ``'TCB'``,
        ``'EVI'``, ``'LAI'``, ``'SAVI'``, ``'MSAVI'``)
        produce different results if surface reflectance values are not
        scaled between 0.0 and 1.0 prior to calculating the index.
        Setting `normalise=True` first scales values to a 0.0-1.0 range
        by dividing by 10000.0. Defaults to True.

    drop : bool, optional
        Provides the option to drop the original input data, thus saving
        space. If `drop=True`, returns only the index and its values.

    deep_copy: bool, optional
        If `deep_copy=False`, calculate_indices will modify the original
        array, adding bands to the input dataset and not removing them.
        If the calculate_indices function is run more than once, variables
        may be dropped incorrectly producing unexpected behaviour. This is
        a bug and may be fixed in future releases. This is only a problem
        when `drop=True`.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a
        new varible containing the remote sensing index as a DataArray.
        If drop = True, the new variable/s as DataArrays in the
        original Dataset.
    """

    # Set ds equal to a copy of itself in order to prevent the function
    # from editing the input dataset. This is to prevent unexpected
    # behaviour though it uses twice as much memory.

    if deep_copy:
        ds = ds.copy(deep=True)

    # Capture input band names in order to drop these if drop=True
    if drop:
        bands_to_drop = list(ds.data_vars)
        print(f"Dropping bands {bands_to_drop}")

    # Dictionary containing remote sensing index band recipes
    index_dict = {
        # Normalised Difference Vegation Index, Rouse 1973
        "NDVI": lambda ds: (ds.nir - ds.red) / (ds.nir + ds.red),
        # Enhanced Vegetation Index, Huete 2002
        "EVI": lambda ds: (
            2.5 * ((ds.nir - ds.red) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))
        ),
        # Leaf Area Index, Boegh 2002
        "LAI": lambda ds: (
            3.618
            * ((2.5 * (ds.nir - ds.red)) / (ds.nir + (6 * ds.red) - (7.5 * ds.blue) + 1))
            - 0.118
        ),
        # Soil Adjusted Vegetation Index, Huete 1988
        "SAVI": lambda ds: ((1.5 * (ds.nir - ds.red)) / (ds.nir + ds.red + 0.5)),
        # Mod. Soil Adjusted Vegetation Index, Qi et al. 1994
        "MSAVI": lambda ds: (
            (2 * ds.nir + 1 - ((2 * ds.nir + 1) ** 2 - 8 * (ds.nir - ds.red)) ** 0.5)
            / 2
        ),
        # Normalised Difference Moisture Index, Gao 1996
        "NDMI": lambda ds: (ds.nir - ds.swir16) / (ds.nir + ds.swir16),
        # Normalised Burn Ratio, Lopez Garcia 1991
        "NBR": lambda ds: (ds.nir - ds.swir22) / (ds.nir + ds.swir22),
        # Burn Area Index, Martin 1998
        "BAI": lambda ds: (1.0 / ((0.10 - ds.red) ** 2 + (0.06 - ds.nir) ** 2)),
        # Normalised Difference Chlorophyll Index,
        # (Mishra & Mishra, 2012)
        "NDCI": lambda ds: (ds.rededge - ds.red) / (ds.rededge + ds.red),
        # Normalised Difference Snow Index, Hall 1995
        "NDSI": lambda ds: (ds.green - ds.swir16) / (ds.green + ds.swir16),
        # Normalised Difference Water Index, McFeeters 1996
        "NDWI": lambda ds: (ds.green - ds.nir) / (ds.green + ds.nir),
        # Modified Normalised Difference Water Index, Xu 2006
        "MNDWI": lambda ds: (ds.green - ds.swir16) / (ds.green + ds.swir16),
        # Normalised Difference Built-Up Index, Zha 2003
        "NDBI": lambda ds: (ds.swir16 - ds.nir) / (ds.swir_1 + ds.nir),
        # Built-Up Index, He et al. 2010
        "BUI": lambda ds: ((ds.swir16 - ds.nir) / (ds.swir_1 + ds.nir))
        - ((ds.nir - ds.red) / (ds.nir + ds.red)),
        # Built-up Area Extraction Index, Bouzekri et al. 2015
        "BAEI": lambda ds: (ds.red + 0.3) / (ds.green + ds.swir16),
        # New Built-up Index, Jieli et al. 2010
        "NBI": lambda ds: (ds.swir16 + ds.red) / ds.nir,
        # Bare Soil Index, Rikimaru et al. 2002
        "BSI": lambda ds: ((ds.swir16 + ds.red) - (ds.nir + ds.blue))
        / ((ds.swir16 + ds.red) + (ds.nir + ds.blue)),
        # Automated Water Extraction Index (no shadows), Feyisa 2014
        "AWEI_ns": lambda ds: (
            4 * (ds.green - ds.swir16) - (0.25 * ds.nir * +2.75 * ds.swir22)
        ),
        # Automated Water Extraction Index (shadows), Feyisa 2014
        "AWEI_sh": lambda ds: (
            ds.blue + 2.5 * ds.green - 1.5 * \
            (ds.nir + ds.swir16) - 0.25 * ds.swir22
        ),
        # Water Index, Fisher 2016
        "WI": lambda ds: (
            1.7204
            + 171 * ds.green
            + 3 * ds.red
            - 70 * ds.nir
            - 45 * ds.swir16
            - 71 * ds.swir22
        ),
        # Tasseled Cap Wetness, Crist 1985
        "TCW": lambda ds: (
            0.0315 * ds.blue
            + 0.2021 * ds.green
            + 0.3102 * ds.red
            + 0.1594 * ds.nir
            + -0.6806 * ds.swir16
            + -0.6109 * ds.swir22
        ),
        # Tasseled Cap Greeness, Crist 1985
        "TCG": lambda ds: (
            -0.1603 * ds.blue
            + -0.2819 * ds.green
            + -0.4934 * ds.red
            + 0.7940 * ds.nir
            + -0.0002 * ds.swir16
            + -0.1446 * ds.swir22
        ),
        # Tasseled Cap Brightness, Crist 1985
        "TCB": lambda ds: (
            0.2043 * ds.blue
            + 0.4158 * ds.green
            + 0.5524 * ds.red
            + 0.5741 * ds.nir
            + 0.3124 * ds.swir16
            + -0.2303 * ds.swir22
        ),
        # Clay Minerals Ratio, Drury 1987
        "CMR": lambda ds: (ds.swir16 / ds.swir22),
        # Ferrous Minerals Ratio, Segal 1982
        "FMR": lambda ds: (ds.swir16 / ds.nir),
        # Iron Oxide Ratio, Segal 1982
        "IOR": lambda ds: (ds.red / ds.blue),
        # Normalized Difference Turbidity Index, Lacaux, J.P. et al. 2007
        "NDTI": lambda ds: (ds.red - ds.green) / (ds.red + ds.green),
        # Modified Bare Soil Index, Nguyen et al. 2021
        "MBI": lambda ds: ((ds.swir16 - ds.swir22 - ds.nir) / (ds.swir16 + ds.swir22 + ds.nir)) + 0.5,
    }

    # Enhanced Normalised Difference Impervious Surfaces Index, Chen et al. 2019
    def mndwi(ds):
        return (ds.green - ds.swir16) / (ds.green + ds.swir16)

    def swir_diff(ds):
        return ds.swir16/ds.swir22

    def alpha(ds):
        return (2*(np.mean(ds.blue)))/(np.mean(swir_diff(ds)) + np.mean(mndwi(ds)**2))

    def ENDISI(ds):
        m = mndwi(ds)
        s = swir_diff(ds)
        a = alpha(ds)
        return (ds.blue - (a)*(s + m**2))/(ds.blue + (a)*(s + m**2))

    index_dict["ENDISI"] = ENDISI

    # Artificial Surface Index, Yongquan Zhao & Zhe Zhu 2022
    def af(ds):
        AF = (ds.nir - ds.blue) / (ds.nir + ds.blue)
        AF_norm = (AF - AF.min(dim=["y", "x"])) / \
            (AF.max(dim=["y", "x"]) - AF.min(dim=["y", "x"]))
        return AF_norm

    def ndvi(ds):
        return (ds.nir - ds.red) / (ds.nir + ds.red)

    def msavi(ds):
        return ((2 * ds.nir + 1 - ((2 * ds.nir + 1) ** 2 - 8 * (ds.nir - ds.red)) ** 0.5) / 2)

    def vsf(ds):
        NDVI = ndvi(ds)
        MSAVI = msavi(ds)
        VSF = 1 - NDVI * MSAVI
        VSF_norm = (VSF - VSF.min(dim=["y", "x"])) / \
            (VSF.max(dim=["y", "x"]) - VSF.min(dim=["y", "x"]))
        return VSF_norm

    def mbi(ds):
        return ((ds.swir16 - ds.swir22 - ds.nir) / (ds.swir16 + ds.swir22 + ds.nir)) + 0.5

    def embi(ds):
        MBI = mbi(ds)
        MNDWI = mndwi(ds)
        return (MBI - MNDWI - 0.5) / (MBI + MNDWI + 1.5)

    def ssf(ds):
        EMBI = embi(ds)
        SSF = 1 - EMBI
        SSF_norm = (SSF - SSF.min(dim=["y", "x"])) / \
            (SSF.max(dim=["y", "x"]) - SSF.min(dim=["y", "x"]))
        return SSF_norm
    # Overall modulation using the  Modulation Factor (MF).

    def mf(ds):
        MF = ((ds.blue + ds.green) - (ds.nir + ds.swir16)) / \
            ((ds.blue + ds.green) + (ds.nir + ds.swir16))
        MF_norm = (MF - MF.min(dim=["y", "x"])) / \
            (MF.max(dim=["y", "x"]) - MF.min(dim=["y", "x"]))
        return MF_norm

    def ASI(ds):
        AF = af(ds)
        VSF = vsf(ds)
        SSF = ssf(ds)
        MF = mf(ds)
        return AF * VSF * SSF * MF

    index_dict["ASI"] = ASI

    # If index supplied is not a list, convert to list. This allows us to
    # iterate through either multiple or single indices in the loop below
    indices = index if isinstance(index, list) else [index]

    # calculate for each index in the list of indices supplied (indexes)
    for index in indices:

        # Select an index function from the dictionary
        index_func = index_dict.get(str(index))

        # If no index is provided or if no function is returned due to an
        # invalid option being provided, raise an exception informing user to
        # choose from the list of valid options
        if index is None:

            raise ValueError(
                f"No remote sensing `index` was provided. Please "
                "refer to the function \ndocumentation for a full "
                "list of valid options for `index` (e.g. 'NDVI')"
            )

        elif (
            index
            in [
                "WI",
                "BAEI",
                "AWEI_ns",
                "AWEI_sh",
                "EVI",
                "LAI",
                "SAVI",
                "MSAVI",
            ]
            and not normalise
        ):

            warnings.warn(
                f"\nA coefficient-based index ('{index}') normally "
                "applied to surface reflectance values in the \n"
                "0.0-1.0 range was applied to values in the 0-10000 "
                "range. This can produce unexpected results; \nif "
                "required, resolve this by setting `normalise=True`"
            )

        elif index_func is None:

            raise ValueError(
                f"The selected index '{index}' is not one of the "
                "valid remote sensing index options. \nPlease "
                "refer to the function documentation for a full "
                "list of valid options for `index`"
            )

        # Deprecation warning if `collection` is specified instead of `satellite_mission`.
        if collection is not None:
            warnings.warn('`collection` was deprecated in version 0.1.7. Use `satelite_mission` instead.',
                          DeprecationWarning,
                          stacklevel=2)
            # Map the collection values to the valid satellite_mission values.
            if collection == "c2":
                satellite_mission = "ls"
            elif collection == "s2":
                satellite_mission = "s2"
            # Raise error if no valid collection name is provided:
            else:
                raise ValueError(
                    f"'{collection}' is not a valid option for "
                    "`collection`. Please specify either \n"
                    "'c2' or 's2'.")

        # Rename bands to a consistent format if depending on what satellite mission
        # is specified in `satellite_mission`. This allows the same index calculations
        # to be applied to all satellite missions. If no satellite mission was provided,
        # raise an exception.
        if satellite_mission is None:

            raise ValueError(
                "No `satellite_mission` was provided. Please specify "
                "either 'ls' or 's2' to ensure the \nfunction "
                "calculates indices using the correct spectral "
                "bands."
            )

        elif satellite_mission == "ls":
            sr_max = 1.0
            # Dictionary mapping full data names to simpler alias names
            # This only applies to properly-scaled "ls" data i.e. from
            # the Landsat geomedians. calculate_indices will not show
            # correct output for raw (unscaled) Landsat data (i.e. default
            # outputs from dc.load)
            bandnames_dict = {
                "SR_B1": "blue",
                "SR_B2": "green",
                "SR_B3": "red",
                "SR_B4": "nir",
                "SR_B5": "swir_1",
                "SR_B7": "swir_2",
            }

            # Rename bands in dataset to use simple names (e.g. 'red')
            bands_to_rename = {
                a: b for a, b in bandnames_dict.items() if a in ds.variables
            }

        elif satellite_mission == "s2":
            sr_max = 10000
            # Dictionary mapping full data names to simpler alias names
            bandnames_dict = {

                "B02": "blue",
                "B03": "green",
                "B04": "red",
                "B05": "rededge",
                "B06": "rededge",
                "B07": "rededge",
                "B08": "nir",
                "B08A": "rededge",
                "B11": "swir16",
                "B12": "swir22",
            }

            # Rename bands in dataset to use simple names (e.g. 'red')
            bands_to_rename = {
                a: b for a, b in bandnames_dict.items() if a in ds.variables
            }

        # Raise error if no valid satellite_mission name is provided:
        else:
            raise ValueError(
                f"'{satellite_mission}' is not a valid option for "
                "`satellite_mission`. Please specify either \n"
                "'ls' or 's2'"
            )

        # Apply index function
        try:
            # If normalised=True, divide data by 10,000 before applying func
            mult = sr_max if normalise else 1.0
            index_array = index_func(ds.rename(bands_to_rename) / mult)

        except AttributeError:
            raise ValueError(
                f"Please verify that all bands required to "
                f"compute {index} are present in `ds`."
            )

        # Add as a new variable in dataset
        output_band_name = custom_varname if custom_varname else index
        ds[output_band_name] = index_array

    # Once all indexes are calculated, drop input bands if drop=True
    if drop:
        ds = ds.drop(bands_to_drop)

    # Return input dataset with added water index variable
    return ds


def dualpol_indices(
    ds,
    co_pol='vv',
    cross_pol='vh',
    index=None,
    custom_varname=None,
    drop=False,
    deep_copy=True,
):
    """
    Takes an xarray dataset containing dual-polarization radar backscatter,
    calculates one or a set of indices, and adds the resulting array as a
    new variable in the original dataset.

    Last modified: July 2021

    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array containing the
        two polarization bands.

    co_pol: str
        Measurement name for the co-polarization band.
        Default is 'vv' for Sentinel-1.

    cross_pol: str
        Measurement name for the cross-polarization band.
        Default is 'vh' for Sentinel-1.

    index : str or list of strs
        A string giving the name of the index to calculate or a list of
        strings giving the names of the indices to calculate:

        * ``'RVI'`` (Radar Vegetation Index for dual-pol, Trudel et al. 2012; Nasirzadehdizaji et al., 2019; Gururaj et al., 2019)
        * ``'VDDPI'`` (Vertical dual depolarization index, Periasamy 2018)
        * ``'theta'`` (pseudo scattering-type, Bhogapurapu et al. 2021)
        * ``'entropy'`` (pseudo scattering entropy, Bhogapurapu et al. 2021)
        * ``'purity'`` (co-pol purity, Bhogapurapu et al. 2021)
        * ``'ratio'`` (cross-pol/co-pol ratio)

    custom_varname : str, optional
        By default, the original dataset will be returned with
        a new index variable named after `index` (e.g. 'RVI'). To
        specify a custom name instead, you can supply e.g.
        `custom_varname='custom_name'`. Defaults to None, which uses
        `index` to name the variable.

    drop : bool, optional
        Provides the option to drop the original input data, thus saving
        space. If `drop=True`, returns only the index and its values.

    deep_copy: bool, optional
        If `deep_copy=False`, calculate_indices will modify the original
        array, adding bands to the input dataset and not removing them.
        If the calculate_indices function is run more than once, variables
        may be dropped incorrectly producing unexpected behaviour. This is
        a bug and may be fixed in future releases. This is only a problem
        when `drop=True`.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a
        new varible containing the remote sensing index as a DataArray.
        If drop = True, the new variable/s as DataArrays in the
        original Dataset.
    """

    if not co_pol in list(ds.data_vars):
        raise ValueError(f"{co_pol} measurement is not in the dataset")
    if not cross_pol in list(ds.data_vars):
        raise ValueError(f"{cross_pol} measurement is not in the dataset")

    # Set ds equal to a copy of itself in order to prevent the function
    # from editing the input dataset. This is to prevent unexpected
    # behaviour though it uses twice as much memory.
    if deep_copy:
        ds = ds.copy(deep=True)

    # Capture input band names in order to drop these if drop=True
    if drop:
        bands_to_drop = list(ds.data_vars)
        print(f"Dropping bands {bands_to_drop}")

    def ratio(ds):
        return ds[cross_pol] / ds[co_pol]

    def purity(ds):
        return (1 - ratio(ds)) / (1 + ratio(ds))

    def theta(ds):
        return np.arctan((1 - ratio(ds))**2 / (1 + ratio(ds)**2 - ratio(ds)))

    def P1(ds):
        return 1 / (1 + ratio(ds))

    def P2(ds):
        return 1 - P1(ds)

    def entropy(ds):
        return P1(ds)*np.log2(P1(ds)) + P2(ds)*np.log2(P2(ds))

    # Dictionary containing remote sensing index band recipes
    index_dict = {
        # Radar Vegetation Index for dual-pol, Trudel et al. 2012
        "RVI": lambda ds: 4*ds[cross_pol] / (ds[co_pol] + ds[cross_pol]),
        # Vertical dual depolarization index, Periasamy 2018
        "VDDPI": lambda ds: (ds[co_pol] + ds[cross_pol]) / ds[co_pol],
        # cross-pol/co-pol ratio
        "ratio": ratio,
        # co-pol purity, Bhogapurapu et al. 2021
        "purity": purity,
        # pseudo scattering-type, Bhogapurapu et al. 2021
        "theta": theta,
        # pseudo scattering entropy, Bhogapurapu et al. 2021
        "entropy": entropy,
    }

    # If index supplied is not a list, convert to list. This allows us to
    # iterate through either multiple or single indices in the loop below
    indices = index if isinstance(index, list) else [index]

    # calculate for each index in the list of indices supplied (indexes)
    for index in indices:

        # Select an index function from the dictionary
        index_func = index_dict.get(str(index))

        # If no index is provided or if no function is returned due to an
        # invalid option being provided, raise an exception informing user to
        # choose from the list of valid options
        if index is None:

            raise ValueError(
                f"No radar `index` was provided. Please "
                "refer to the function \ndocumentation for a full "
                "list of valid options for `index` (e.g. 'RVI')"
            )

        elif index_func is None:

            raise ValueError(
                f"The selected index '{index}' is not one of the "
                "valid remote sensing index options. \nPlease "
                "refer to the function documentation for a full "
                "list of valid options for `index`"
            )

        # Apply index function
        index_array = index_func(ds)

        # Add as a new variable in dataset
        output_band_name = custom_varname if custom_varname else index
        ds[output_band_name] = index_array

    # Once all indexes are calculated, drop input bands if drop=True
    if drop:
        ds = ds.drop(bands_to_drop)

    # Return input dataset with added water index variable
    return ds

#========================
# from DEA spatial.py
#========================
def xr_rasterize(gdf,
                 da,
                 attribute_col=False,
                 crs=None,
                 transform=None,
                 name=None,
                 x_dim='x',
                 y_dim='y',
                 export_tiff=None,
                 verbose=False,
                 **rasterio_kwargs):
    """
    Rasterizes a geopandas.GeoDataFrame into an xarray.DataArray.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A geopandas.GeoDataFrame object containing the vector/shapefile
        data you want to rasterise.
    da : xarray.DataArray or xarray.Dataset
        The shape, coordinates, dimensions, and transform of this object 
        are used to build the rasterized shapefile. It effectively 
        provides a template. The attributes of this object are also 
        appended to the output xarray.DataArray.
    attribute_col : string, optional
        Name of the attribute column in the geodataframe that the pixels 
        in the raster will contain.  If set to False, output will be a 
        boolean array of 1's and 0's.
    crs : str, optional
        CRS metadata to add to the output xarray. e.g. 'epsg:3577'.
        The function will attempt get this info from the input 
        GeoDataFrame first.
    transform : affine.Affine object, optional
        An affine.Affine object (e.g. `from affine import Affine; 
        Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "6886890.0) giving the 
        affine transformation used to convert raster coordinates 
        (e.g. [0, 0]) to geographic coordinates. If none is provided, 
        the function will attempt to obtain an affine transformation 
        from the xarray object (e.g. either at `da.transform` or
        `da.geobox.transform`).
    x_dim : str, optional
        An optional string allowing you to override the xarray dimension 
        used for x coordinates. Defaults to 'x'. Useful, for example, 
        if x and y dims instead called 'lat' and 'lon'.   
    y_dim : str, optional
        An optional string allowing you to override the xarray dimension 
        used for y coordinates. Defaults to 'y'. Useful, for example, 
        if x and y dims instead called 'lat' and 'lon'.
    export_tiff: str, optional
        If a filepath is provided (e.g 'output/output.tif'), will export a
        geotiff file. A named array is required for this operation, if one
        is not supplied by the user a default name, 'data', is used
    verbose : bool, optional
        Print debugging messages. Default False.
    **rasterio_kwargs : 
        A set of keyword arguments to rasterio.features.rasterize
        Can include: 'all_touched', 'merge_alg', 'dtype'.

    Returns
    -------
    xarr : xarray.DataArray

    """
    from rasterio.features import rasterize
    import xarray as xr

    from datacube.utils.geometry import assign_crs

    # Check for a crs object
    try:
        crs = da.geobox.crs
    except:
        try:
            crs = da.crs
        except:
            if crs is None:
                raise ValueError("Please add a `crs` attribute to the "
                                 "xarray.DataArray, or provide a CRS using the "
                                 "function's `crs` parameter (e.g. crs='EPSG:3577')")

    # Check if transform is provided as a xarray.DataArray method.
    # If not, require supplied Affine
    if transform is None:
        try:
            # First, try to take transform info from geobox
            transform = da.geobox.transform
        # If no geobox
        except:
            try:
                # Try getting transform from 'transform' attribute
                transform = da.transform
            except:
                # If neither of those options work, raise an exception telling the
                # user to provide a transform
                raise TypeError("Please provide an Affine transform object using the "
                                "`transform` parameter (e.g. `from affine import "
                                "Affine; Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "
                                "6886890.0)`")

    # Grab the 2D dims (not time)
    try:
        dims = da.geobox.dims
    except:
        dims = y_dim, x_dim

    # Coords
    xy_coords = [da[dims[0]], da[dims[1]]]
    # xy_coords = [da['y'], da['x']]

    # Shape
    try:
        y, x = da.geobox.shape
    except:
        y, x = len(xy_coords[0]), len(xy_coords[1])

    # Reproject shapefile to match CRS of raster
    if verbose:
        print(f'Rasterizing to match xarray.DataArray dimensions ({y}, {x})')

    try:
        gdf_reproj = gdf.to_crs(crs=crs)
    except:
        # Sometimes the crs can be a datacube utils CRS object
        # so convert to string before reprojecting
        gdf_reproj = gdf.to_crs(crs={'init': str(crs)})

    # If an attribute column is specified, rasterise using vector
    # attribute values. Otherwise, rasterise into a boolean array
    if attribute_col:
        # Use the geometry and attributes from `gdf` to create an iterable
        shapes = zip(gdf_reproj.geometry, gdf_reproj[attribute_col])
    else:
        # Use geometry directly (will produce a boolean numpy array)
        shapes = gdf_reproj.geometry

    # Rasterise shapes into an array
    arr = rasterize(shapes=shapes,
                    out_shape=(y, x),
                    transform=transform,
                    **rasterio_kwargs)

    # Convert result to a xarray.DataArray
    xarr = xr.DataArray(arr,
                        coords=xy_coords,
                        dims=dims,
                        attrs=da.attrs,
                        name=name if name else None)

    # Add back crs if xarr.attrs doesn't have it
    if xarr.geobox is None:
        xarr = assign_crs(xarr, str(crs))

    if export_tiff:
        if verbose:
            print(f"Exporting GeoTIFF to {export_tiff}")
        write_cog(xarr,
                  export_tiff,
                  overwrite=True)

    return xarr
