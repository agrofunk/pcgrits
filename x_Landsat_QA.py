# Flags definition por Landsat 8-9 
# https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Frequently_used_code/Cloud_and_pixel_quality_masking.html

flags_def = {'cirrus': {'bits': 2,
            'values': {'0': 'not_high_confidence', '1': 'high_confidence'}},
 'cirrus_confidence': {'bits': [14, 15],
                       'values': {'0': 'none',
                                  '1': 'low',
                                  '2': 'reserved',
                                  '3': 'high'}},
 'clear': {'bits': 6, 'values': {'0': False, '1': True}},
 'cloud': {'bits': 3,
           'values': {'0': 'not_high_confidence', '1': 'high_confidence'}},
 'cloud_confidence': {'bits': [8, 9],
                      'values': {'0': 'none',
                                 '1': 'low',
                                 '2': 'medium',
                                 '3': 'high'}},
 'cloud_shadow': {'bits': 4,
                  'values': {'0': 'not_high_confidence',
                             '1': 'high_confidence'}},
 'cloud_shadow_confidence': {'bits': [10, 11],
                             'values': {'0': 'none',
                                        '1': 'low',
                                        '2': 'reserved',
                                        '3': 'high'}},
 'dilated_cloud': {'bits': 1, 'values': {'0': 'not_dilated', '1': 'dilated'}},
 'nodata': {'bits': 0, 'values': {'0': False, '1': True}},
 'snow': {'bits': 5,
          'values': {'0': 'not_high_confidence', '1': 'high_confidence'}},
 'snow_ice_confidence': {'bits': [12, 13],
                         'values': {'0': 'none',
                                    '1': 'low',
                                    '2': 'reserved',
                                    '3': 'high'}},
 'water': {'bits': 7, 'values': {'0': 'land_or_cloud', '1': 'water'}}}


def apply_bitmask(arr) -> xr.DataArray or np.array:
    """Apply QA pixel bit mask for each array depending on platform"""

    unique_platform = np.unique(arr.platform.to_numpy())

    if ["landsat-8", "landsat-9"] in unique_platform:
        mask_bitfields = [1, 2, 3, 4]  # dilated cloud, cirrus, cloud, cloud shadow
    elif ["landsat-4", "landsat-5", "landsat-7"] in unique_platform:
        mask_bitfields = [1, 3, 4, 5]  # dilated cloud, cirrus, cloud, cloud shadow
    elif ["landsat-4", "landsat-5"] in unique_platform:
        mask_bitfields = [1, 3, 4, 5]  # dilated cloud, cirrus, cloud, cloud shadow
    else:
        raise ValueError(f"No bit mask defined for {arr.platform.to_numpy()}")

    print(unique_platform)
    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    qa = arr.sel(band="qa").astype("uint16")
    bad = qa & bitmask  # just look at those 4 bits

    arr = arr.where(bad == 0)

    return arr

# https://archive.li/wykEi
L8_flags = {'dilated_cloud': 1<<1,
        'cirrus': 1<<2, 
        'cloud': 1<<3,
        'shadow': 1<<4, 
        'snow': 1<<5, 
        'clear': 1<<6,
        'water': 1<<7}

def get_mask(mask, flags_list):
	    
# first we will create the result mask filled with zeros and the same shape as the mask
    final_mask = np.zeros_like(mask)
	    
    # then we will loop through the flags and add the 
    for flag in flags_list:
        # get the mask for this flag
        flag_mask = np.bitwise_and(mask, L8_flags[flag])
        
        # add it to the final flag
        final_mask = final_mask | flag_mask

        return final_mask > 0

    
    clouds = get_mask(imgqa, ['cirrus', 'cloud', 'dilated_cloud'])
    shadows = get_mask(imgqa, ['shadow'])
	
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(clouds)
    ax[1].imshow(shadows)


