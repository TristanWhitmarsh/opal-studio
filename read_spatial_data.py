import json
import os
from pathlib import Path

import numpy as np
import numcodecs
import tifffile as tiff

# Input parameters
sdata_path = Path('images/SpatialData/20260202_CRU00363794-051_NET01_20260202-133826-578602')
channel_name = 'Ir(191)'
verbose = True

def read_zarr_v3_image(sdata_path, image_name, channel_name):
    """
    Manually read a Zarr V3 image channel since older spatialdata versions 
    lack native support for the Zarr V3 specification used in this dataset.
    """
    img_path = sdata_path / 'images' / image_name
    zarr_json_path = img_path / 'zarr.json'
    
    if not zarr_json_path.exists():
        return None, f"Metadata not found at {zarr_json_path}"

    with open(zarr_json_path, 'r') as f:
        meta = json.load(f)
    
    # Parse OME-Zarr V3 channels
    channels = meta['attributes']['ome']['omero']['channels']
    channel_labels = [c['label'] for c in channels]
    
    if channel_name not in channel_labels:
        return None, f"Channel '{channel_name}' not found in {image_name}"
    
    c_idx = channel_labels.index(channel_name)
    
    # Read highest resolution (scale 0)
    scale0_path = img_path / '0'
    with open(scale0_path / 'zarr.json', 'r') as f:
        s_meta = json.load(f)
    
    shape = s_meta['shape'] # [c, y, x]
    c_shape = s_meta['chunk_grid']['configuration']['chunk_shape'] # [c, y, x]
    dtype = np.dtype(s_meta['data_type'])
    
    # Initialize output array (Y, X)
    out = np.zeros(shape[1:], dtype=dtype)
    
    # Path to chunks for this channel
    c_dir = scale0_path / 'c' / str(c_idx)
    if not c_dir.exists():
        return None, f"Chunk directory {c_dir} not found"

    decoder = numcodecs.Zstd()
    
    for y_dir in c_dir.iterdir():
        if not y_dir.is_dir(): continue
        y_idx = int(y_dir.name)
        for x_file in y_dir.iterdir():
            x_idx = int(x_file.name)
            
            with open(x_file, 'rb') as f:
                comp_data = f.read()
            
            decomp = decoder.decode(comp_data)
            chunk_arr = np.frombuffer(decomp, dtype=dtype).reshape(c_shape[1:])
            
            ys, ye = y_idx * c_shape[1], min((y_idx + 1) * c_shape[1], shape[1])
            xs, xe = x_idx * c_shape[2], min((x_idx + 1) * c_shape[2], shape[2])
            
            out[ys:ye, xs:xe] = chunk_arr[:ye-ys, :xe-xs]
            
    return out, None

if __name__ == "__main__":
    print(f"Opening SpatialData at: {sdata_path}")

    # Manual Zarr V3 reader (required for compatibility with Python 3.9 environment)
    print(f"Reading full resolution image (scale 0) for {channel_name}...")
    raw_data, error = read_zarr_v3_image(sdata_path, 'stitched', channel_name)

    if raw_data is not None:
        # Remove NaNs and Infs in-place for cleaner image output
        nan_count = np.isnan(raw_data).sum()
        if nan_count > 0:
            print(f"Note: Found and removed {nan_count} NaN values.")
        
        np.nan_to_num(raw_data, copy=False)
        
        output_file = "image.tif"
        tiff.imwrite(output_file, raw_data)
        print(f"Successfully saved full-res {channel_name} to {output_file}")
    else:
        print(f"Error: {error}")
