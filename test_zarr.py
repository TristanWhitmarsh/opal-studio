import tifffile
import zarr

filename = r'c:\opal-studio\images\20230714_EMMA_20230714-181349-894426_Q002.ome.tiff'
tif = tifffile.TiffFile(filename)
level = tif.series[0].levels[1]
z_store = level.aszarr()
z_arr = zarr.open(z_store, mode='r')

if isinstance(z_arr, zarr.hierarchy.Group):
    print("Level 1 aszarr returns Group with keys:", list(z_arr.keys()))
    for k in z_arr.keys():
        print(f'Key {k} shape:', z_arr[k].shape)
else:
    print("Level 1 aszarr returns Array with shape:", z_arr.shape)
