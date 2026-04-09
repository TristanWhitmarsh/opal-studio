import sys
sys.path.insert(0, r'c:\opal-studio')
from opal_studio.image_loader import open_image, get_tile, _get_yx

img = open_image(r'c:\opal-studio\images\EMMA1.ome.tiff')
print('axes:', img.axes)
print('is_rgb:', img.is_rgb)
print('shape:', img.base_shape)
print('channels:', len(img.channel_names))
print('channel_names:', img.channel_names[:5])
print('levels:', len(img.levels))
for i, lvl in enumerate(img.levels):
    print(f'  level {i}: shape={lvl.shape} downsample={lvl.downsample:.1f}')

# Try reading a tile
tile = get_tile(img, len(img.levels)-1, 0, slice(0, 100), slice(0, 100))
print('tile shape:', tile.shape, 'dtype:', tile.dtype)
print('tile min/max:', tile.min(), tile.max())
