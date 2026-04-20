import numpy as np
from skimage.segmentation import find_boundaries
import time

# Simulate a large mask (4000x4000) with many cells
labels = np.zeros((4000, 4000), dtype=np.int32)
for i in range(100):
    for j in range(100):
        y, x = i*40 + 20, j*40 + 20
        labels[y-10:y+10, x-10:x+10] = i*100 + j + 1

start = time.perf_counter()
boundaries = find_boundaries(labels, mode='inner')
end = time.perf_counter()
print(f"find_boundaries took {end - start:.4f}s")

from skimage.measure import find_contours
start = time.perf_counter()
# find_contours on the whole thing? No, that finds surfaces of values.
# If we want contours of EACH cell, we have to loop.
count = 0
for i in range(1, 10001):
    c = find_contours(labels == i, 0.5)
    count += len(c)
end = time.perf_counter()
print(f"find_contours (per cell) took {end - start:.4f}s for 10000 cells")
