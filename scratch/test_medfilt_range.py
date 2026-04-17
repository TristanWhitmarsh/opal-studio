import numpy as np
import scipy.signal

# Try with very large values
data = np.full((10, 10), 50000, dtype=np.float32)
data[5, 5] = 60000
print(f"Input: min={data.min()}, max={data.max()}")

filtered = scipy.signal.medfilt2d(data, 3)
print(f"Filtered: min={filtered.min()}, max={filtered.max()}")

# Try with very small values
data2 = np.full((10, 10), 0.0001, dtype=np.float32)
print(f"Input2: min={data2.min()}, max={data2.max()}")
filtered2 = scipy.signal.medfilt2d(data2, 3)
print(f"Filtered2: min={filtered2.min()}, max={filtered2.max()}")
