import numpy as np
import scipy.signal
import scipy.ndimage

data = np.random.rand(100, 100).astype(np.float32) * 1000
print(f"Input range: {data.min()} to {data.max()}")

filtered_sig = scipy.signal.medfilt2d(data, 3)
print(f"scipy.signal output range: {filtered_sig.min()} to {filtered_sig.max()}")

filtered_nd = scipy.ndimage.median_filter(data, size=3, mode='reflect')
print(f"scipy.ndimage output range: {filtered_nd.min()} to {filtered_nd.max()}")
