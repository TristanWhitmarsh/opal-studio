import ctypes
from pathlib import Path

import numpy as np
from scipy import LowLevelCallable, ndimage

pwd = Path(__file__).parent.resolve()

# Try to load the native C MAD filter (cross-platform)
mad_filter_llc = None
for pattern in ["nice_filters*.so", "nice_filters*.dll", "nice_filters*.pyd", "nice_filters*.dylib"]:
    matches = list(pwd.glob(pattern))
    if not matches:
        matches = list(pwd.parent.glob(pattern))
    if matches:
        try:
            clib = ctypes.cdll.LoadLibrary(str(matches[0]))
            clib.mad_filter.restype = ctypes.c_int
            clib.mad_filter.argtypes = (
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_long,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_void_p,
            )
            mad_filter_llc = LowLevelCallable(clib.mad_filter)
        except Exception:
            pass
        break


def _py_mad_filter(buffer):
    """Pure-Python fallback for the MAD filter."""
    return np.median(np.abs(buffer - np.median(buffer)))


def run(image, threshold=10, npass=3, filter_size=5):
    filter_func = mad_filter_llc if mad_filter_llc is not None else _py_mad_filter
    img = image.copy()
    for i in range(npass):
        img_b = ndimage.median_filter(img, size=[filter_size, filter_size])
        img_r = 1.48 * ndimage.generic_filter(
            img, filter_func, [filter_size, filter_size]
        )
        difference = np.abs(img - img_b)
        filtered = np.where(difference > threshold * img_r, img_b, img)
        img = filtered
    return img
