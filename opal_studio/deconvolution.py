"""Colour deconvolution of brightfield / H&E RGB into single stain channels.

An H&E scan carries two stains mixed into one RGB image: haematoxylin, which binds
nuclei, and eosin, which binds cytoplasm and stroma. Separating them gives a
single-channel image per stain, which is what the nuclei models expect — they were
trained on fluorescence, where one channel is one marker.

The separation is the Ruifrok–Johnston method as implemented by
``skimage.color.rgb2hed``. Its output is **optical density**: a value rises with how
much of that stain is present, so a haematoxylin-stained nucleus comes out *bright*
against *dark* background. That is already the polarity the models want, so nothing
is inverted here — inverting would hand them dark nuclei on a bright field and the
segmentation would fail. See ``hematoxylin`` for the measured numbers.
"""

from __future__ import annotations

import numpy as np

STAINS = {"hematoxylin": 0, "eosin": 1, "dab": 2}


def _as_float_rgb(rgb: np.ndarray) -> np.ndarray:
    """Coerce a raster to (H, W, 3) float in 0..1, which is what rgb2hed expects."""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"expected an RGB raster, got shape {arr.shape}")
    arr = arr[..., :3]

    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)

    arr = arr.astype(np.float32)
    top = float(arr.max())
    return arr / top if top > 1.0 else arr


def separate(rgb: np.ndarray, stain: str = "hematoxylin") -> np.ndarray:
    """Return one stain's optical density from an RGB raster, as float32 (H, W).

    Higher means more of that stain, so nuclei are bright in the haematoxylin
    channel. Values are small in absolute terms (typically 0–0.2); callers
    normalise as they would any other intensity channel.
    """
    from skimage.color import rgb2hed

    try:
        idx = STAINS[stain]
    except KeyError:
        raise ValueError(f"unknown stain {stain!r}; expected one of {sorted(STAINS)}")

    # rgb2hed takes log of the input, so a true zero would be -inf.
    src = np.clip(_as_float_rgb(rgb), 1e-6, 1.0)
    return np.ascontiguousarray(rgb2hed(src)[..., idx].astype(np.float32))


def hematoxylin(rgb: np.ndarray) -> np.ndarray:
    """The haematoxylin channel: bright nuclei on a dark field.

    Measured on an H&E scan, over the darkest and lightest tenth of pixels by
    luminance (nuclei and background respectively): mean 0.078 on nuclei against
    0.012 on background. No inversion is applied, and none is wanted — the values
    already rise with stain, which is the polarity every nuclei model expects.
    """
    return separate(rgb, "hematoxylin")
