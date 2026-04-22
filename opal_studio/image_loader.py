"""
Image loader for OME-TIFF files with pyramid support.

Uses tifffile with efficient caching and lazy page-level reading to handle
massive multi-channel datasets without exhausting memory.

Public API (consumed by main_window, image_canvas, image_renderer):
  open_image(path)           -> ImageData
  get_tile(img, level, ch, y_slice, x_slice) -> np.ndarray   (unchanged)
  best_level_for_zoom(img, spp) -> int
  get_cached_tile(cache, img, level, ch, tile_row, tile_col, tile_size) -> np.ndarray  (NEW)
  ImageData.get_full_channel_data(ch, level) -> np.ndarray   (unchanged)

Future backends (OME-Zarr, etc.) should implement the same interface as
ImageData + expose compatible get_tile / get_cached_tile semantics.
"""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LevelInfo:
    """Metadata for one pyramid resolution level."""
    index: int
    shape: tuple        # full shape of this level
    downsample: float   # relative to level-0
    _pages: list = field(default_factory=list)          # tifffile.TiffPage refs
    _cache: Optional[np.ndarray] = None                 # full-level cache for small levels


@dataclass
class ImageData:
    """
    Wraps an opened OME-TIFF, exposing pyramid levels and channel metadata.

    Backend contract
    ----------------
    Any future image source (OME-Zarr, etc.) must expose:
      - is_rgb: bool
      - channel_names: list[str]
      - levels: list[LevelInfo]
      - dtype: np.dtype
      - base_shape: tuple
      - axes: str
      - get_full_channel_data(channel_idx, level) -> np.ndarray
    and support the same get_tile() / get_cached_tile() helpers below.
    """
    path: Path
    is_rgb: bool = False
    channel_names: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    dtype: np.dtype = np.dtype("uint16")
    base_shape: tuple = ()
    axes: str = ""

    # internal handle kept alive for the life of the ImageData
    _tif: tifffile.TiffFile = field(default=None, repr=False)

    def get_full_channel_data(self, channel_idx: int, level: int = 0) -> np.ndarray:
        """Reads the full image for a single channel at the requested pyramid level."""
        if level >= len(self.levels):
            level = len(self.levels) - 1

        info = self.levels[level]
        if channel_idx < len(info._pages):
            return info._pages[channel_idx].asarray()
        return np.zeros((1, 1), dtype=self.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# LRU Tile Cache
# ──────────────────────────────────────────────────────────────────────────────

class TileCache:
    """
    Thread-safe LRU cache for decoded raster tiles.

    Key: (level_idx, channel_idx_or_None, tile_row, tile_col, tile_size)
    Value: numpy array [tile_size, tile_size] or [tile_size, tile_size, C]

    Memory budget is approximate (based on array nbytes).
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key: tuple, data: np.ndarray) -> None:
        with self._lock:
            if key in self._cache:
                self._current_bytes -= self._cache[key].nbytes
                del self._cache[key]
            self._cache[key] = data
            self._current_bytes += data.nbytes
            self._cache.move_to_end(key)
            # Evict LRU entries until under budget
            while self._current_bytes > self._max_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._current_bytes -= evicted.nbytes

    def invalidate_channel(self, channel_idx: int) -> None:
        """Remove all cached tiles for a specific channel (e.g. after processed_data changes)."""
        with self._lock:
            keys_to_del = [k for k in self._cache if k[1] == channel_idx]
            for k in keys_to_del:
                self._current_bytes -= self._cache[k].nbytes
                del self._cache[k]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    @property
    def size_mb(self) -> float:
        return self._current_bytes / (1024 * 1024)


# ──────────────────────────────────────────────────────────────────────────────
# Image opening
# ──────────────────────────────────────────────────────────────────────────────

def open_image(path: str | Path) -> ImageData:
    """Open an OME-TIFF and return an *ImageData* (lazy — no pixels read yet)."""
    path = Path(path)
    tif = tifffile.TiffFile(str(path))

    series = tif.series[0]
    base_level = series.levels[0]
    base_shape = base_level.shape
    axes = series.axes.upper() if hasattr(series, 'axes') else ""
    dtype = base_level.dtype

    # ── Detect RGB vs multi-channel ──────────────────────────────────────────
    is_rgb = False
    if len(base_shape) == 3 and base_shape[-1] in (3, 4):
        is_rgb = True
    elif 'S' in axes and len(base_shape) >= 3:
        is_rgb = True
    elif len(base_shape) == 2:
        is_rgb = False

    # ── Channel names from OME-XML ───────────────────────────────────────────
    channel_names: list[str] = []
    if not is_rgb:
        channel_names = _extract_channel_names(tif, base_shape, axes)

    # ── Build level list ─────────────────────────────────────────────────────
    levels: list[LevelInfo] = []
    base_y, base_x = _get_yx(base_shape, axes, is_rgb)

    for i, level_series in enumerate(series.levels):
        ly, lx = _get_yx(level_series.shape, axes, is_rgb)
        downsample = base_x / lx if lx > 0 else 1.0

        # Store page references for lazy access.
        # For multi-channel OME-TIFF each page is one channel.
        pages = list(level_series.pages)

        levels.append(LevelInfo(
            index=i,
            shape=level_series.shape,
            downsample=downsample,
            _pages=pages,
        ))

    img = ImageData(
        path=path,
        is_rgb=is_rgb,
        channel_names=channel_names,
        levels=levels,
        dtype=dtype,
        base_shape=base_shape,
        axes=axes,
        _tif=tif,
    )
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Tile reading — original API (unchanged, used by segmentation/preprocessing)
# ──────────────────────────────────────────────────────────────────────────────

def get_tile(
    img: ImageData,
    level_idx: int,
    channel: int | None,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """
    Read a rectangular region from one pyramid level.

    This is the original API; segmentation and preprocessing workflows
    continue to call it directly via get_full_channel_data().
    """
    lvl = img.levels[level_idx]

    # 1. Check full-level cache (usually for overview levels)
    if lvl._cache is not None:
        return _slice_array(lvl._cache, img.axes, img.is_rgb, channel, y_slice, x_slice)

    # 2. If the level is small enough, cache it for future calls
    n_pixels = 1
    for dim in lvl.shape:
        n_pixels *= dim

    if n_pixels * img.dtype.itemsize < 128 * 1024 * 1024:
        lvl._cache = img._tif.series[0].levels[level_idx].asarray()
        return _slice_array(lvl._cache, img.axes, img.is_rgb, channel, y_slice, x_slice)

    # 3. True lazy access: read only the required page and slice
    if img.is_rgb:
        page = lvl._pages[0]
        try:
            return page.asarray(key=None)[y_slice, x_slice]
        except Exception:
            return page.asarray()[y_slice, x_slice]
    else:
        if channel is not None and channel < len(lvl._pages):
            page = lvl._pages[channel]
            return page.asarray()[y_slice, x_slice]
        else:
            return np.zeros(
                (y_slice.stop - y_slice.start, x_slice.stop - x_slice.start),
                dtype=img.dtype,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Cached tile API — NEW, used by the new renderer
# ──────────────────────────────────────────────────────────────────────────────

def get_cached_tile(
    cache: TileCache,
    img: ImageData,
    level_idx: int,
    channel: int | None,
    tile_row: int,
    tile_col: int,
    tile_size: int,
) -> np.ndarray:
    """
    Return a [tile_size, tile_size] (or smaller at edges) decoded array for
    one tile of the image pyramid, using the LRU tile cache.

    channel=None is used for RGB images (returns [H, W, 3/4]).
    """
    key = (level_idx, channel, tile_row, tile_col, tile_size)
    cached = cache.get(key)
    if cached is not None:
        return cached

    lvl = img.levels[level_idx]
    lh, lw = _get_yx(lvl.shape, img.axes, img.is_rgb)

    y0 = tile_row * tile_size
    x0 = tile_col * tile_size
    y1 = min(y0 + tile_size, lh)
    x1 = min(x0 + tile_size, lw)

    data = get_tile(img, level_idx, channel, slice(y0, y1), slice(x0, x1))
    # Ensure contiguous C array for fast numpy ops
    data = np.ascontiguousarray(data)
    cache.put(key, data)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Pyramid level selection
# ──────────────────────────────────────────────────────────────────────────────

def best_level_for_zoom(img: ImageData, screen_pixels_per_image_pixel: float) -> int:
    """
    Return the COARSEST pyramid level that still provides enough resolution
    for the current screen density.  Using a coarser level when zoomed out
    dramatically reduces how many pixels need to be read and composited.

    "Enough resolution": one level-pixel must cover no more than one
    screen-pixel worth of image data, i.e. (1 / downsample) >= spp.

    We iterate finest → coarsest, keep updating `best` as long as the
    condition holds, and stop as soon as a level is too coarse.  This
    returns the last (coarsest) level that still satisfies the criterion.
    """
    if screen_pixels_per_image_pixel >= 1.0:
        return 0  # Zoomed in past 1:1 — must use full-resolution level.

    best = 0  # fallback: finest level is always acceptable
    for lvl in img.levels:
        if 1.0 / lvl.downsample >= screen_pixels_per_image_pixel:
            best = lvl.index  # still good enough — try the next coarser level
        else:
            break             # this level is too coarse — stop
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _slice_array(data, axes, is_rgb, channel, y_slice, x_slice):
    """Slice a standard CZYX/YXC array."""
    ax_map = {a: i for i, a in enumerate(axes)}
    full_slice = [slice(None)] * data.ndim
    if 'Y' in ax_map:
        full_slice[ax_map['Y']] = y_slice
    if 'X' in ax_map:
        full_slice[ax_map['X']] = x_slice

    if is_rgb:
        return data[tuple(full_slice)]
    else:
        if 'C' in ax_map:
            full_slice[ax_map['C']] = channel
        elif 'S' in ax_map:
            full_slice[ax_map['S']] = channel
        elif data.ndim == 3:
            return data[channel, y_slice, x_slice]
        res = data[tuple(full_slice)]
        if res.ndim > 2:
            res = np.squeeze(res)
            while res.ndim > 2:
                res = res[0]
        return res


def _get_yx(shape: tuple, axes: str, is_rgb: bool) -> tuple:
    """Return (height, width) from a shape tuple."""
    if axes and "Y" in axes and "X" in axes:
        yi = axes.index("Y")
        xi = axes.index("X")
        return shape[yi], shape[xi]
    if is_rgb:
        return shape[0], shape[1]
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3:
        return shape[1], shape[2]  # Assume CYX
    return shape[-2], shape[-1]


def _extract_channel_names(tif: tifffile.TiffFile, base_shape: tuple, axes: str) -> list:
    """Pull channel names from OME-XML, falling back to Channel 0, 1, …"""
    n_channels = _get_num_channels(base_shape, axes)
    try:
        ome_xml = tif.ome_metadata
        if ome_xml:
            root = ET.fromstring(ome_xml)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            channels = root.findall(".//ome:Channel", ns)
            if not channels:
                channels = root.findall(".//{*}Channel")
            names = []
            for ch in channels[:n_channels]:
                name = ch.get("Name") or ch.get("ID") or f"Channel {len(names)}"
                names.append(name)
            if len(names) == n_channels:
                return names
    except Exception:
        pass
    return [f"Channel {i}" for i in range(n_channels)]


def _get_num_channels(shape: tuple, axes: str) -> int:
    if axes and "C" in axes:
        return shape[axes.index("C")]
    if len(shape) == 2:
        return 1
    if len(shape) == 3:
        return shape[0]    # assume CYX
    if len(shape) >= 4:
        return shape[1]    # assume TCYX
    return 1
