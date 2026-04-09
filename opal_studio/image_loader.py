"""
Image loader for OME-TIFF files with pyramid support.

Uses tifffile with efficient caching and lazy page-level reading to handle 
massive multi-channel datasets without exhausting memory.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class LevelInfo:
    """Metadata for one pyramid resolution level."""
    index: int
    shape: tuple  # full shape of this level
    downsample: float  # relative to level-0
    _pages: list[tifffile.TiffPage] = field(default_factory=list)
    _cache: np.ndarray | None = None  # Full level data cache (only for smaller levels)


@dataclass
class ImageData:
    """
    Wraps an opened OME-TIFF, exposing pyramid levels and channel metadata.
    """
    path: Path
    is_rgb: bool = False
    channel_names: list[str] = field(default_factory=list)
    levels: list[LevelInfo] = field(default_factory=list)
    dtype: np.dtype = np.dtype("uint16")
    base_shape: tuple = ()
    axes: str = ""

    # internal handle kept alive for the life of the ImageData
    _tif: tifffile.TiffFile | None = None

    def get_full_channel_data(self, channel_idx: int, level: int = 0) -> np.ndarray:
        """Reads the full image for a single channel at the requested pyramid level."""
        if level >= len(self.levels):
            level = len(self.levels) - 1
        
        info = self.levels[level]
        if channel_idx < len(info._pages):
            return info._pages[channel_idx].asarray()
        return np.zeros((1, 1), dtype=self.dtype)


def open_image(path: str | Path) -> ImageData:
    """Open an OME-TIFF and return an *ImageData* (lazy — no pixels read yet)."""
    path = Path(path)
    tif = tifffile.TiffFile(str(path))

    series = tif.series[0]
    base_level = series.levels[0]
    base_shape = base_level.shape
    axes = series.axes.upper() if hasattr(series, 'axes') else ""
    dtype = base_level.dtype

    # ------------------------------------------------------------------
    # Detect RGB vs multi-channel
    # ------------------------------------------------------------------
    is_rgb = False
    if len(base_shape) == 3 and base_shape[-1] in (3, 4):
        is_rgb = True
    elif 'S' in axes and len(base_shape) >= 3:
        is_rgb = True
    elif len(base_shape) == 2:
        is_rgb = False

    # ------------------------------------------------------------------
    # Channel names from OME-XML
    # ------------------------------------------------------------------
    channel_names: list[str] = []
    if not is_rgb:
        channel_names = _extract_channel_names(tif, base_shape, axes)

    # ------------------------------------------------------------------
    # Build level list
    # ------------------------------------------------------------------
    levels: list[LevelInfo] = []
    base_y, base_x = _get_yx(base_shape, axes, is_rgb)

    for i, level_series in enumerate(series.levels):
        ly, lx = _get_yx(level_series.shape, axes, is_rgb)
        downsample = base_x / lx if lx > 0 else 1.0
        
        # Store page references for lazy access
        # level_series.pages is a Flat list of pages in this resolution level
        # For multi-channel OME-TIFF, each page is one channel
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


# ------------------------------------------------------------------
# Tile reading
# ------------------------------------------------------------------

def get_tile(
    img: ImageData,
    level_idx: int,
    channel: int | None,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """
    Read a rectangular region from one pyramid level.
    """
    lvl = img.levels[level_idx]
    
    # 1. Check full cache (usually for overview levels)
    if lvl._cache is not None:
        return _slice_array(lvl._cache, img.axes, img.is_rgb, channel, y_slice, x_slice)

    # 2. Lazy read from page (for hi-res levels where full-asarray is too much)
    # If the level is small enough, load it into cache now for next time
    n_pixels = 1
    for dim in lvl.shape: n_pixels *= dim
    
    # Cache if total size < 128 MB (per channel or per level depending on structure)
    if n_pixels * img.dtype.itemsize < 128 * 1024 * 1024:
        # Full level is small enough, cache it
        lvl._cache = img._tif.series[0].levels[level_idx].asarray()
        return _slice_array(lvl._cache, img.axes, img.is_rgb, channel, y_slice, x_slice)

    # 3. True lazy access: read only the required page and slice
    if img.is_rgb:
        # RGB images usually have 3 channels in one page
        page = lvl._pages[0] # assuming single page RGB
        try:
            return page.asarray(key=None)[y_slice, x_slice]
        except:
            return page.asarray()[y_slice, x_slice]
    else:
        # Multi-channel: each channel is its own page in our _pages list
        # We assume _pages index matches the logical channels
        if channel is not None and channel < len(lvl._pages):
            page = lvl._pages[channel]
            return page.asarray()[y_slice, x_slice]
        else:
            # Fallback for weird TIFFs
            return np.zeros((y_slice.stop - y_slice.start, x_slice.stop - x_slice.start), dtype=img.dtype)


def _slice_array(data, axes, is_rgb, channel, y_slice, x_slice):
    """Internal helper to slice a standard CZYX/YXC array."""
    ax_map = {a: i for i, a in enumerate(axes)}
    full_slice = [slice(None)] * data.ndim
    if 'Y' in ax_map: full_slice[ax_map['Y']] = y_slice
    if 'X' in ax_map: full_slice[ax_map['X']] = x_slice

    if is_rgb:
        return data[tuple(full_slice)]
    else:
        if 'C' in ax_map: full_slice[ax_map['C']] = channel
        elif 'S' in ax_map: full_slice[ax_map['S']] = channel
        elif data.ndim == 3:
            return data[channel, y_slice, x_slice]
        res = data[tuple(full_slice)]
        if res.ndim > 2:
            res = np.squeeze(res)
            while res.ndim > 2: res = res[0]
        return res


def best_level_for_zoom(img: ImageData, screen_pixels_per_image_pixel: float) -> int:
    """Pick the resolution level that most closely matches the screen pixels."""
    # If we are zoomed in (more screen pixels than image pixels), we MUST use level 0.
    if screen_pixels_per_image_pixel >= 1.0:
        return 0
    
    # Selection logic: pick first level that has at least 50% of the screen resolution
    for lvl in img.levels:
        if 1.0 / lvl.downsample >= screen_pixels_per_image_pixel * 0.5:
            return lvl.index
    return img.levels[-1].index


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_yx(shape: tuple, axes: str, is_rgb: bool) -> tuple[int, int]:
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


def _extract_channel_names(tif: tifffile.TiffFile, base_shape: tuple, axes: str) -> list[str]:
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
        return shape[0]  # assume CYX
    if len(shape) >= 4:
        return shape[1]  # assume TCYX
    return 1
