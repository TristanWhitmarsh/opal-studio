"""
Image loader for OME-TIFF and SpatialData (Zarr V3) files with pyramid support.

Uses tifffile with efficient caching and lazy page-level reading to handle
massive multi-channel datasets without exhausting memory.  SpatialData
directories are read without the spatialdata library by manually decoding
Zarr V3 chunks (zstd-compressed) via numcodecs.

Public API (consumed by main_window, image_canvas, image_renderer):
  open_image(path)              -> ImageData   (OME-TIFF)
  open_spatialdata(path)        -> ImageData   (SpatialData / Zarr V3)
  open_spatialdata_collection(path) -> SpatialDataCollection
  get_tile(img, level, ch, y_slice, x_slice) -> np.ndarray   (unchanged)
  best_level_for_zoom(img, spp) -> int
  get_cached_tile(cache, img, level, ch, tile_row, tile_col, tile_size) -> np.ndarray
  ImageData.get_full_channel_data(ch, level) -> np.ndarray   (unchanged)

Both backends expose an identical ImageData object so all rendering,
segmentation and preprocessing code works without modification.
"""

from __future__ import annotations

import json
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

try:
    import numcodecs as _numcodecs
    _ZSTD_DECODER = _numcodecs.Zstd()
except ImportError:
    _numcodecs = None  # type: ignore
    _ZSTD_DECODER = None


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
    _zarr: Optional[object] = None                      # zarr array (or ZarrV3Array) for lazy reading


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

        # SpatialData backend: _zarr is a ZarrV3Array
        if isinstance(info._zarr, ZarrV3Array):
            _, h, w = info._zarr.shape
            return info._zarr[channel_idx, 0:h, 0:w]

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

    def __init__(self, max_bytes: int = 4 * 1024 * 1024 * 1024):  # 4 GB default
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

        # Try to create a zarr store for fast chunked access
        z_arr = None
        try:
            import zarr
            z_store = level_series.aszarr()
            z_arr = zarr.open(z_store, mode='r')
            if isinstance(z_arr, zarr.hierarchy.Group):
                if '0' in z_arr:
                    z_arr = z_arr['0']
        except Exception:
            pass

        levels.append(LevelInfo(
            index=i,
            shape=level_series.shape,
            downsample=downsample,
            _pages=pages,
            _zarr=z_arr,
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
# ZarrV3Array — lightweight Zarr V3 chunk reader (no zarr library needed)
# ──────────────────────────────────────────────────────────────────────────────

class ZarrV3Array:
    """
    Minimal numpy-compatible reader for a Zarr V3 array stored on disk.

    Supports `arr[c, y0:y1, x0:x1]` style indexing which is all that the
    image renderer (via _read_channel_slice) and segmentation engine need.

    Chunks are zstd-compressed raw bytes. The chunk directory layout follows
    the Zarr V3 default separator convention: ``c/y_tile/x_tile``.

    Thread-safety: multiple threads may call __getitem__ concurrently because
    each call opens files independently (no shared file handles).
    """

    def __init__(self, array_path: Path, meta: dict):
        self._path = array_path
        self._shape = tuple(meta['shape'])         # (C, Y, X)
        cs = meta['chunk_grid']['configuration']['chunk_shape']
        self._chunk_shape = tuple(cs)              # (c_c, c_y, c_x)
        self._dtype = np.dtype(meta['data_type'])
        self._fill_value = meta.get('fill_value', 0)
        # Detect codec from codecs list
        codecs = meta.get('codecs', [])
        self._codec = None
        for codec in codecs:
            name = codec.get('name', '')
            if name == 'bytes':
                endian = codec.get('configuration', {}).get('endian')
                if endian == 'little':
                    self._dtype = self._dtype.newbyteorder('<')
                elif endian == 'big':
                    self._dtype = self._dtype.newbyteorder('>')
            if name == 'zstd' and _ZSTD_DECODER is not None:
                self._codec = _ZSTD_DECODER

        key_encoding = meta.get('chunk_key_encoding', {})
        self._chunk_key_name = key_encoding.get('name', 'default')
        self._chunk_separator = key_encoding.get('configuration', {}).get('separator', '/')

        # All channels in a SpatialData tile are commonly stored in one chunk.
        # Cache decoded chunks so concurrent per-channel rendering reads and
        # decompresses each physical chunk only once.
        self._decoded_chunks: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._decoded_chunks_lock = threading.Lock()
        self._chunk_locks: dict[tuple[int, int, int], threading.Lock] = {}
        self._max_decoded_chunks = 16

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def channel_maxima(self) -> list[float]:
        """Compute maxima for every channel while decoding each chunk once."""
        C, H, W = self._shape
        cc, cy, cx = self._chunk_shape
        maxima = np.full(C, -np.inf, dtype=np.float64)
        for c_chunk in range((C + cc - 1) // cc):
            for y_chunk in range((H + cy - 1) // cy):
                for x_chunk in range((W + cx - 1) // cx):
                    chunk = self._read_chunk(c_chunk, y_chunk, x_chunk)
                    if chunk is None:
                        continue
                    actual_c = min(cc, C - c_chunk * cc)
                    actual_y = min(cy, H - y_chunk * cy)
                    actual_x = min(cx, W - x_chunk * cx)
                    valid = chunk[:actual_c, :actual_y, :actual_x]
                    if np.issubdtype(valid.dtype, np.floating):
                        chunk_maxima = np.nanmax(valid, axis=(1, 2))
                    else:
                        chunk_maxima = np.max(valid, axis=(1, 2))
                    start = c_chunk * cc
                    maxima[start:start + actual_c] = np.maximum(
                        maxima[start:start + actual_c], chunk_maxima
                    )
        return [float(value) if np.isfinite(value) and value > 0 else 1.0
                for value in maxima]

    def __getitem__(self, idx):
        """
        idx must be a 3-tuple: (c_int, y_slice, x_slice)
        Returns a 2-D numpy array of shape (y1-y0, x1-x0).
        """
        if not isinstance(idx, tuple) or len(idx) != 3:
            raise IndexError(f"ZarrV3Array requires 3-element index, got {idx!r}")

        c_idx, y_sl, x_sl = idx
        if not isinstance(y_sl, slice) or not isinstance(x_sl, slice):
            raise IndexError("y and x indices must be slices")

        C, H, W = self._shape
        cc, cy, cx = self._chunk_shape

        y0 = max(0, y_sl.start if y_sl.start is not None else 0)
        y1 = min(H, y_sl.stop  if y_sl.stop  is not None else H)
        x0 = max(0, x_sl.start if x_sl.start is not None else 0)
        x1 = min(W, x_sl.stop  if x_sl.stop  is not None else W)

        if not 0 <= c_idx < C:
            raise IndexError(f"channel index {c_idx} outside array with {C} channels")

        out = np.full((y1 - y0, x1 - x0), self._fill_value, dtype=self._dtype)

        # Which chunk tiles overlap with the requested region?
        c_chunk = c_idx // cc   # c tile index (typically == c_idx since cc=1)
        ty_start = y0 // cy
        ty_end   = (y1 - 1) // cy
        tx_start = x0 // cx
        tx_end   = (x1 - 1) // cx

        for ty in range(ty_start, ty_end + 1):
            for tx in range(tx_start, tx_end + 1):
                chunk = self._read_chunk(c_chunk, ty, tx)
                if chunk is None:
                    continue  # fill_value (0) already in out
                channel_in_chunk = c_idx - c_chunk * cc
                channel_data = chunk[channel_in_chunk]

                # Region of this chunk that overlaps with request
                t_y0 = ty * cy;  t_y1 = t_y0 + cy
                t_x0 = tx * cx;  t_x1 = t_x0 + cx

                # Overlap in global coords
                gy0 = max(y0, t_y0);  gy1 = min(y1, t_y1)
                gx0 = max(x0, t_x0);  gx1 = min(x1, t_x1)
                if gy1 <= gy0 or gx1 <= gx0:
                    continue

                # Clip chunk to valid data size (edge chunks may be smaller)
                cy_real = min(cy, H - t_y0)
                cx_real = min(cx, W - t_x0)

                # Source slice inside chunk array
                sy0 = gy0 - t_y0;  sy1 = gy1 - t_y0
                sx0 = gx0 - t_x0;  sx1 = gx1 - t_x0
                sy1 = min(sy1, cy_real)
                sx1 = min(sx1, cx_real)
                if sy1 <= sy0 or sx1 <= sx0:
                    continue

                # Destination slice inside out array
                dy0 = gy0 - y0;  dy1 = dy0 + (sy1 - sy0)
                dx0 = gx0 - x0;  dx1 = dx0 + (sx1 - sx0)

                out[dy0:dy1, dx0:dx1] = channel_data[sy0:sy1, sx0:sx1]

        # SpatialData stores float32; replace NaN/Inf with 0 for clean rendering
        if np.issubdtype(self._dtype, np.floating):
            np.nan_to_num(out, copy=False)

        return out

    def _chunk_path(self, c_chunk: int, y_chunk: int, x_chunk: int) -> Path:
        indices = [str(c_chunk), str(y_chunk), str(x_chunk)]
        if self._chunk_key_name == 'default':
            indices.insert(0, 'c')
        key = self._chunk_separator.join(indices)
        return self._path / key

    def _read_chunk(self, c_chunk: int, y_chunk: int, x_chunk: int) -> Optional[np.ndarray]:
        key = (c_chunk, y_chunk, x_chunk)
        with self._decoded_chunks_lock:
            cached = self._decoded_chunks.get(key)
            if cached is not None:
                self._decoded_chunks.move_to_end(key)
                return cached
            key_lock = self._chunk_locks.setdefault(key, threading.Lock())

        with key_lock:
            with self._decoded_chunks_lock:
                cached = self._decoded_chunks.get(key)
                if cached is not None:
                    self._decoded_chunks.move_to_end(key)
                    return cached

            chunk_path = self._chunk_path(c_chunk, y_chunk, x_chunk)
            if not chunk_path.exists():
                return None

            with open(chunk_path, 'rb') as fh:
                raw = fh.read()
            if self._codec is not None:
                raw = self._codec.decode(raw)

            cc, cy, cx = self._chunk_shape
            C, H, W = self._shape
            actual_shape = (
                min(cc, C - c_chunk * cc),
                min(cy, H - y_chunk * cy),
                min(cx, W - x_chunk * cx),
            )
            values = np.frombuffer(raw, dtype=self._dtype)
            full_size = int(np.prod(self._chunk_shape))
            actual_size = int(np.prod(actual_shape))
            if values.size == full_size:
                chunk = values.reshape(self._chunk_shape).copy()
            elif values.size == actual_size:
                chunk = values.reshape(actual_shape).copy()
            else:
                raise ValueError(
                    f"Unexpected decoded chunk size {values.size} at {chunk_path}; "
                    f"expected {full_size} or {actual_size} values"
                )

            with self._decoded_chunks_lock:
                self._decoded_chunks[key] = chunk
                self._decoded_chunks.move_to_end(key)
                while len(self._decoded_chunks) > self._max_decoded_chunks:
                    old_key, _ = self._decoded_chunks.popitem(last=False)
                    self._chunk_locks.pop(old_key, None)
            return chunk


# ──────────────────────────────────────────────────────────────────────────────
# SpatialData directory opener
# ──────────────────────────────────────────────────────────────────────────────

def _parse_mcd_panel(sdata_path: Path) -> dict:
    """
    Parse extras/mcd_schema.xml (Standard BioTools / Fluidigm MCD format) and
    return a dict mapping metal-label -> protein-label.

    Example: {'Ir(191)': 'DNA1', 'Gd(155)': 'PDPN', ...}
    Returns an empty dict if the file is missing or unparseable.
    """
    xml_path = sdata_path / 'extras' / 'mcd_schema.xml'
    if not xml_path.exists():
        return {}
    try:
        import re
        content = xml_path.read_text(encoding='utf-8', errors='replace')
        pattern = r'<AcquisitionChannel\b[^>]*>([\s\S]*?)</AcquisitionChannel>'
        blocks = re.findall(pattern, content)

        def get_tag(block, tag):
            m = re.search(r'<' + tag + r'>(.*?)</' + tag + r'>', block)
            return m.group(1).strip() if m else ''

        panel: dict[str, str] = {}
        for block in blocks:
            metal = get_tag(block, 'ChannelName')
            label = get_tag(block, 'ChannelLabel')
            if metal and label and label not in (metal, ''):
                # Strip the leading mass number prefix (e.g. "141Pr_aSMA" -> "aSMA")
                clean = re.sub(r'^\d+[A-Za-z]+_?', '', label)
                if clean:
                    panel[metal] = clean
        return panel
    except Exception:
        return {}


def spatialdata_channel_maxima(img: 'ImageData') -> list[float]:
    """
    Return a list of per-channel maximum values read from the coarsest pyramid
    level in a single pass.  Uses ZarrV3Array so no additional I/O libraries
    are required.

    Returns a list of floats, length == number of channels.
    All-zero channels get a placeholder max of 1.0 so the slider stays usable.
    """
    if not img.levels:
        return [1.0] * len(img.channel_names)

    coarsest = img.levels[-1]
    z_arr = coarsest._zarr
    if not isinstance(z_arr, ZarrV3Array):
        return [1.0] * len(img.channel_names)

    return z_arr.channel_maxima()


class SpatialDataCollection:
    """Metadata-only view of the image elements in a SpatialData store."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.images_dir = self.path / 'images'
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"No 'images' directory found in {self.path}")

        self._metadata: dict[str, dict] = {}
        root_meta_path = self.path / 'zarr.json'
        if root_meta_path.exists():
            with open(root_meta_path, 'r') as fh:
                root_meta = json.load(fh)
            metadata = root_meta.get('consolidated_metadata', {}).get('metadata', {})
            if isinstance(metadata, dict):
                self._metadata = metadata

        image_names = []
        for key, meta in self._metadata.items():
            if not key.startswith('images/'):
                continue
            relative = key[len('images/'):]
            if relative and '/' not in relative and meta.get('node_type') == 'group':
                image_names.append(relative)

        if not image_names:
            image_names = [
                item.name for item in self.images_dir.iterdir()
                if item.is_dir() and (item / 'zarr.json').exists()
            ]

        def natural_key(value: str):
            import re
            return [int(part) if part.isdigit() else part.lower()
                    for part in re.split(r'(\d+)', value)]

        self.image_names = sorted(set(image_names), key=natural_key)
        if 'stitched' in self.image_names:
            self.image_names.remove('stitched')
            self.image_names.insert(0, 'stitched')
        if not self.image_names:
            raise FileNotFoundError(f"No valid image group found inside {self.images_dir}")

        self._panel = _parse_mcd_panel(self.path)

    def __len__(self) -> int:
        return len(self.image_names)

    def open_image(self, image: int | str = 0) -> ImageData:
        if isinstance(image, int):
            try:
                image_name = self.image_names[image]
            except IndexError as exc:
                raise IndexError(f"SpatialData image index {image} is out of range") from exc
        else:
            image_name = image
            if image_name not in self.image_names:
                raise KeyError(f"SpatialData image '{image_name}' was not found")
        return open_spatialdata(
            self.path,
            image_name,
            _metadata=self._metadata,
            _panel=self._panel,
        )


def open_spatialdata_collection(path: str | Path) -> SpatialDataCollection:
    """Open a SpatialData store without reading any image pixels."""
    return SpatialDataCollection(path)


def open_spatialdata(
    path: str | Path,
    image_name: str | None = None,
    *,
    _metadata: dict[str, dict] | None = None,
    _panel: dict | None = None,
) -> ImageData:
    """
    Open a SpatialData directory and return an *ImageData* compatible with the
    OME-TIFF backend.

    The directory must contain an ``images/`` sub-directory holding one or more
    OME-Zarr V3 image groups.  Each group has a ``zarr.json`` with
    ``attributes.ome.omero.channels`` and ``attributes.ome.multiscales``.

    If ``extras/mcd_schema.xml`` is present (Standard BioTools MCD format),
    channel names are enriched to ``"Metal(mass) / Protein"`` notation.

    Parameters
    ----------
    path:
        Path to the SpatialData root directory (contains ``images/``).
    image_name:
        Name of the image sub-group inside ``images/`` to open.  If *None*,
        the function picks the first group it finds that has a valid
        ``zarr.json``.

    Returns
    -------
    ImageData
        Fully populated with pyramid levels backed by *ZarrV3Array* objects.
        The ``_tif`` field is *None* (SpatialData has no TiffFile).
    """
    sdata_path = Path(path)
    images_dir = sdata_path / 'images'
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No 'images' directory found in {sdata_path}")

    # ── Discover image group ────────────────────────────────────────────────
    if image_name is None:
        # Try 'stitched' first (common SpatialData convention), then any group
        candidates = ['stitched'] + sorted(
            d.name for d in images_dir.iterdir()
            if d.is_dir() and d.name != 'stitched' and (d / 'zarr.json').exists()
        )
        image_name = next(
            (c for c in candidates if (images_dir / c / 'zarr.json').exists()),
            None
        )
        if image_name is None:
            raise FileNotFoundError(f"No valid image group found inside {images_dir}")

    img_root = images_dir / image_name
    group_meta_path = img_root / 'zarr.json'
    group_meta = (_metadata or {}).get(f'images/{image_name}')
    if group_meta is None:
        if not group_meta_path.exists():
            raise FileNotFoundError(f"zarr.json not found at {group_meta_path}")
        with open(group_meta_path, 'r') as fh:
            group_meta = json.load(fh)

    # ── Channel names from OME-Zarr metadata ───────────────────────────────
    try:
        ch_defs = group_meta['attributes']['ome']['omero']['channels']
        metal_names = [str(c.get('label', f'Channel {i}')) for i, c in enumerate(ch_defs)]
    except (KeyError, TypeError):
        metal_names = []

    # ── Enrich names with protein labels from MCD panel ────────────────────
    panel = _panel if _panel is not None else _parse_mcd_panel(sdata_path)
    if panel:
        channel_names = [
            f"{m} / {panel[m]}" if m in panel else m
            for m in metal_names
        ]
    else:
        channel_names = metal_names

    # ── Multiscale paths & downsamples ──────────────────────────────────────
    try:
        multiscales = group_meta['attributes']['ome']['multiscales'][0]
        datasets = multiscales['datasets']  # list of {path, coordinateTransformations}
    except (KeyError, TypeError, IndexError):
        # Fallback: list numeric sub-directories as levels
        datasets = [
            {'path': str(d.name), 'coordinateTransformations': [{'type': 'scale', 'scale': [1.0, 1.0, 1.0]}]}
            for d in sorted(img_root.iterdir())
            if d.is_dir() and d.name.isdigit() and (d / 'zarr.json').exists()
        ]

    # ── Build LevelInfo list ────────────────────────────────────────────────
    levels: list[LevelInfo] = []
    base_shape = None
    base_x = None

    for i, ds_entry in enumerate(datasets):
        lvl_path = img_root / ds_entry['path']
        arr_meta_path = lvl_path / 'zarr.json'
        level_name = str(ds_entry['path']).strip('/')
        arr_meta = (_metadata or {}).get(f'images/{image_name}/{level_name}')
        if arr_meta is None:
            if not arr_meta_path.exists():
                continue
            with open(arr_meta_path, 'r') as fh:
                arr_meta = json.load(fh)

        shape = tuple(arr_meta['shape'])  # (C, Y, X)
        if len(shape) != 3:
            continue

        if base_shape is None:
            base_shape = shape
            base_x = shape[2]

        lx = shape[2]
        downsample = base_x / lx if lx > 0 else 1.0

        z_arr = ZarrV3Array(lvl_path, arr_meta)

        levels.append(LevelInfo(
            index=i,
            shape=shape,
            downsample=downsample,
            _pages=[],           # not used for SpatialData
            _zarr=z_arr,
        ))

    if not levels:
        raise RuntimeError(f"No valid pyramid levels found in {img_root}")

    if not channel_names:
        channel_names = [f'Channel {i}' for i in range(base_shape[0])]
    elif len(channel_names) != base_shape[0]:
        # Trim or pad if metadata is inconsistent with actual array size
        n = base_shape[0]
        channel_names = (channel_names + [f'Channel {i}' for i in range(n)])[:n]

    img = ImageData(
        path=sdata_path,
        is_rgb=False,
        channel_names=channel_names,
        levels=levels,
        dtype=levels[0]._zarr.dtype,
        base_shape=base_shape,   # (C, Y, X)
        axes='CYX',
        _tif=None,
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
    Supports both the OME-TIFF backend (tifffile + zarr) and the
    SpatialData backend (ZarrV3Array).
    """
    lvl = img.levels[level_idx]

    # 1. Check full-level cache (usually for overview levels)
    if lvl._cache is not None:
        return _slice_array(lvl._cache, img.axes, img.is_rgb, channel, y_slice, x_slice)

    # 2. SpatialData fast-path: ZarrV3Array handles chunked reads directly.
    #    Skip the tifffile level-cache fill (img._tif is None for SpatialData).
    if isinstance(lvl._zarr, ZarrV3Array):
        return lvl._zarr[channel, y_slice, x_slice]

    # 3. If the level is small enough, cache it for future calls (OME-TIFF only)
    n_pixels = 1
    for dim in lvl.shape:
        n_pixels *= dim

    if n_pixels * img.dtype.itemsize < 128 * 1024 * 1024 and img._tif is not None:
        lvl._cache = img._tif.series[0].levels[level_idx].asarray()
        return _slice_array(lvl._cache, img.axes, img.is_rgb, channel, y_slice, x_slice)

    # 4. True lazy access via tifffile-backed zarr store (OME-TIFF only)
    if lvl._zarr is not None:
        try:
            return _slice_array(lvl._zarr, img.axes, img.is_rgb, channel, y_slice, x_slice)
        except Exception as exc:
            print(f"[Opal] Zarr read fallback: {exc}")
            pass  # fallback to full-page read

    # 5. Fallback: full-page read (very slow for large compressed images)
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
        # Allow up to 25% sub-sampling before dropping to a coarser level to load faster.
        # This matches QuPath's behavior of favoring render speed during interactions.
        if 1.0 / lvl.downsample >= screen_pixels_per_image_pixel * 0.75:
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
