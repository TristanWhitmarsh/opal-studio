"""
CPU image compositing pipeline.

Architecture
------------
render_viewport_tiled()  <- primary entry point for the canvas
  |
  |- Compute viewport in level-space coordinates (pyramid level chosen by canvas)
  |
  |- Per disk channel: get_tile(img, level, ch, y_slice, x_slice)
  |     One page read per channel.  Submitted in parallel to _READ_POOL so
  |     all 40 channels of an IMC image are read concurrently.
  |
  +- Composite all channels into one float32 canvas -> single QImage.
       Returned as one atomic image (no visible tile seams ever).

Layer caching (LayerCache)
--------------------------
Compositing — not disk I/O — dominates once tiles are warm, so a per-viewport
LayerCache lets display-only changes skip the expensive work, treating channels
as reusable layers:
  * each channel's normalised alpha coverage is cached, so changing colour /
    opacity / visibility (or adding one channel) reuses the others' alpha;
  * the summed, pre-brightness intensity image is cached, so a brightness change
    is a single multiply.
The cache is scoped to one (level, viewport-rect); pan/zoom resets it because the
pixels differ.  Passing layer_cache=None disables it (identical output, no reuse).

render_overview()  <- coarsest pyramid level, rendered once per image/channel
                     change.  Used as the always-available background layer.

render_viewport()  <- compatibility shim (explicit-slice API used by the
                     overview helper and any direct callers).

Future backend support
----------------------
Any image source that exposes the ImageData interface (see image_loader.py)
works automatically -- no changes needed here.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage

from opal_studio.channel_model import Channel
from opal_studio.image_loader import (
    ImageData, TileCache, ZarrV3Array, get_tile, _get_yx,
)

import threading

import tifffile as _tifffile

# Thread pool for parallel channel reads.
# I/O (page.asarray) releases the GIL so we get real wall-clock concurrency.
# 6 workers matches typical disk/SSD queue depth.
_READ_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="opal-reader")

# Per-thread TiffFile cache.
# Each worker thread opens its OWN TiffFile (and therefore its own file
# handle).  This is the key to thread-safety: tifffile's TiffPage.asarray()
# seeks and reads through the parent TiffFile's shared filehandle.  Two
# threads calling asarray() on pages that share a filehandle will see each
# other's seek positions, causing corrupted compressed data and ZSTD errors.
# By giving each thread its own TiffFile, all seeks are independent.
_tif_local = threading.local()
_zarr_local = threading.local()


def _get_thread_tif(img: "ImageData") -> "_tifffile.TiffFile":
    """Return a thread-local TiffFile opened from img.path."""
    cache = getattr(_tif_local, "tifs", None)
    if cache is None:
        _tif_local.tifs = {}
    key = str(img.path)
    if key not in _tif_local.tifs:
        _tif_local.tifs[key] = _tifffile.TiffFile(key)
    return _tif_local.tifs[key]


def _get_thread_zarr(img: "ImageData", level_idx: int):
    """Return a thread-local zarr store opened from img.path for a specific level."""
    cache = getattr(_zarr_local, "stores", None)
    if cache is None:
        _zarr_local.stores = {}
    key = f"{img.path}_{level_idx}"
    if key not in _zarr_local.stores:
        import zarr
        tif = _get_thread_tif(img)
        z_store = tif.series[0].levels[level_idx].aszarr()
        z_arr = zarr.open(z_store, mode='r')
        if isinstance(z_arr, zarr.hierarchy.Group):
            if '0' in z_arr:
                z_arr = z_arr['0']
        _zarr_local.stores[key] = z_arr
    return _zarr_local.stores[key]


# ─────────────────────────────────────────────────────────────────────────────
# Compositing layer order
# ─────────────────────────────────────────────────────────────────────────────

class LayerCache:
    """
    Per-viewport cache of composited layers, enabling QuPath-style "render as
    layers" so display-only changes don't recompute the whole image.

    Two levels of caching, both scoped to a single (level, viewport-rect):

      * ``alpha``  — each intensity channel's normalised 0-1 coverage array.
                     Reused when only colour / opacity / brightness / channel
                     visibility change (alpha depends solely on the pixels and
                     the channel's data-range + display-range).
      * ``intensity_sum`` — the additive sum of every visible intensity
                     channel's *coloured* contribution, **before** the global
                     brightness scale.  Reused across brightness changes, making
                     a brightness drag a single multiply.

    Any change of viewport or pyramid level resets the cache (the pixels are
    different).  Only the render worker thread touches an instance; the canvas
    swaps in a fresh LayerCache when a new image loads, so there is never
    cross-thread mutation of a live cache.
    """

    __slots__ = ("_key", "_alpha", "intensity_sum", "intensity_sig")

    def __init__(self):
        self._key = None
        self._alpha: Dict[int, tuple] = {}   # id(ch) -> (params, ndarray)
        self.intensity_sum: Optional[np.ndarray] = None
        self.intensity_sig = None

    def ensure(self, key: tuple) -> None:
        """Reset all cached data if the viewport / level changed."""
        if key != self._key:
            self._key = key
            self._alpha.clear()
            self.intensity_sum = None
            self.intensity_sig = None

    @staticmethod
    def channel_params(ch: "Channel") -> tuple:
        """Everything that affects a channel's alpha coverage array."""
        proc = bool(ch.is_processed and ch.processed_data is not None)
        return (
            round(float(ch.data_min), 6), round(float(ch.data_max), 6),
            round(float(ch.range_min), 6), round(float(ch.range_max), 6),
            proc, id(ch.processed_data) if proc else 0,
        )

    def get_alpha(self, ch: "Channel") -> Optional[np.ndarray]:
        entry = self._alpha.get(id(ch))
        if entry is not None and entry[0] == self.channel_params(ch):
            return entry[1]
        return None

    def put_alpha(self, ch: "Channel", alpha: np.ndarray) -> None:
        self._alpha[id(ch)] = (self.channel_params(ch), alpha)

    def prune(self, keep_ids: set) -> None:
        for cid in [c for c in self._alpha if c not in keep_ids]:
            del self._alpha[cid]


def _channel_alpha(raw: np.ndarray, ch: "Channel") -> np.ndarray:
    """Normalise a raw channel slice to a 0-1 float32 coverage (alpha) array.

    Mirrors the intensity branch of :func:`_composite_channel` exactly so cached
    and freshly-composited frames are pixel-identical.
    """
    dmin, dmax = ch.data_min, ch.data_max
    if dmax > dmin:
        alpha = np.clip((raw - dmin) / (dmax - dmin), 0.0, 1.0)
    else:
        alpha = np.zeros_like(raw, dtype=np.float32)

    rng_w = ch.range_max - ch.range_min
    if rng_w > 0:
        alpha = np.clip((alpha - ch.range_min) / rng_w, 0.0, 1.0)
    else:
        alpha = (alpha >= ch.range_max).astype(np.float32)
    return np.ascontiguousarray(alpha, dtype=np.float32)


def _accumulate_colored(dst: np.ndarray, alpha: np.ndarray, col: np.ndarray) -> None:
    """dst += alpha[...,None] * col, done per-plane to avoid a HxWx3 temporary.

    The broadcast form allocates a full HxWx3 float array per channel; summing
    into each colour plane separately is ~3x faster on typical viewports.
    """
    if col[0]:
        dst[..., 0] += alpha * col[0]
    if col[1]:
        dst[..., 1] += alpha * col[1]
    if col[2]:
        dst[..., 2] += alpha * col[2]


def _composite_order(ch: "Channel") -> int:
    """
    Return a priority integer used to sort channels before compositing.

    Layers (bottom → top):
      0 – raw intensity channels (disk / zarr reads)
      1 – processed / merged channels (in-memory computed data)
      2 – segmentation masks  (is_mask)
      3 – cell pos/neg masks  (is_cell_mask)
      4 – phenotype type masks (is_type_mask)

    Python's sorted() is stable, so channels with the same priority are
    painted in their original list order.
    """
    if ch.is_type_mask:
        return 4
    if ch.is_cell_mask:
        return 3
    if ch.is_mask:
        return 2
    if ch.is_processed and ch.processed_data is not None:
        return 1
    return 0



def render_viewport_tiled(
    cache: TileCache,       # kept in API for future tile-prefetch use
    img: ImageData,
    channels: List[Channel],
    level_idx: int,
    viewport: QRectF,       # in base-resolution image space
    brightness: float = 1.0,
    tile_size: int = 512,   # reserved for future use
    layer_cache: Optional["LayerCache"] = None,
) -> tuple:
    """
    Composite the visible viewport into one QImage.

    Returns (QImage, actual_rect) where actual_rect is the base-image-space
    rectangle the image actually covers.  This differs from the requested
    viewport by up to 'downsample' pixels due to integer rounding in level-
    space coordinates, so callers MUST use actual_rect (not viewport) when
    positioning the returned image on screen.

    Disk channels are read in parallel (one page.asarray() per channel).
    Processed / mask channels are sliced from their in-memory arrays.
    Result is a single image -- atomic swap, no visible tile seams.
    """
    if img.is_rgb:
        return _render_viewport_rgb(cache, img, level_idx, viewport, tile_size)
    return _render_viewport_multichannel(
        cache, img, channels, level_idx, viewport, brightness, tile_size,
        layer_cache,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Overview helper
# ─────────────────────────────────────────────────────────────────────────────

def render_overview(
    img: ImageData,
    channels: List[Channel],
    brightness: float = 1.0,
) -> Optional[QImage]:
    """Render the entire coarsest pyramid level for the background overview."""
    if not img or not img.levels:
        return None
    coarsest = img.levels[-1]
    ch_h, ch_w = _get_yx(coarsest.shape, img.axes, img.is_rgb)
    try:
        return render_viewport(
            img, channels, coarsest.index,
            slice(0, ch_h), slice(0, ch_w),
            ch_h, ch_w,
            brightness,
        )
    except Exception as exc:
        print(f"[Opal] overview render error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility shim -- explicit-slice API
# ─────────────────────────────────────────────────────────────────────────────

def render_viewport(
    img: ImageData,
    channels: List[Channel],
    level_idx: int,
    viewport_y: slice,
    viewport_x: slice,
    output_h: int,
    output_w: int,
    brightness: float = 1.0,
) -> QImage:
    """Compatibility shim: render an explicit slice region of one level."""
    if img.is_rgb:
        return _render_rgb(img, level_idx, viewport_y, viewport_x)
    return _render_multichannel(
        img, channels, level_idx, viewport_y, viewport_x,
        output_h, output_w, brightness,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RGB paths
# ─────────────────────────────────────────────────────────────────────────────

def _render_viewport_rgb(cache: TileCache, img: ImageData, level_idx: int, viewport: QRectF, tile_size: int) -> tuple:
    lvl = img.levels[level_idx]
    ds = lvl.downsample
    lh, lw = _get_yx(lvl.shape, img.axes, img.is_rgb)
    lv_y0 = max(0, int(viewport.top()    / ds))
    lv_y1 = min(lh, int(math.ceil(viewport.bottom() / ds)))
    lv_x0 = max(0, int(viewport.left()   / ds))
    lv_x1 = min(lw, int(math.ceil(viewport.right()  / ds)))
    actual_rect = QRectF(lv_x0 * ds, lv_y0 * ds,
                         (lv_x1 - lv_x0) * ds, (lv_y1 - lv_y0) * ds)
    if lv_y1 <= lv_y0 or lv_x1 <= lv_x0:
        return _blank_qimage(1, 1), actual_rect
    tile = _read_channel_slice(cache, img, level_idx, None, slice(lv_y0, lv_y1), slice(lv_x0, lv_x1), tile_size)
    return _rgb_array_to_qimage(_to_uint8(tile)), actual_rect


def _render_rgb(img, level_idx, vy, vx) -> QImage:
    tile = get_tile(img, level_idx, None, vy, vx)
    return _rgb_array_to_qimage(_to_uint8(tile))


def _rgb_array_to_qimage(tile: np.ndarray) -> QImage:
    h, w = tile.shape[:2]
    if tile.ndim == 2:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = tile[..., None]
        rgba[..., 3] = 255
    elif tile.shape[2] == 3:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = tile
        rgba[..., 3] = 255
    else:
        rgba = tile
    return _ndarray_to_qimage(rgba)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-channel -- primary path
# ─────────────────────────────────────────────────────────────────────────────

def _render_viewport_multichannel(
    cache: TileCache,
    img: ImageData,
    channels: List[Channel],
    level_idx: int,
    viewport: QRectF,
    brightness: float,
    tile_size: int,
    layer_cache: Optional["LayerCache"] = None,
) -> tuple:
    lvl = img.levels[level_idx]
    ds = lvl.downsample
    lh, lw = _get_yx(lvl.shape, img.axes, img.is_rgb)

    # Viewport in level-space pixel coordinates (integer-snapped)
    lv_y0 = max(0, int(viewport.top()    / ds))
    lv_y1 = min(lh, int(math.ceil(viewport.bottom() / ds)))
    lv_x0 = max(0, int(viewport.left()   / ds))
    lv_x1 = min(lw, int(math.ceil(viewport.right()  / ds)))

    # The image will cover exactly this rectangle in base-image space.
    # Use this (not the original floating-point viewport) for on-screen placement.
    actual_rect = QRectF(lv_x0 * ds, lv_y0 * ds,
                         (lv_x1 - lv_x0) * ds, (lv_y1 - lv_y0) * ds)

    height = lv_y1 - lv_y0
    width  = lv_x1 - lv_x0
    if not channels or height <= 0 or width <= 0:
        return _blank_qimage(1, 1), actual_rect

    # Intensity (additive) channels vs. mask overlays (alpha-blended on top).
    intensity_channels = [c for c in channels if not c.is_mask
                          and not c.is_cell_mask and not c.is_type_mask
                          and not getattr(c, 'is_region', False)]
    scale = brightness / len(intensity_channels) if intensity_channels else brightness

    # Scope the layer cache to this exact viewport / level.  A mismatch resets it.
    if layer_cache is not None:
        layer_cache.ensure((level_idx, lv_y0, lv_y1, lv_x0, lv_x1))

    # ── Intensity layer sum (brightness-independent, cached) ─────────────────
    # Signature captures everything that changes the summed intensity image
    # *except* the global brightness scale, so a brightness drag reuses the sum.
    sig = tuple(
        (id(c), c.color.rgb(), round(float(c.alpha), 4)) + LayerCache.channel_params(c)
        for c in intensity_channels
    )

    if (layer_cache is not None and layer_cache.intensity_sum is not None
            and layer_cache.intensity_sig == sig):
        intensity_sum = layer_cache.intensity_sum
    else:
        intensity_sum = _build_intensity_sum(
            cache, img, level_idx, intensity_channels, layer_cache,
            slice(lv_y0, lv_y1), slice(lv_x0, lv_x1), ds, height, width, tile_size,
        )
        if layer_cache is not None:
            layer_cache.intensity_sum = intensity_sum
            layer_cache.intensity_sig = sig
            layer_cache.prune({id(c) for c in intensity_channels})

    # Apply the global brightness scale onto a private copy (cache stays clean).
    canvas = intensity_sum * scale

    # ── Mask / cell / type overlays (alpha-blended, painted on top) ──────────
    mask_channels = [c for c in channels
                     if (c.is_mask or c.is_cell_mask or c.is_type_mask)
                     and c.visible and c.mask_data is not None]
    if mask_channels:
        by0 = int(lv_y0 * ds);  by1 = int(lv_y1 * ds)
        bx0 = int(lv_x0 * ds);  bx1 = int(lv_x1 * ds)
        for ch in sorted(mask_channels, key=_composite_order):
            raw = ch.mask_data[by0:by1, bx0:bx1].astype(np.float32)
            if raw.shape[:2] != (height, width):
                raw = _fast_resize(raw, height, width)
            _composite_channel(canvas, raw, ch, scale)

    np.clip(canvas, 0.0, 1.0, out=canvas)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = (canvas * 255).astype(np.uint8)
    rgba[..., 3] = 255
    return _ndarray_to_qimage(rgba), actual_rect


def _build_intensity_sum(
    cache: TileCache,
    img: ImageData,
    level_idx: int,
    intensity_channels: List[Channel],
    layer_cache: Optional["LayerCache"],
    y_sl: slice,
    x_sl: slice,
    ds: float,
    height: int,
    width: int,
    tile_size: int,
) -> np.ndarray:
    """Sum every intensity channel's coloured (pre-brightness) contribution.

    Each channel's normalised alpha is pulled from the layer cache when its
    display range is unchanged; only channels whose alpha is missing get a disk
    read.  So adding one channel reads one channel, not all of them.
    """
    # Which channels still need a raw disk read (no cached alpha, on-disk source)?
    need_read: List[Channel] = []
    for ch in intensity_channels:
        cached_alpha = layer_cache.get_alpha(ch) if layer_cache is not None else None
        is_processed = ch.is_processed and ch.processed_data is not None
        if cached_alpha is None and not is_processed:
            need_read.append(ch)

    # ── Parallel fetch: one read per channel that needs it.  I/O releases the
    #    GIL so many-channel reads overlap on the worker pool. ────────────────
    channel_arrays: Dict[int, np.ndarray] = {}
    if need_read:
        future_to_ch = {
            _READ_POOL.submit(
                _read_channel_slice,
                cache, img, level_idx, ch.index, y_sl, x_sl, tile_size,
            ): ch
            for ch in need_read
        }
        for fut in as_completed(future_to_ch):
            ch = future_to_ch[fut]
            try:
                channel_arrays[id(ch)] = fut.result().astype(np.float32)
            except Exception as exc:
                print(f"[Opal] channel read error ({ch.name}): {exc}")
                channel_arrays[id(ch)] = np.zeros((height, width), dtype=np.float32)

    intensity_sum = np.zeros((height, width, 3), dtype=np.float32)
    for ch in intensity_channels:
        alpha = layer_cache.get_alpha(ch) if layer_cache is not None else None
        if alpha is None:
            if ch.is_processed and ch.processed_data is not None:
                by0 = int(y_sl.start * ds);  by1 = int(y_sl.stop * ds)
                bx0 = int(x_sl.start * ds);  bx1 = int(x_sl.stop * ds)
                raw = ch.processed_data[by0:by1, bx0:bx1].astype(np.float32)
            else:
                raw = channel_arrays.get(id(ch))
                if raw is None:
                    continue
            if raw.shape[:2] != (height, width):
                raw = _fast_resize(raw, height, width)
            alpha = _channel_alpha(raw, ch)
            if layer_cache is not None:
                layer_cache.put_alpha(ch, alpha)

        col = np.array([ch.color.redF(), ch.color.greenF(), ch.color.blueF()],
                       dtype=np.float32) * ch.alpha
        _accumulate_colored(intensity_sum, alpha, col)

    return intensity_sum


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility: explicit-slice multichannel (overview + shim)
# ─────────────────────────────────────────────────────────────────────────────

def _render_multichannel(
    img: ImageData,
    channels: List[Channel],
    level_idx: int,
    vy: slice,
    vx: slice,
    out_h: int,
    out_w: int,
    brightness: float = 1.0,
) -> QImage:
    if not channels:
        return _blank_qimage(max(1, out_h), max(1, out_w))

    intensity_channels = [c for c in channels if not c.is_mask
                          and not c.is_cell_mask and not c.is_type_mask
                          and not getattr(c, 'is_region', False)]
    scale = brightness / len(intensity_channels) if intensity_channels else brightness

    h, w = 0, 0
    canvas = None

    for ch in sorted(channels, key=_composite_order):
        if getattr(ch, 'is_region', False):
            continue
        if (ch.is_mask or ch.is_cell_mask or ch.is_type_mask) and ch.mask_data is not None:
            ds = img.levels[level_idx].downsample
            sy = slice(int(vy.start * ds), int(vy.stop * ds))
            sx = slice(int(vx.start * ds), int(vx.stop * ds))
            raw = ch.mask_data[sy, sx].astype(np.float32)
        elif ch.is_processed and ch.processed_data is not None:
            ds = img.levels[level_idx].downsample
            sy = slice(int(vy.start * ds), int(vy.stop * ds))
            sx = slice(int(vx.start * ds), int(vx.stop * ds))
            raw = ch.processed_data[sy, sx].astype(np.float32)
        else:
            raw = get_tile(img, level_idx, ch.index, vy, vx).astype(np.float32)

        if canvas is None:
            h, w = raw.shape[:2]
            canvas = np.zeros((h, w, 3), dtype=np.float32)

        if raw.shape[:2] != (h, w):
            raw = _fast_resize(raw, h, w)

        _composite_channel(canvas, raw, ch, scale)

    if canvas is None:
        canvas = np.zeros((max(1, out_h), max(1, out_w), 3), dtype=np.float32)

    np.clip(canvas, 0.0, 1.0, out=canvas)
    rgba = np.zeros((canvas.shape[0], canvas.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = (canvas * 255).astype(np.uint8)
    rgba[..., 3] = 255
    return _ndarray_to_qimage(rgba)


# ─────────────────────────────────────────────────────────────────────────────
# Channel read helper (called from thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def _read_channel_slice(
    cache: TileCache,
    img: ImageData,
    level_idx: int,
    channel: Optional[int],
    y_sl: slice,
    x_sl: slice,
    tile_size: int,
) -> np.ndarray:
    """
    Assemble the viewport from cached tiles.
    Missing tiles are read via a thread-local TiffFile instance to avoid collisions.
    """
    # Fast path: use the in-RAM level cache -- numpy reads are thread-safe.
    lvl = img.levels[level_idx]
    if lvl._cache is not None:
        return get_tile(img, level_idx, channel, y_sl, x_sl)  # slices cache

    vh = y_sl.stop - y_sl.start
    vw = x_sl.stop - x_sl.start
    
    if img.is_rgb:
        out = np.zeros((vh, vw, 3), dtype=img.dtype)
    else:
        out = np.zeros((vh, vw), dtype=img.dtype)

    col_start = x_sl.start // tile_size
    col_end   = (x_sl.stop - 1) // tile_size
    row_start = y_sl.start // tile_size
    row_end   = (y_sl.stop - 1) // tile_size

    lh, lw = _get_yx(lvl.shape, img.axes, img.is_rgb)
    z_arr = None

    # SpatialData fast-path: ZarrV3Array is already thread-safe (opens files
    # per-call) so we bypass the tifffile-based _get_thread_zarr entirely.
    if isinstance(lvl._zarr, ZarrV3Array):
        for tr in range(row_start, row_end + 1):
            for tc in range(col_start, col_end + 1):
                key = (level_idx, channel, tr, tc, tile_size)
                tile = cache.get(key)

                if tile is None:
                    ty0 = tr * tile_size
                    tx0 = tc * tile_size
                    ty1 = min(ty0 + tile_size, lh)
                    tx1 = min(tx0 + tile_size, lw)
                    tile = lvl._zarr[channel, slice(ty0, ty1), slice(tx0, tx1)]
                    tile = np.ascontiguousarray(tile)
                    cache.put(key, tile)

                ty0 = tr * tile_size
                tx0 = tc * tile_size
                dy = max(0, ty0 - y_sl.start)
                dx = max(0, tx0 - x_sl.start)
                sy = max(0, y_sl.start - ty0)
                sx = max(0, x_sl.start - tx0)
                h = min(tile.shape[0] - sy, vh - dy)
                w = min(tile.shape[1] - sx, vw - dx)
                if h > 0 and w > 0:
                    out[dy:dy+h, dx:dx+w] = tile[sy:sy+h, sx:sx+w]
        return out

    for tr in range(row_start, row_end + 1):
        for tc in range(col_start, col_end + 1):
            key = (level_idx, channel, tr, tc, tile_size)
            tile = cache.get(key)
            
            if tile is None:
                if z_arr is None and not img.is_rgb:
                    try:
                        z_arr = _get_thread_zarr(img, level_idx)
                    except Exception:
                        pass # Fallback to get_tile
                        
                ty0 = tr * tile_size
                tx0 = tc * tile_size
                ty1 = min(ty0 + tile_size, lh)
                tx1 = min(tx0 + tile_size, lw)
                
                if z_arr is not None and not img.is_rgb:
                    try:
                        if len(z_arr.shape) > 2:
                            # typically (C, Y, X)
                            tile = z_arr[channel, ty0:ty1, tx0:tx1]
                        else:
                            tile = z_arr[ty0:ty1, tx0:tx1]
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[Opal] Zarr thread slice error: {e}")
                        tile = get_tile(img, level_idx, channel, slice(ty0, ty1), slice(tx0, tx1))
                else:
                    tile = get_tile(img, level_idx, channel, slice(ty0, ty1), slice(tx0, tx1))
                    
                tile = np.ascontiguousarray(tile)
                cache.put(key, tile)
                
            # Paste tile into output array
            ty0 = tr * tile_size
            tx0 = tc * tile_size
            
            dy = max(0, ty0 - y_sl.start)
            dx = max(0, tx0 - x_sl.start)
            
            sy = max(0, y_sl.start - ty0)
            sx = max(0, x_sl.start - tx0)
            
            h = min(tile.shape[0] - sy, vh - dy)
            w = min(tile.shape[1] - sx, vw - dx)
            
            if h > 0 and w > 0:
                out[dy:dy+h, dx:dx+w] = tile[sy:sy+h, sx:sx+w]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared compositing logic
# ─────────────────────────────────────────────────────────────────────────────

def _composite_channel(
    canvas: np.ndarray,
    raw: np.ndarray,
    ch: Channel,
    scale: float,
) -> None:
    """In-place additive blend of one channel onto the float32 RGB canvas."""

    if ch.is_mask or ch.is_cell_mask or ch.is_type_mask:
        if not ch.visible or ch.mask_data is None:
            return
        labels = raw.astype(np.int32)
        mask_active = labels > 0
        if not np.any(mask_active):
            return
        alpha_mask = ch.range_max

        if ch.is_mask:
            import random as py_random
            rng = py_random.Random()
            unique_ids = np.unique(labels[mask_active])
            max_id = int(np.max(unique_ids))
            if max_id < 2_000_000:
                lut = np.zeros((max_id + 1, 3), dtype=np.float32)
                for lid in unique_ids:
                    rng.seed(int(lid))
                    lut[lid] = [rng.random(), rng.random(), rng.random()]
                canvas[mask_active] = (
                    (1.0 - alpha_mask) * canvas[mask_active]
                    + alpha_mask * lut[labels[mask_active]]
                )
            else:
                for lid in unique_ids:
                    rng.seed(int(lid))
                    col = np.array([rng.random(), rng.random(), rng.random()],
                                   dtype=np.float32)
                    canvas[labels == lid] = (
                        (1.0 - alpha_mask) * canvas[labels == lid] + alpha_mask * col
                    )

        elif ch.is_cell_mask:
            if ch.pos_lut is not None:
                # Use LUT to map unique labels to positivity states
                states = ch.pos_lut[labels]
                m1, m2 = states == 1, states == 2
            else:
                # Fallback to direct state map (0,1,2)
                m1, m2 = labels == 1, labels == 2
                
            if np.any(m1):
                canvas[m1] = ((1.0 - alpha_mask) * canvas[m1]
                              + alpha_mask * np.array([0.0, 1.0, 0.0], np.float32))
            if np.any(m2):
                canvas[m2] = ((1.0 - alpha_mask) * canvas[m2]
                              + alpha_mask * np.array([1.0, 0.0, 0.0], np.float32))

        elif ch.is_type_mask:
            col = np.array([ch.color.redF(), ch.color.greenF(), ch.color.blueF()],
                           dtype=np.float32)
            canvas[mask_active] = (
                (1.0 - alpha_mask) * canvas[mask_active] + alpha_mask * col
            )

    else:
        dmin, dmax = ch.data_min, ch.data_max
        alpha = np.clip((raw - dmin) / (dmax - dmin), 0.0, 1.0) if dmax > dmin \
            else np.zeros_like(raw)

        rng_w = ch.range_max - ch.range_min
        if rng_w > 0:
            alpha = np.clip((alpha - ch.range_min) / rng_w, 0.0, 1.0)
        else:
            alpha = np.where(alpha >= ch.range_max, 1.0, 0.0)

        col = np.array([ch.color.redF(), ch.color.greenF(), ch.color.blueF()],
                       dtype=np.float32)
        col *= scale * ch.alpha
        _accumulate_colored(canvas, alpha, col)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fast_resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize via numpy indexing -- far faster than skimage."""
    if arr.shape[:2] == (h, w):
        return arr
    ys = np.linspace(0, arr.shape[0] - 1, h).astype(np.int32)
    xs = np.linspace(0, arr.shape[1] - 1, w).astype(np.int32)
    return arr[np.ix_(ys, xs)]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
    if info is not None:
        return ((arr.astype(np.float32) / info.max) * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def _ndarray_to_qimage(rgba: np.ndarray) -> QImage:
    h, w, _ = rgba.shape
    rgba = np.ascontiguousarray(rgba)
    qimg = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return qimg.copy()   # own the buffer so it's safe across threads


def _blank_qimage(h: int, w: int) -> QImage:
    data = np.zeros((h, w, 4), dtype=np.uint8)
    data[..., 3] = 255
    return _ndarray_to_qimage(data)
