"""
CPU image compositing — produces the final RGBA QImage for the viewport.

For multi-channel IMC images: additive blending of tinted channels on black.
For RGB H&E images: direct tile read.
"""

from __future__ import annotations

from typing import List

import numpy as np
from PySide6.QtGui import QImage
from skimage.transform import resize

from opal_studio.channel_model import Channel
from opal_studio.image_loader import ImageData, get_tile


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
    """
    Composite the visible viewport into a QImage.
    """
    if img.is_rgb:
        return _render_rgb(img, level_idx, viewport_y, viewport_x, output_h, output_w)
    else:
        return _render_multichannel(img, channels, level_idx, viewport_y, viewport_x, output_h, output_w, brightness)


# ------------------------------------------------------------------
# RGB
# ------------------------------------------------------------------

def _render_rgb(
    img: ImageData,
    level_idx: int,
    vy: slice,
    vx: slice,
    out_h: int,
    out_w: int,
) -> QImage:
    tile = get_tile(img, level_idx, None, vy, vx)
    tile = _to_uint8(tile)

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


# ------------------------------------------------------------------
# Multi-channel (IMC)
# ------------------------------------------------------------------

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
        return _ndarray_to_qimage(np.zeros((max(1, out_h), max(1, out_w), 4), dtype=np.uint8))
        
    # User's brightness formula: brightness * n_channels
    # Only count non-mask channels for brightness scaling
    intensity_channels = [c for c in channels if not c.is_mask]
    scale = brightness * len(intensity_channels) if intensity_channels else brightness
    
    # Establish canvas size from first available source
    h, w = 0, 0
    canvas = None

    for ch in channels:
        # 1. Fetch raw data (either from disk page or resident mask)
        if (ch.is_mask or ch.is_cell_mask) and ch.mask_data is not None:
            # Slicing resident data (always base resolution)
            down = img.levels[level_idx].downsample
            sy = slice(int(vy.start * down), int(vy.stop * down))
            sx = slice(int(vx.start * down), int(vx.stop * down))
            raw = ch.mask_data[sy, sx].astype(np.float32)
        elif ch.is_processed and ch.processed_data is not None:
            down = img.levels[level_idx].downsample
            sy = slice(int(vy.start * down), int(vy.stop * down))
            sx = slice(int(vx.start * down), int(vx.stop * down))
            raw = ch.processed_data[sy, sx].astype(np.float32)
        else:
            raw = get_tile(img, level_idx, ch.index, vy, vx).astype(np.float32)

        # 2. Init canvas on first pass
        if canvas is None:
            h, w = raw.shape[:2]
            canvas = np.zeros((h, w, 3), dtype=np.float32)

        # 3. Handle shape mismatch (pyramid level aliasing)
        if raw.shape[:2] != (h, w):
            raw = resize(raw, (h, w), order=0, preserve_range=True)

        if ch.is_mask:
            # 1. Vectorized color assignment via LUT for speed (instant rendering)
            labels = raw.astype(np.int32)
            mask_active = labels > 0
            if np.any(mask_active):
                import random
                state = random.getstate()
                
                # Identify unique cells in this tile
                unique_ids = np.unique(labels[mask_active])
                
                # Pre-calculate colors for these IDs
                max_id = int(np.max(unique_ids))
                alpha_mask = ch.range_max
                if max_id < 1000000: # Standard label range
                    lut = np.zeros((max_id + 1, 3), dtype=np.float32)
                    for lid in unique_ids:
                        random.seed(int(lid))
                        lut[lid] = [random.random(), random.random(), random.random()]
                    
                    # Blend onto existing canvas
                    canvas[mask_active] = (1.0 - alpha_mask) * canvas[mask_active] + alpha_mask * lut[labels[mask_active]]
                else:
                    # Fallback for extremely sparse/high label IDs
                    for lid in unique_ids:
                        random.seed(int(lid))
                        col = np.array([random.random(), random.random(), random.random()], dtype=np.float32)
                        idx = labels == lid
                        canvas[idx] = (1.0 - alpha_mask) * canvas[idx] + alpha_mask * col
                
                random.setstate(state)
        elif ch.is_cell_mask:
            # Cell positivity rendering: 1 = green, 2 = red
            alpha_mask = ch.range_max
            mask_1 = raw == 1
            mask_2 = raw == 2
            
            if np.any(mask_1):
                color_green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                canvas[mask_1] = (1.0 - alpha_mask) * canvas[mask_1] + alpha_mask * color_green
                
            if np.any(mask_2):
                color_red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                canvas[mask_2] = (1.0 - alpha_mask) * canvas[mask_2] + alpha_mask * color_red
        else:
            # Normal intensity channel rendering
            dmin, dmax = ch.data_min, ch.data_max
            if dmax - dmin > 0:
                alpha = np.clip((raw - dmin) / (dmax - dmin), 0, 1)
            else:
                alpha = np.zeros_like(raw)
            
            # Apply display window
            rng = ch.range_max - ch.range_min
            if rng > 0:
                alpha = np.clip((alpha - ch.range_min) / rng, 0, 1)
            else:
                alpha = np.where(alpha >= ch.range_max, 1.0, 0.0)

            color_vec = np.array([ch.color.redF(), ch.color.greenF(), ch.color.blueF()], dtype=np.float32)
            color_vec *= (scale * ch.alpha)
            
            if np.any(color_vec > 0):
                canvas += alpha[..., None] * color_vec

    # Clamp & convert to RGBA uint8
    if canvas is None:
        canvas = np.zeros((max(1, out_h), max(1, out_w), 3), dtype=np.float32)

    np.clip(canvas, 0.0, 1.0, out=canvas)
    rgba = np.zeros((canvas.shape[0], canvas.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = (canvas * 255).astype(np.uint8)
    rgba[..., 3] = 255

    return _ndarray_to_qimage(rgba)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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
    qimg._numpy_data = rgba
    return qimg
