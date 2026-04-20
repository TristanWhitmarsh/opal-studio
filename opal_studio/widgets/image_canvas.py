"""
Central image canvas — renders the composited OME-TIFF with zoom and pan.

Zoom behaviour mirrors QuPath: scroll wheel scales around the mouse
cursor position so the image point under the cursor stays fixed.

Lazy loading: a low-res overview is shown immediately, and the full-res
tiles for the current viewport are loaded in a background thread and
swapped in when ready.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PySide6.QtCore import (
    Qt, QPointF, QRectF, QThread, Signal, QObject, Slot, QTimer,
)
from PySide6.QtGui import QImage, QPainter, QWheelEvent, QMouseEvent, QColor, QPolygonF
from PySide6.QtWidgets import QWidget, QSizePolicy

from opal_studio.channel_model import ChannelListModel
from opal_studio.image_loader import ImageData, best_level_for_zoom, get_tile, _get_yx
from opal_studio.image_renderer import render_viewport


# ======================================================================
# Background tile loader
# ======================================================================

class _TileRequest:
    """Value-object describing what the worker should render."""
    def __init__(self, img, channels, level, vy, vx, out_h, out_w, seq, viewport, brightness):
        self.img = img
        self.channels = channels
        self.level = level
        self.vy = vy
        self.vx = vx
        self.out_h = out_h
        self.out_w = out_w
        self.seq = seq
        self.viewport = QRectF(viewport)
        self.brightness = brightness


class _TileWorker(QObject):
    """Runs render_viewport in a background thread."""
    tile_ready = Signal(QImage, int, QRectF)  # image, sequence, viewport
    request = Signal(object, int)      # request, latest_seq

    def __init__(self):
        super().__init__()
        self.request.connect(self._process)
        self._latest_seq = -1

    @Slot(object, int)
    def _process(self, req: _TileRequest, latest_seq: int):
        self._latest_seq = max(self._latest_seq, latest_seq)
        
        if req.seq < self._latest_seq:
            return # skip stale from queue
        try:
            #import time
            #start = time.perf_counter()
            qimg = render_viewport(
                req.img, req.channels, req.level,
                req.vy, req.vx, req.out_h, req.out_w,
                req.brightness
            )
            #elapsed = time.perf_counter() - start
            #print(f"[Opal Studio] Rendered sequence {req.seq}: {len(req.channels)} channels in {elapsed:.4f}s")
            self.tile_ready.emit(qimg, req.seq, req.viewport)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[Opal Studio] tile render error: {exc}")


# ======================================================================
# ImageCanvas widget
# ======================================================================

class ImageCanvas(QWidget):
    """
    Pannable / zoomable image viewport with lazy pyramid loading.
    """

    ZOOM_FACTOR = 1.15  # per wheel step
    pixelHovered = Signal(int, int)

    def __init__(self, channel_model: ChannelListModel, parent=None):
        super().__init__(parent)
        self._model = channel_model
        self._img: ImageData | None = None

        # Viewport: the portion of the *base-resolution* image that is
        # currently visible, expressed as a floating-point rect.
        self._viewport = QRectF(0, 0, 1, 1)

        # Cached images
        self._display_image: QImage | None = None  # what is currently painted
        self._overview: QImage | None = None  # low-res fallback

        # Background thread for rendering
        self._seq = 0
        self._pending_seq = -1
        self._thread = QThread(self)
        self._worker = _TileWorker()
        self._worker.moveToThread(self._thread)
        self._worker.tile_ready.connect(self._on_tile_ready, Qt.ConnectionType.QueuedConnection)
        self._display_viewport = QRectF()
        self._thread.start()

        # Debounce timer for render requests
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(30)  # ms
        self._render_timer.timeout.connect(self._do_request_render)

        # Pan state
        self._panning = False
        self._pan_start = QPointF()
        self._viewport_at_pan_start = QRectF()

        # Appearance
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        # Re-render when channels change
        self._model.channels_changed.connect(self._on_channels_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, img: ImageData):
        """Load a new image and fit to window."""
        self._img = img
        axes = img._tif.series[0].axes.upper() if img._tif else ""
        h, w = _get_yx(img.base_shape, axes, img.is_rgb)
        self._viewport = QRectF(0, 0, w, h)
        self._display_image = None
        self._display_viewport = QRectF()
        self._overview = None
        self._fit_viewport()
        self._load_overview()
        self._request_render()

    def _on_channels_changed(self):
        """Update overview cached image because visibility/colors changed, then request hi-res render."""
        self._load_overview()
        self._request_render()
        self.update()

    # ------------------------------------------------------------------
    # Viewport helpers
    # ------------------------------------------------------------------

    def _fit_viewport(self):
        """Reset viewport to fit entire image in the widget."""
        if not self._img:
            return
        axes = self._img._tif.series[0].axes.upper() if self._img._tif else ""
        ih, iw = _get_yx(self._img.base_shape, axes, self._img.is_rgb)
        ww, wh = self.width(), self.height()
        if ww <= 0 or wh <= 0:
            return
        scale = min(ww / iw, wh / ih)
        vw = ww / scale
        vh = wh / scale
        cx, cy = iw / 2, ih / 2
        self._viewport = QRectF(cx - vw / 2, cy - vh / 2, vw, vh)

    def _screen_pixels_per_image_pixel(self) -> float:
        if self._viewport.width() <= 0:
            return 1.0
        return self.width() / self._viewport.width()

    # ------------------------------------------------------------------
    # Low-res overview
    # ------------------------------------------------------------------

    def _load_overview(self):
        """Load the coarsest pyramid level immediately (runs on main thread)."""
        if not self._img or not self._img.levels:
            return
        coarsest = self._img.levels[-1]
        axes = self._img._tif.series[0].axes.upper() if self._img._tif else ""
        ch, cw = _get_yx(coarsest.shape, axes, self._img.is_rgb)
        channels = self._model.visible_channels()
        if not channels and not self._img.is_rgb:
            self._overview = None
            return
        try:
            self._overview = render_viewport(
                self._img, channels, coarsest.index,
                slice(0, ch), slice(0, cw),
                ch, cw,
                self._model.brightness
            )
        except Exception as exc:
            print(f"[Opal Studio] overview render error: {exc}")
            self._overview = None
        self.update()

    # ------------------------------------------------------------------
    # Render requests (debounced + threaded)
    # ------------------------------------------------------------------

    def _request_render(self):
        """Schedule a render (debounced)."""
        self._render_timer.start()

    def _do_request_render(self):
        if not self._img:
            return

        # As per user request: always prioritize the highest resolution.
        # We no longer cap the pixel count because the lazy-loader handles it.
        level_idx = 0
        lvl = self._img.levels[level_idx]
        ds = lvl.downsample
        
        lvl_h, lvl_w = _get_yx(lvl.shape, self._img.axes, self._img.is_rgb)

        y0 = max(0, int(self._viewport.top() / ds))
        y1 = min(lvl_h, int(math.ceil(self._viewport.bottom() / ds)))
        x0 = max(0, int(self._viewport.left() / ds))
        x1 = min(lvl_w, int(math.ceil(self._viewport.right() / ds)))

        if y1 <= y0 or x1 <= x0:
            return

        channels = self._model.visible_channels()
        if not channels and not self._img.is_rgb:
            # Nothing visible — show black
            self._display_image = None
            self.update()
            return

        self._seq += 1
        # The tile we are requesting covers exactly these integer level coordinates:
        tile_vpt = QRectF(x0 * ds, y0 * ds, (x1 - x0) * ds, (y1 - y0) * ds)
        
        req = _TileRequest(
            self._img, channels, level_idx,
            slice(y0, y1), slice(x0, x1),
            y1 - y0, x1 - x0, self._seq,
            tile_vpt, self._model.brightness
        )
        self._pending_seq = self._seq

        # Emit signal to worker thread — Qt delivers it in the worker's event loop
        self._worker.request.emit(req, self._seq)

        # While waiting, repaint with whatever we have (overview or last hi-res)
        self.update()

    @Slot(QImage, int, QRectF)
    def _on_tile_ready(self, qimg: QImage, seq: int, vpoint: QRectF):
        if seq < self._pending_seq:
            return  # stale
        self._display_image = qimg
        self._display_viewport = vpoint
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            # NEAREST NEIGHBOR EVERYTHING — NO SMEARING
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            p.fillRect(self.rect(), QColor(17, 17, 17))
    
            if not self._img:
                p.setPen(QColor(80, 80, 80))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
                p.end()
                return
    
            spp = self._screen_pixels_per_image_pixel()
    
            # 1. Background layer: Overview
            if self._overview:
                ih, iw = _get_yx(self._img.base_shape, self._img.axes, self._img.is_rgb)
                # Map whole image to screen space
                dst_x = -self._viewport.left() * spp
                dst_y = -self._viewport.top() * spp
                dst_w = iw * spp
                dst_h = ih * spp
                p.drawImage(QRectF(dst_x, dst_y, dst_w, dst_h), self._overview)
    
            # 2. Foreground layer: High-res tile
            if self._display_image and not self._display_viewport.isEmpty():
                vpt = self._display_viewport
                dst_x = (vpt.left() - self._viewport.left()) * spp
                dst_y = (vpt.top() - self._viewport.top()) * spp
                dst_w = vpt.width() * spp
                dst_h = vpt.height() * spp
                # Draw on top
                p.drawImage(QRectF(dst_x, dst_y, dst_w, dst_h), self._display_image)
    
            # 3. Vector layer: Mask Contours
            p.save()
            # Map image space to screen space
            p.translate(-self._viewport.left() * spp, -self._viewport.top() * spp)
            p.scale(spp, spp)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            
            import random as py_random
            rng = py_random.Random()

            for ch in self._model.visible_channels():
                if (ch.is_mask or ch.is_cell_mask) and ch.contour_visible and ch.contour_data:
                    pen = p.pen()
                    # A cosmetic pen keeps its width (3px) constant regardless of painter scale.
                    pen.setWidth(3)
                    pen.setCosmetic(True)
                    
                    visible_vpt = self._viewport
                    
                    for lid, data in ch.contour_data.items():
                        bbox = data["bbox"] # [y0, x0, y1, x1]
                        # Fast viewport intersection check
                        if (bbox[2] < visible_vpt.top() or bbox[0] > visible_vpt.bottom() or
                            bbox[3] < visible_vpt.left() or bbox[1] > visible_vpt.right()):
                            continue
                            
                        if ch.is_mask:
                            rng.seed(int(lid))
                            col = QColor.fromRgbF(rng.random(), rng.random(), rng.random())
                        else:
                            col = QColor(0, 255, 0) if lid == 1 else QColor(255, 0, 0)
                            
                        pen.setColor(col)
                        p.setPen(pen)
                        for qpoly in data["polygons"]:
                            p.drawPolyline(qpoly)
            
            p.restore()
            p.end()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR: paintEvent failed: {e}")

    # ------------------------------------------------------------------
    # Zoom (QuPath style — around cursor)
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent):
        if not self._img:
            return

        # Mouse position in widget coords
        mx, my = event.position().x(), event.position().y()

        # Corresponding image coords
        img_x = self._viewport.left() + mx / self._screen_pixels_per_image_pixel()
        img_y = self._viewport.top() + my / self._screen_pixels_per_image_pixel()

        # Zoom
        delta = event.angleDelta().y()
        if delta > 0:
            factor = 1 / self.ZOOM_FACTOR
        elif delta < 0:
            factor = self.ZOOM_FACTOR
        else:
            return

        new_w = self._viewport.width() * factor
        new_h = self._viewport.height() * factor

        # Prevent extreme zoom
        axes = self._img._tif.series[0].axes.upper() if self._img._tif else ""
        ih, iw = _get_yx(self._img.base_shape, self._img.axes, self._img.is_rgb)
        if new_w < 1 or new_h < 1:
            return
        # Removed zoom-out limit so user can zoom out as much as they want

        # Keep image point under cursor fixed
        frac_x = mx / max(self.width(), 1)
        frac_y = my / max(self.height(), 1)
        new_left = img_x - frac_x * new_w
        new_top = img_y - frac_y * new_h

        self._viewport = QRectF(new_left, new_top, new_w, new_h)
        self._request_render()
        self.update()

    # ------------------------------------------------------------------
    # Pan
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._panning = True
            self._pan_start = event.position()
            self._viewport_at_pan_start = QRectF(self._viewport)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        spp = self._screen_pixels_per_image_pixel()
        if spp > 0:
            mx, my = event.position().x(), event.position().y()
            img_x = self._viewport.left() + mx / spp
            img_y = self._viewport.top() + my / spp
            self.pixelHovered.emit(int(img_x), int(img_y))

        if self._panning:
            delta = event.position() - self._pan_start
            dx = -delta.x() / spp
            dy = -delta.y() / spp
            self._viewport = QRectF(
                self._viewport_at_pan_start.left() + dx,
                self._viewport_at_pan_start.top() + dy,
                self._viewport_at_pan_start.width(),
                self._viewport_at_pan_start.height(),
            )
            self._request_render()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._thread.quit()
        self._thread.wait()
        super().closeEvent(event)

    def resizeEvent(self, event):
        old_size = event.oldSize()
        new_size = event.size()

        if self._img and old_size.width() > 0 and old_size.height() > 0:
            # Maintain the current image scale (pixels-on-screen per image-pixel)
            # and keep the center of the view fixed.
            spp = old_size.width() / max(self._viewport.width(), 0.001)
            cx, cy = self._viewport.center().x(), self._viewport.center().y()
            
            new_vw = new_size.width() / spp
            new_vh = new_size.height() / spp
            
            self._viewport = QRectF(cx - new_vw / 2, cy - new_vh / 2, new_vw, new_vh)
            self._request_render()

        super().resizeEvent(event)
