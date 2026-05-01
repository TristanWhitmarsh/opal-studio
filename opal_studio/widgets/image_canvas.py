"""
Central image canvas — renders the composited OME-TIFF with zoom and pan.

Architecture
------------
One background QThread runs render_viewport_tiled() which composites ALL
visible channels for the full viewport into a single QImage — no visible
tile seams.  The result is swapped in atomically on the main thread.

Layers (painted back-to-front):
  1. Overview   — coarsest pyramid level, rendered once per channel change.
                  Always available as instant background.
  2. Hi-res frame — last completed full-viewport composite. Shown even while
                    a new frame is being computed, so the display never goes
                    blank during panning.  Scaled to match the current viewport
                    (slightly blurry during fast pan, resolves when settled).
  3. Vector contours — drawn in image space via QPainter transforms.

Pyramid level selection
-----------------------
best_level_for_zoom() is called per render so coarser levels are used when
the user is zoomed out — much less data to read and composite.

Tile cache
----------
Decoded channel data lives in a TileCache (see image_loader.py).
render_viewport_tiled() assembles each channel's viewport slice from
512×512 cache tiles so subsequent pans at the same zoom level are served
entirely from RAM.

Debouncing
----------
Wheel and pan events accumulate for 40 ms before triggering a render
request.  While waiting, paintEvent() draws the scaled-up previous frame
so the display tracks the viewport change instantly.

QuPath-parity design notes
--------------------------
• Single atomic swap → no seams.
• Overview always painted first → no blank screen.
• Previous frame shown during loading → no flicker.
• Pyramid level selection → low zoom reads coarse data only.
• Parallel channel reads (6 threads) → ~6× faster for many-channel IMC.
• TileCache → subsequent pans at same zoom reuse decoded data.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import (
    Qt, QPointF, QRectF, QThread, Signal, QObject, Slot, QTimer,
)
from PySide6.QtGui import QImage, QPainter, QWheelEvent, QMouseEvent, QColor
from PySide6.QtWidgets import QWidget, QSizePolicy

from opal_studio.channel_model import ChannelListModel
from opal_studio.image_loader import (
    ImageData, TileCache, best_level_for_zoom, get_tile, _get_yx,
)
from opal_studio.image_renderer import render_viewport_tiled, render_overview

# ── Tunable constants ─────────────────────────────────────────────────────────
TILE_SIZE       = 512   # tile size passed to the renderer / cache key unit
DECODE_CACHE_MB = 512   # TileCache memory budget for decoded channel data
DEBOUNCE_MS     = 40    # ms to wait after last interaction before requesting render


# ──────────────────────────────────────────────────────────────────────────────
# Background render worker
# ──────────────────────────────────────────────────────────────────────────────

class _ViewportRequest:
    """Value object carrying everything needed for one render."""
    __slots__ = (
        "img", "channels", "cache", "level_idx",
        "viewport", "brightness", "seq", "channel_version", "is_progressive",
    )

    def __init__(self, img, channels, cache, level_idx,
                 viewport: QRectF, brightness, seq, channel_version, is_progressive=False):
        self.img = img
        self.channels = channels
        self.cache = cache
        self.level_idx = level_idx
        self.viewport = QRectF(viewport)   # snapshot
        self.brightness = brightness
        self.seq = seq
        self.channel_version = channel_version
        self.is_progressive = is_progressive


class _RenderWorker(QObject):
    """
    Runs render_viewport_tiled() in a dedicated QThread.
    Requests are serialised through this worker's event-loop queue so only
    one render runs at a time.  Stale requests (superseded by a newer seq)
    are dropped before any work is done.
    """
    frame_ready = Signal(QImage, QRectF, int, int, int, bool)
    # (image, viewport_it_covers, seq, channel_version, level_idx, is_progressive)
    overview_ready = Signal(QImage, int)
    # (image, channel_version)

    request = Signal(object, int)  # (_ViewportRequest, latest_seq)
    overview_request = Signal(object, int)  # (_ViewportRequest, latest_ch_ver)

    def __init__(self):
        super().__init__()
        self._latest_seq = -1
        self._latest_overview_ver = -1
        self.request.connect(self._process, Qt.ConnectionType.QueuedConnection)
        self.overview_request.connect(self._process_overview, Qt.ConnectionType.QueuedConnection)

    @Slot(object, int)
    def _process(self, req: _ViewportRequest, latest_seq: int):
        self._latest_seq = max(self._latest_seq, latest_seq)
        if req.seq < self._latest_seq:
            return  # stale — a newer request is already queued
        try:
            qimg, actual_rect = render_viewport_tiled(
                req.cache,
                req.img,
                req.channels,
                req.level_idx,
                req.viewport,
                req.brightness,
                TILE_SIZE,
            )
            # Emit actual_rect (integer-snapped level coords back-projected to
            # base-image space), NOT req.viewport which is floating-point and
            self.frame_ready.emit(qimg, actual_rect, req.seq, req.channel_version, req.level_idx, req.is_progressive)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[Opal] render error: {exc}")

    @Slot(object, int)
    def _process_overview(self, req: _ViewportRequest, latest_ch_ver: int):
        self._latest_overview_ver = max(self._latest_overview_ver, latest_ch_ver)
        if req.channel_version < self._latest_overview_ver:
            return  # stale overview request

        try:
            qimg, _ = render_viewport_tiled(
                req.cache,
                req.img,
                req.channels,
                req.level_idx,
                req.viewport,
                req.brightness,
                TILE_SIZE,
            )
            self.overview_ready.emit(qimg, req.channel_version)
        except Exception as exc:
            print(f"[Opal] overview render error: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# ImageCanvas widget
# ──────────────────────────────────────────────────────────────────────────────

class ImageCanvas(QWidget):
    """Pannable / zoomable image canvas with progressive pyramid loading."""

    ZOOM_FACTOR = 1.15
    pixelHovered = Signal(int, int)

    def __init__(self, channel_model: ChannelListModel, parent=None):
        super().__init__(parent)
        self._model = channel_model
        self._img: Optional[ImageData] = None

        # Viewport: rectangle in base-resolution image space that is visible.
        self._viewport = QRectF(0, 0, 1, 1)

        # Decoded channel data (shared with renderer, thread-safe LRU)
        self._tile_cache = TileCache(max_bytes=DECODE_CACHE_MB * 1024 * 1024)

        # Low-res overview — entire image at coarsest level
        self._overview: Optional[QImage] = None
        # Channel version when the overview was last rendered.
        # We only PAINT the overview if it matches the current channel version
        # so stale overview data never bleeds through (deselected channels,
        # wrong brightness, etc.).
        self._overview_channel_version: int = -1
        # Last completed full-viewport frame (shown while next frame loads)
        self._display_image: Optional[QImage] = None
        self._display_viewport: QRectF = QRectF()  # image-space rect it covers
        self._display_level_idx: int = -1
        self._display_channel_version: int = 0

        # Coarser progressive frame (shown underneath display_image while loading)
        self._progressive_image: Optional[QImage] = None
        self._progressive_viewport: QRectF = QRectF()
        self._progressive_level_idx: int = -1

        # Sequencing: channel version + request seq to discard stale frames
        self._channel_version = 0
        self._seq = 0
        self._pending_seq = -1

        # Worker thread
        self._thread = QThread(self)
        self._worker = _RenderWorker()
        self._worker.moveToThread(self._thread)
        self._worker.frame_ready.connect(self._on_frame_ready,
                                         Qt.ConnectionType.QueuedConnection)
        self._worker.overview_ready.connect(self._on_overview_ready,
                                            Qt.ConnectionType.QueuedConnection)
        self._thread.start()

        # Debounce timer
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(DEBOUNCE_MS)
        self._render_timer.timeout.connect(self._submit_render)

        # Pan state
        self._panning = False
        self._pan_start = QPointF()
        self._viewport_at_pan_start = QRectF()

        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._model.channels_changed.connect(self._on_channels_changed)

        # Ensure the worker thread is cleanly stopped when the application
        # exits, regardless of whether closeEvent is triggered.
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_worker)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_image(self, img: ImageData):
        """Load a new image, clear all caches, fit to window."""
        self._img = img
        self._tile_cache.clear()
        self._display_image = None
        self._display_viewport = QRectF()
        self._display_level_idx = -1
        self._display_channel_version = 0
        self._progressive_image = None
        self._progressive_viewport = QRectF()
        self._progressive_level_idx = -1
        self._overview = None
        self._channel_version = 0
        self._seq = 0
        self._pending_seq = -1

        axes = img._tif.series[0].axes.upper() if img._tif else ""
        h, w = _get_yx(img.base_shape, axes, img.is_rgb)
        self._viewport = QRectF(0, 0, w, h)
        self._fit_viewport()
        self._load_overview()
        self._schedule_render()

    # ── Channel change ────────────────────────────────────────────────────────

    def _on_channels_changed(self):
        """
        Bump the channel version and schedule a new render.
        """
        self._channel_version += 1
        self._schedule_render(immediate_progressive=False)
        self.update()

    # ── Viewport helpers ──────────────────────────────────────────────────────

    def _fit_viewport(self):
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

    def _current_level(self) -> int:
        if not self._img:
            return 0
        return best_level_for_zoom(self._img, self._screen_pixels_per_image_pixel())

    # ── Overview (background fallback) ────────────────────────────────────────

    def _load_overview(self):
        """Initial synchronous overview load."""
        if not self._img or not self._img.levels:
            return
        channels = self._model.visible_channels()
        try:
            self._overview = render_overview(self._img, channels, self._model.brightness)
            self._overview_channel_version = self._channel_version
        except Exception as exc:
            print(f"[Opal] overview error: {exc}")
            self._overview = None
            self._overview_channel_version = -1
        self.update()

    def _request_overview_update(self):
        """Request an asynchronous overview update from the worker thread."""
        if not self._img or not self._img.levels:
            return
        channels = self._model.visible_channels()
        if not channels and not self._img.is_rgb:
            self._overview = None
            self._overview_channel_version = self._channel_version
            return

        level_idx = len(self._img.levels) - 1
        axes = self._img._tif.series[0].axes.upper() if self._img._tif else ""
        h, w = _get_yx(self._img.base_shape, axes, self._img.is_rgb)

        req = _ViewportRequest(
            img=self._img,
            channels=list(channels),
            cache=self._tile_cache,
            level_idx=level_idx,
            viewport=QRectF(0, 0, w, h),
            brightness=self._model.brightness,
            seq=-1,  # Not used for overview
            channel_version=self._channel_version,
            is_progressive=False,
        )
        self._worker.overview_request.emit(req, self._channel_version)

    @Slot(QImage, int)
    def _on_overview_ready(self, qimg: QImage, ch_ver: int):
        """Receive updated overview from worker."""
        if ch_ver != self._channel_version:
            return
        self._overview = qimg
        self._overview_channel_version = ch_ver
        self.update()

    # ── Render request scheduling ─────────────────────────────────────────────

    def _schedule_render(self, immediate_progressive=False):
        """Debounced render request. If immediate_progressive is True, submit a coarse render instantly."""
        if immediate_progressive:
            self._submit_render(progressive=True)
        self._render_timer.start()

    def _submit_render(self, progressive=False):
        """Build and post a render request to the worker thread."""
        if not self._img:
            return

        channels = self._model.visible_channels()
        if not channels and not self._img.is_rgb:
            self._display_image = None
            self._overview = None
            self.update()
            return

        level_idx = self._current_level()
        
        # QuPath-style progressive rendering: if panning/zooming, request a faster 
        # coarse level first to give immediate visual feedback.
        if progressive and level_idx + 1 < len(self._img.levels):
            level_idx = min(level_idx + 2, len(self._img.levels) - 1)

        self._seq += 1
        self._pending_seq = self._seq

        req = _ViewportRequest(
            img=self._img,
            channels=list(channels),  # snapshot
            cache=self._tile_cache,
            level_idx=level_idx,
            viewport=self._viewport,
            brightness=self._model.brightness,
            seq=self._seq,
            channel_version=self._channel_version,
            is_progressive=progressive,
        )
        self._worker.request.emit(req, self._seq)
        
        # Only request an overview update on the final debounced high-res render
        # to prevent queue clogging when dragging sliders.
        if not progressive:
            self._request_overview_update()
            
        # Draw immediately with whatever we have (overview + scaled old frame)
        self.update()

    @Slot(QImage, QRectF, int, int, int, bool)
    def _on_frame_ready(self, qimg: QImage, viewport: QRectF, seq: int, ch_ver: int, level_idx: int, is_progressive: bool):
        """Receive completed frame from worker."""
        if seq < self._pending_seq:
            return   # superseded
        if ch_ver != self._channel_version:
            return   # channel state changed since this was rendered
            
        if is_progressive:
            self._progressive_image = qimg
            self._progressive_viewport = viewport
            self._progressive_level_idx = level_idx
            
            # If our sharp display image is from a different channel version, OR if it's 
            # actually coarser (blurrier) than this progressive image (e.g. huge zoom-in), drop it.
            if (self._display_channel_version != ch_ver or 
                self._display_level_idx > level_idx):
                self._display_image = None
        else:
            self._display_image = qimg
            self._display_viewport = viewport
            self._display_channel_version = ch_ver
            self._display_level_idx = level_idx
            self._progressive_image = None

        self.update()

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            p.fillRect(self.rect(), QColor(17, 17, 17))

            if not self._img:
                p.setPen(QColor(80, 80, 80))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
                p.end()
                return

            spp = self._screen_pixels_per_image_pixel()
            ih, iw = _get_yx(self._img.base_shape, self._img.axes, self._img.is_rgb)

            # ── Layer 1: Overview (background fallback) ───────────────────────
            # Always paint the overview to act as a placeholder, preventing
            # black flickers when zooming out. QuPath-style rendering leaves
            # background data until the hi-res foreground is ready.
            if self._overview:
                dst_x = -self._viewport.left() * spp
                dst_y = -self._viewport.top()  * spp
                # Draw slightly faded if stale, else normal
                alpha = 1.0 if self._overview_channel_version == self._channel_version else 0.6
                if alpha < 1.0:
                    p.setOpacity(alpha)
                p.drawImage(QRectF(dst_x, dst_y, iw * spp, ih * spp), self._overview)
                p.setOpacity(1.0)

            # ── Layer 2: Progressive frame (coarse, fills black edges) ─────
            # Drawn underneath the high-res frame so that any sharp data we
            # ALREADY have is preserved, but any new areas (edges) get filled.
            if self._progressive_image and not self._progressive_viewport.isEmpty():
                vpt = self._progressive_viewport
                dst_x = (vpt.left() - self._viewport.left()) * spp
                dst_y = (vpt.top()  - self._viewport.top())  * spp
                dst_w = vpt.width()  * spp
                dst_h = vpt.height() * spp
                p.drawImage(QRectF(dst_x, dst_y, dst_w, dst_h), self._progressive_image)

            # ── Layer 3: Last completed hi-res frame ───────────────────────
            # Drawn scaled to the current viewport so it tracks pan/zoom
            # instantly while a new render is computing.
            if self._display_image and not self._display_viewport.isEmpty():
                vpt = self._display_viewport
                dst_x = (vpt.left() - self._viewport.left()) * spp
                dst_y = (vpt.top()  - self._viewport.top())  * spp
                dst_w = vpt.width()  * spp
                dst_h = vpt.height() * spp
                p.drawImage(QRectF(dst_x, dst_y, dst_w, dst_h), self._display_image)

            # ── Layer 3: Vector contours ───────────────────────────────────
            p.save()
            p.translate(-self._viewport.left() * spp, -self._viewport.top() * spp)
            p.scale(spp, spp)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            import random as py_random
            rng = py_random.Random()
            for ch in self._model.visible_channels():
                if (ch.is_mask or ch.is_cell_mask) and ch.contour_visible and ch.contour_data:
                    pen = p.pen()
                    pen.setWidth(3)
                    pen.setCosmetic(True)
                    vpt = self._viewport
                    for lid, data in ch.contour_data.items():
                        bbox = data["bbox"]
                        if (bbox[2] < vpt.top()    or bbox[0] > vpt.bottom() or
                                bbox[3] < vpt.left() or bbox[1] > vpt.right()):
                            continue
                        if ch.is_mask:
                            if ch.random_contour_colors:
                                rng.seed(int(lid))
                                col = QColor.fromRgbF(rng.random(), rng.random(), rng.random())
                            else:
                                col = ch.color
                        else:
                            # is_cell_mask
                            state = lid # fallback
                            if ch.pos_lut is not None and lid < len(ch.pos_lut):
                                state = ch.pos_lut[lid]
                            
                            # Inverted colors for contrast: Red contour on Green cell, Green contour on Red cell
                            col = QColor(255, 0, 0) if state == 2 else QColor(0, 255, 0) # 2=pos (red contour), 1=neg (green contour)
                        pen.setColor(col)
                        p.setPen(pen)
                        for qpoly in data["polygons"]:
                            p.drawPolyline(qpoly)
            p.restore()
            p.end()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Opal] paintEvent error: {e}")

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        if not self._img:
            return
        mx, my = event.position().x(), event.position().y()
        spp = self._screen_pixels_per_image_pixel()
        img_x = self._viewport.left() + mx / spp
        img_y = self._viewport.top()  + my / spp

        delta = event.angleDelta().y()
        if delta > 0:
            factor = 1.0 / self.ZOOM_FACTOR
        elif delta < 0:
            factor = self.ZOOM_FACTOR
        else:
            return

        new_w = self._viewport.width()  * factor
        new_h = self._viewport.height() * factor
        if new_w < 1 or new_h < 1:
            return

        frac_x = mx / max(self.width(), 1)
        frac_y = my / max(self.height(), 1)
        self._viewport = QRectF(
            img_x - frac_x * new_w,
            img_y - frac_y * new_h,
            new_w, new_h,
        )
        self._schedule_render(immediate_progressive=True)
        self.update()

    # ── Pan ───────────────────────────────────────────────────────────────────

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
            self.pixelHovered.emit(
                int(self._viewport.left() + mx / spp),
                int(self._viewport.top()  + my / spp),
            )
        if self._panning:
            delta = event.position() - self._pan_start
            dx = -delta.x() / spp
            dy = -delta.y() / spp
            self._viewport = QRectF(
                self._viewport_at_pan_start.left() + dx,
                self._viewport_at_pan_start.top()  + dy,
                self._viewport_at_pan_start.width(),
                self._viewport_at_pan_start.height(),
            )
            self._schedule_render(immediate_progressive=True)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ── Resize ────────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        old_size = event.oldSize()
        new_size = event.size()
        if self._img and old_size.width() > 0 and old_size.height() > 0:
            spp = old_size.width() / max(self._viewport.width(), 0.001)
            cx, cy = self._viewport.center().x(), self._viewport.center().y()
            new_vw = new_size.width()  / spp
            new_vh = new_size.height() / spp
            self._viewport = QRectF(cx - new_vw / 2, cy - new_vh / 2, new_vw, new_vh)
            self._schedule_render(immediate_progressive=True)
        super().resizeEvent(event)

    def _stop_worker(self):
        """Cleanly shut down the background render thread."""
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)  # 3 s timeout; avoid hanging on exit

    def closeEvent(self, event):
        self._stop_worker()
        super().closeEvent(event)
