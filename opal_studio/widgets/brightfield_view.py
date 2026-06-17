"""Zoomable viewer for the generated brightfield RGB image with mask/region overlays."""

from __future__ import annotations

import math
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, Slot
from PySide6.QtGui import (
    QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF, QWheelEvent,
)
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from opal_studio.widgets.geometry import clip_polygon_to_rect


class BrightfieldView(QWidget):
    """
    Displays a brightfield RGB image with all visible mask/cell/type overlays
    and vector contours/regions drawn on top — mirroring the ImageCanvas layers.

    Also supports freehand region drawing in draw mode, emitting regionDrawn.

    Zoom: mouse wheel (anchored at cursor)   Pan: left-click drag   Reset: double-click
    """

    ZOOM_FACTOR = 1.15

    regionDrawn = Signal(list)      # list[QPointF] — simplified closed polygon
    viewportChanged = Signal(QRectF)  # emitted on user zoom/pan (image-space rect)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._model = channel_model
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._offset = QPointF()
        self._pan_start: QPointF | None = None
        self._panning = False

        # Draw-mode state (mirrors ImageCanvas)
        self._draw_mode = False
        self._drawing = False
        self._drawn_points: list[QPointF] = []
        self._simplification_epsilon = 1.0
        self._dragging_point = False
        self._drag_point_info = None
        self._hovered_point_info = None
        self._hovered_edge_info = None

        # Per-channel overlay cache: row → (fingerprint, QPixmap at full opacity)
        self._overlay_cache: dict[int, tuple[tuple, QPixmap]] = {}

        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        lbl = QLabel("Generate a brightfield image using the\nBrightfield tab in Pre-processing.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(lbl)
        self._placeholder = lbl

        self._model.channels_changed.connect(self.update)
        self._model.dataChanged.connect(lambda *_: self.update())

    # ── Public API ───────────────────────────────────────────────────────────

    def set_image(self, rgb_array: np.ndarray):
        """Accept a (H, W, 3) uint8 numpy array and display it."""
        arr = np.ascontiguousarray(rgb_array.astype(np.uint8))
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self._placeholder.setVisible(False)
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._overlay_cache.clear()
        self.update()

    def clear(self):
        self._pixmap = None
        self._overlay_cache.clear()
        self._placeholder.setVisible(True)
        self.update()

    def get_image_viewport(self) -> QRectF | None:
        """Return the currently visible region as an image-space QRectF."""
        if self._pixmap is None:
            return None
        fit = self._fit_scale()
        spp = fit * self._zoom
        if spp <= 0:
            return None
        iW, iH = self._pixmap.width(), self._pixmap.height()
        W, H = self.width(), self.height()
        ox = (W - iW * fit * self._zoom) / 2 + self._offset.x()
        oy = (H - iH * fit * self._zoom) / 2 + self._offset.y()
        return QRectF(-ox / spp, -oy / spp, W / spp, H / spp)

    def set_image_viewport(self, vp: QRectF):
        """
        Sync to an image-space viewport from ImageCanvas without emitting
        viewportChanged (avoids feedback loops).
        Uses the same spp as ImageCanvas: spp = canvas_widget_width / vp.width().
        We replicate that spp here so both views show the same magnification.
        """
        if self._pixmap is None or vp.width() <= 0:
            return
        fit = self._fit_scale()
        if fit <= 0:
            return
        iW, iH = self._pixmap.width(), self._pixmap.height()
        # Target: spp derived from viewport width to match canvas magnification
        target_spp = self.width() / vp.width()
        new_zoom = max(0.05, min(50.0, target_spp / fit))
        cx, cy = vp.center().x(), vp.center().y()
        self._zoom = new_zoom
        self._offset = QPointF(
            fit * self._zoom * (iW / 2 - cx),
            fit * self._zoom * (iH / 2 - cy),
        )
        self.update()

    @Slot(bool)
    def set_draw_mode(self, enabled: bool):
        self._draw_mode = enabled
        if not enabled:
            self._drawing = False
            self._drawn_points = []
            self._dragging_point = False
            self._drag_point_info = None
            self._hovered_point_info = None
            self._hovered_edge_info = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    @Slot(float)
    def set_simplification_epsilon(self, eps: float):
        self._simplification_epsilon = eps

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _screen_pixels_per_image_pixel(self) -> float:
        if self._pixmap is None:
            return 1.0
        return self._fit_scale() * self._zoom

    def _screen_to_image(self, mx: float, my: float) -> tuple[float, float]:
        """Convert widget pixel position to image-space coordinates."""
        spp = self._screen_pixels_per_image_pixel()
        fit = self._fit_scale()
        draw_w = self._pixmap.width()  * fit * self._zoom
        draw_h = self._pixmap.height() * fit * self._zoom
        ox = (self.width()  - draw_w) / 2 + self._offset.x()
        oy = (self.height() - draw_h) / 2 + self._offset.y()
        return (mx - ox) / spp, (my - oy) / spp

    def _image_to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        """Convert image-space coordinates to widget pixel position."""
        spp = self._screen_pixels_per_image_pixel()
        fit = self._fit_scale()
        draw_w = self._pixmap.width()  * fit * self._zoom
        draw_h = self._pixmap.height() * fit * self._zoom
        ox = (self.width()  - draw_w) / 2 + self._offset.x()
        oy = (self.height() - draw_h) / 2 + self._offset.y()
        return ox + ix * spp, oy + iy * spp

    # ── Contour simplification (RDP) ──────────────────────────────────────────

    def _simplify_contour(self, points: list[QPointF], epsilon: float) -> list[QPointF]:
        if len(points) <= 2 or epsilon <= 0.0:
            return points

        pts = [(pt.x(), pt.y()) for pt in points]

        def rdp(coords, eps):
            if len(coords) <= 2:
                return coords
            dmax = 0.0
            index = 0
            end = len(coords) - 1
            p1 = np.array(coords[0])
            p2 = np.array(coords[end])
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-9:
                for i in range(1, end):
                    d = np.linalg.norm(np.array(coords[i]) - p1)
                    if d > dmax:
                        index = i
                        dmax = d
            else:
                line_unit = line_vec / line_len
                for i in range(1, end):
                    p = np.array(coords[i])
                    v = p - p1
                    proj = np.dot(v, line_unit)
                    d = np.linalg.norm(v - proj * line_unit)
                    if d > dmax:
                        index = i
                        dmax = d
            if dmax > eps:
                return rdp(coords[:index + 1], eps)[:-1] + rdp(coords[index:], eps)
            else:
                return [coords[0], coords[end]]

        return [QPointF(x, y) for (x, y) in rdp(pts, epsilon)]

    def _point_to_segment_screen_dist(self, px, py, ax, ay, bx, by) -> float:
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def _remove_region_point(self, point_info):
        from PySide6.QtGui import QPolygonF
        ch, lid, poly_idx, pt_idx = point_info
        poly = ch.contour_data[lid]["polygons"][poly_idx]
        pts = [poly[i] for i in range(len(poly))]
        n_unique = len(pts) - 1
        if n_unique <= 3:
            return
        if pt_idx == len(pts) - 1:
            pt_idx = 0
        del pts[pt_idx]
        del pts[-1]
        pts.append(QPointF(pts[0]))
        new_poly = QPolygonF(pts)
        ch.contour_data[lid]["polygons"][poly_idx] = new_poly
        xs = [pt.x() for pt in pts]
        ys = [pt.y() for pt in pts]
        ch.contour_data[lid]["bbox"] = [min(ys), min(xs), max(ys), max(xs)]
        self._hovered_point_info = None
        self._hovered_edge_info = None
        self._model.channels_changed.emit()
        self.update()

    def _insert_region_point(self, edge_info, img_x, img_y):
        from PySide6.QtGui import QPolygonF
        ch, lid, poly_idx, edge_idx = edge_info
        poly = ch.contour_data[lid]["polygons"][poly_idx]
        pts = [poly[i] for i in range(len(poly))]
        new_pt_idx = edge_idx + 1
        pts.insert(new_pt_idx, QPointF(img_x, img_y))
        new_poly = QPolygonF(pts)
        ch.contour_data[lid]["polygons"][poly_idx] = new_poly
        xs = [pt.x() for pt in pts]
        ys = [pt.y() for pt in pts]
        ch.contour_data[lid]["bbox"] = [min(ys), min(xs), max(ys), max(xs)]
        self._hovered_edge_info = None
        self._model.channels_changed.emit()
        self.update()
        return (ch, lid, poly_idx, new_pt_idx)

    # ── Per-channel overlay pixmap cache ─────────────────────────────────────

    @staticmethod
    def _fingerprint(ch) -> tuple:
        return (
            id(ch.mask_data),
            ch.color.rgb() if hasattr(ch.color, 'rgb') else 0,
            ch.is_mask,
            ch.is_cell_mask,
            ch.is_type_mask,
            getattr(ch, 'random_contour_colors', False),
        )

    def _channel_overlay_pixmap(self, row: int, ch) -> QPixmap | None:
        """
        Return a fully-opaque RGBA QPixmap for the mask channel.
        Cached by row; rebuilt only when fingerprint changes (not on opacity change).
        """
        fp = self._fingerprint(ch)
        cached = self._overlay_cache.get(row)
        if cached is not None and cached[0] == fp:
            return cached[1]

        if self._pixmap is None or ch.mask_data is None:
            return None

        h, w = self._pixmap.height(), self._pixmap.width()
        raw = ch.mask_data
        if raw.shape[0] != h or raw.shape[1] != w:
            from skimage.transform import resize as sk_resize
            raw = sk_resize(raw, (h, w), order=0, preserve_range=True,
                            anti_aliasing=False).astype(raw.dtype)
        labels = raw.astype(np.int32)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        import random as py_random
        rng = py_random.Random()

        if ch.is_type_mask:
            active = labels > 0
            if not np.any(active):
                return None
            rgba[active, 0] = ch.color.red()
            rgba[active, 1] = ch.color.green()
            rgba[active, 2] = ch.color.blue()
            rgba[active, 3] = 255

        elif ch.is_mask:
            unique_ids = np.unique(labels)
            unique_ids = unique_ids[unique_ids > 0]
            if len(unique_ids) == 0:
                return None
            for lid in unique_ids:
                if ch.random_contour_colors:
                    rng.seed(int(lid))
                    r = int(rng.random() * 255)
                    g = int(rng.random() * 255)
                    b = int(rng.random() * 255)
                else:
                    r, g, b = ch.color.red(), ch.color.green(), ch.color.blue()
                m = labels == lid
                rgba[m, 0] = r; rgba[m, 1] = g; rgba[m, 2] = b; rgba[m, 3] = 255

        elif ch.is_cell_mask:
            if ch.pos_lut is None:
                return None
            max_id = int(labels.max())
            safe_lut = np.zeros(max_id + 1, dtype=np.int16)
            end = min(len(ch.pos_lut), max_id + 1)
            safe_lut[:end] = ch.pos_lut[:end]
            state_map = safe_lut[labels]
            m1 = state_map == 1
            m2 = state_map == 2
            rgba[m1, 1] = 255; rgba[m1, 3] = 255
            rgba[m2, 0] = 255; rgba[m2, 3] = 255

        else:
            return None

        rgba_c = np.ascontiguousarray(rgba)
        qimg = QImage(rgba_c.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg.copy())
        self._overlay_cache[row] = (fp, pix)
        return pix

    # ── Painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if self._pixmap is None:
            super().paintEvent(event)
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        fit = self._fit_scale()
        draw_w = self._pixmap.width()  * fit * self._zoom
        draw_h = self._pixmap.height() * fit * self._zoom
        ox = (self.width()  - draw_w) / 2 + self._offset.x()
        oy = (self.height() - draw_h) / 2 + self._offset.y()
        dst = QRectF(ox, oy, draw_w, draw_h)
        src_full = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())

        # Layer 1: brightfield base
        p.drawPixmap(dst, self._pixmap, src_full)

        # Layer 2: per-channel mask overlays at individual opacity
        for i in range(self._model.rowCount()):
            ch = self._model.channel(i)
            if getattr(ch, 'is_region', False):
                continue
            if not (ch.is_mask or ch.is_cell_mask or ch.is_type_mask):
                continue
            if not ch.visible or ch.mask_data is None:
                continue
            pix = self._channel_overlay_pixmap(i, ch)
            if pix is None:
                continue
            p.setOpacity(float(ch.range_max))
            p.drawPixmap(dst, pix, QRectF(pix.rect()))
        p.setOpacity(1.0)

        # Layer 3: vector contours, regions, and draw-mode UI (image-space coords)
        p.save()
        img_scale = fit * self._zoom
        p.translate(ox, oy)
        p.scale(img_scale, img_scale)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        spp = img_scale  # screen pixels per image pixel for handle sizing

        # In-progress freehand polygon
        if self._draw_mode and self._drawing and len(self._drawn_points) > 1:
            pen = QPen(QColor(255, 255, 0))
            pen.setWidth(3)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawPolyline(QPolygonF(self._drawn_points))

        import random as py_random
        rng = py_random.Random()

        for ch in self._model.visible_channels():
            is_r = getattr(ch, "is_region", False)
            draw_contours = (ch.is_mask or ch.is_cell_mask) and ch.contour_visible and ch.contour_data
            draw_region   = is_r and ch.visible and ch.contour_data
            if not draw_contours and not draw_region:
                continue

            pen = QPen()
            pen.setWidth(3)
            pen.setCosmetic(True)
            for lid, data in ch.contour_data.items():
                if is_r:
                    col = ch.color
                elif ch.is_mask:
                    if ch.random_contour_colors:
                        rng.seed(int(lid))
                        col = QColor.fromRgbF(rng.random(), rng.random(), rng.random())
                    else:
                        col = ch.color
                else:
                    state = lid
                    if ch.pos_lut is not None and lid < len(ch.pos_lut):
                        state = ch.pos_lut[lid]
                    col = QColor(255, 0, 0) if state == 2 else QColor(0, 255, 0)
                pen.setColor(col)
                p.setPen(pen)
                for qpoly in data["polygons"]:
                    p.drawPolyline(qpoly)

        # Region vertex handles in draw mode
        if self._draw_mode:
            for ch in self._model.visible_channels():
                if getattr(ch, "is_region", False) and ch.visible and ch.contour_data:
                    for lid, data in ch.contour_data.items():
                        for poly in data["polygons"]:
                            for pt in poly:
                                p.setPen(ch.color)
                                p.setBrush(ch.color)
                                p.drawEllipse(pt, 4.0 / spp, 4.0 / spp)

        p.restore()
        p.end()

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        if self._draw_mode and event.button() == Qt.MouseButton.RightButton:
            if self._hovered_point_info is not None:
                self._remove_region_point(self._hovered_point_info)
            return
        if self._draw_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._hovered_point_info is not None:
                self._dragging_point = True
                self._drag_point_info = self._hovered_point_info
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif self._hovered_edge_info is not None and self._pixmap is not None:
                mx, my = event.position().x(), event.position().y()
                ix, iy = self._screen_to_image(mx, my)
                drag_info = self._insert_region_point(self._hovered_edge_info, ix, iy)
                self._dragging_point = True
                self._drag_point_info = drag_info
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self._drawing = True
                self._drawn_points = []
                if self._pixmap is not None:
                    mx, my = event.position().x(), event.position().y()
                    ix, iy = self._screen_to_image(mx, my)
                    self._drawn_points.append(QPointF(ix, iy))
                self.update()
        elif event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._pixmap is None:
            return
        mx, my = event.position().x(), event.position().y()

        if self._draw_mode and self._dragging_point:
            ix, iy = self._screen_to_image(mx, my)
            ch, lid, poly_idx, pt_idx = self._drag_point_info
            poly = ch.contour_data[lid]["polygons"][poly_idx]
            new_pt = QPointF(ix, iy)
            poly[pt_idx] = new_pt
            # Keep closure: first == last
            if pt_idx == 0:
                poly[len(poly) - 1] = new_pt
            elif pt_idx == len(poly) - 1:
                poly[0] = new_pt
            xs = [pt.x() for pt in poly]
            ys = [pt.y() for pt in poly]
            ch.contour_data[lid]["bbox"] = [min(ys), min(xs), max(ys), max(xs)]
            self._model.channels_changed.emit()
            self.update()

        elif self._draw_mode and self._drawing:
            ix, iy = self._screen_to_image(mx, my)
            self._drawn_points.append(QPointF(ix, iy))
            self.update()

        elif self._panning and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._offset += delta
            self._pan_start = event.position()
            self.update()
            vp = self.get_image_viewport()
            if vp is not None:
                self.viewportChanged.emit(vp)

        elif self._draw_mode:
            self._update_hover(mx, my)

    def _update_hover(self, mx: float, my: float):
        hovered_pt = None
        best_pt_dist = 8.0
        hovered_edge = None
        best_edge_dist = 8.0
        for ch in self._model._channels:
            if getattr(ch, "is_region", False) and ch.visible and ch.contour_data:
                for lid, data in ch.contour_data.items():
                    for poly_idx, poly in enumerate(data["polygons"]):
                        n = len(poly)
                        for pt_idx, pt in enumerate(poly):
                            sx, sy = self._image_to_screen(pt.x(), pt.y())
                            dist = math.hypot(mx - sx, my - sy)
                            if dist < best_pt_dist:
                                best_pt_dist = dist
                                hovered_pt = (ch, lid, poly_idx, pt_idx)
                        for edge_idx in range(n - 1):
                            a = poly[edge_idx]
                            b = poly[edge_idx + 1]
                            ax, ay = self._image_to_screen(a.x(), a.y())
                            bx, by = self._image_to_screen(b.x(), b.y())
                            dist = self._point_to_segment_screen_dist(mx, my, ax, ay, bx, by)
                            if dist < best_edge_dist:
                                best_edge_dist = dist
                                hovered_edge = (ch, lid, poly_idx, edge_idx)
        if hovered_pt is not None:
            self._hovered_point_info = hovered_pt
            self._hovered_edge_info = None
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif hovered_edge is not None:
            self._hovered_point_info = None
            self._hovered_edge_info = hovered_edge
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._hovered_point_info = None
            self._hovered_edge_info = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._draw_mode and self._dragging_point:
            self._dragging_point = False
            self._drag_point_info = None
            mx, my = event.position().x(), event.position().y()
            self._update_hover(mx, my)
            self.update()
        elif self._draw_mode and self._drawing:
            self._drawing = False
            if len(self._drawn_points) >= 3:
                self._drawn_points.append(self._drawn_points[0])
                simplified = self._simplify_contour(
                    self._drawn_points, self._simplification_epsilon)
                if self._pixmap is not None:
                    simplified = clip_polygon_to_rect(
                        simplified, self._pixmap.width(), self._pixmap.height())
                if len(simplified) >= 4:  # 3 unique vertices + closing point
                    self.regionDrawn.emit(simplified)
            self._drawn_points = []
            self.update()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ── Zoom / Pan ────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.ZOOM_FACTOR if delta > 0 else (1.0 / self.ZOOM_FACTOR)

        mx = event.position().x()
        my = event.position().y()
        fit = self._fit_scale()

        draw_w = self._pixmap.width()  * fit * self._zoom
        draw_h = self._pixmap.height() * fit * self._zoom
        ox = (self.width()  - draw_w) / 2 + self._offset.x()
        oy = (self.height() - draw_h) / 2 + self._offset.y()

        img_x = (mx - ox) / (fit * self._zoom)
        img_y = (my - oy) / (fit * self._zoom)

        new_zoom = max(0.05, min(50.0, self._zoom * factor))
        new_draw_w = self._pixmap.width()  * fit * new_zoom
        new_draw_h = self._pixmap.height() * fit * new_zoom

        new_ox = mx - img_x * fit * new_zoom
        new_oy = my - img_y * fit * new_zoom

        self._zoom = new_zoom
        self._offset = QPointF(
            new_ox - (self.width()  - new_draw_w) / 2,
            new_oy - (self.height() - new_draw_h) / 2,
        )
        self.update()
        vp = self.get_image_viewport()
        if vp is not None:
            self.viewportChanged.emit(vp)

    def contextMenuEvent(self, event):
        if self._draw_mode:
            event.accept()
            return
        super().contextMenuEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if not self._draw_mode:
            self._zoom = 1.0
            self._offset = QPointF(0, 0)
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fit_scale(self) -> float:
        if self._pixmap is None:
            return 1.0
        return min(self.width() / max(1, self._pixmap.width()),
                   self.height() / max(1, self._pixmap.height()))
