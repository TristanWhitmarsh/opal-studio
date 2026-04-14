"""
Custom dual-handle range slider widget.

Allows setting a min/max window for channel intensity mapping.
The full track represents [0, 1] and the two handles define the
sub-range that maps to alpha 0 → 1.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QLinearGradient, QPalette
from PySide6.QtWidgets import QWidget, QSizePolicy


class RangeSlider(QWidget):
    """
    Horizontal slider with two draggable handles.

    Signals
    -------
    rangeChanged(min_val: float, max_val: float)
        Emitted whenever either handle is moved.  Values are in [0, 1].
    """

    rangeChanged = Signal(float, float)

    # Appearance
    HANDLE_W = 10
    HANDLE_H = 18
    TRACK_H = 6
    SIDE_MARGIN = 5
    MIN_HEIGHT = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min_val = 0.0
        self._max_val = 1.0
        self._dragging: str | None = None  # "min", "max", or None
        self.setMinimumHeight(self.MIN_HEIGHT)
        self.setMaximumHeight(self.MIN_HEIGHT + 4)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ---- properties ---------------------------------------------------

    @property
    def min_val(self) -> float:
        return self._min_val

    @min_val.setter
    def min_val(self, v: float):
        self._min_val = max(0.0, min(v, self._max_val))
        self.update()

    @property
    def max_val(self) -> float:
        return self._max_val

    @max_val.setter
    def max_val(self, v: float):
        self._max_val = min(1.0, max(v, self._min_val))
        self.update()


    def set_range(self, mn: float, mx: float):
        self._min_val = max(0.0, min(mn, 1.0))
        self._max_val = min(1.0, max(mx, 0.0))
        if self._min_val > self._max_val:
            self._min_val, self._max_val = self._max_val, self._min_val
        self.update()

    # ---- coordinate helpers -------------------------------------------

    def _track_rect(self) -> QRect:
        m = self.HANDLE_W // 2 + self.SIDE_MARGIN
        y = (self.height() - self.TRACK_H) // 2
        return QRect(m, y, self.width() - (self.HANDLE_W + 2 * self.SIDE_MARGIN), self.TRACK_H)

    def _val_to_x(self, v: float) -> int:
        tr = self._track_rect()
        return int(tr.x() + v * tr.width())

    def _x_to_val(self, x: int) -> float:
        tr = self._track_rect()
        return max(0.0, min(1.0, (x - tr.x()) / max(tr.width(), 1)))

    # ---- painting -----------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        tr = self._track_rect()

        # Track background
        p.setPen(Qt.PenStyle.NoPen)
        track_color = self.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Mid)
        p.setBrush(track_color)
        p.drawRoundedRect(tr, 3, 3)

        # Active range fill (neutral highlight color)
        x1 = self._val_to_x(self._min_val)
        x2 = self._val_to_x(self._max_val)
        active = QRect(x1, tr.y(), max(x2 - x1, 1), tr.height())

        fill_color = self.palette().color(QPalette.ColorRole.Highlight)
        p.setBrush(fill_color)
        p.drawRoundedRect(active, 3, 3)

        # Handles
        handle_brush = self.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Button)
        handle_pen = QPen(self.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Dark), 1)
        
        for val in (self._min_val, self._max_val):
            hx = self._val_to_x(val) - self.HANDLE_W // 2
            hy = (self.height() - self.HANDLE_H) // 2
            handle_rect = QRect(hx, hy, self.HANDLE_W, self.HANDLE_H)
            p.setBrush(handle_brush)
            p.setPen(handle_pen)
            p.drawRoundedRect(handle_rect, 3, 3)

        p.end()

    # ---- mouse interaction --------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        dist_min = abs(x - self._val_to_x(self._min_val))
        dist_max = abs(x - self._val_to_x(self._max_val))
        if dist_min <= dist_max:
            self._dragging = "min"
        else:
            self._dragging = "max"

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        v = self._x_to_val(int(event.position().x()))
        if self._dragging == "min":
            self._min_val = max(0.0, min(v, self._max_val - 0.01))
        else:
            self._max_val = min(1.0, max(v, self._min_val + 0.01))
        self.update()
        self.rangeChanged.emit(self._min_val, self._max_val)

    def mouseReleaseEvent(self, event):
        self._dragging = None
