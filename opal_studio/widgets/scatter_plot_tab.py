"""
Scatter plot tab for displaying t-SNE / UMAP dimensionality reduction results.

Renders purely with QPainter — no matplotlib dependency.
Colours match the cluster mask colours in the channel panel.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QBrush
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy


# ---------------------------------------------------------------------------
# Scatter canvas — the drawing surface
# ---------------------------------------------------------------------------
class _ScatterCanvas(QWidget):
    """Internal widget that paints the scatter plot."""

    _POINT_RADIUS = 2.5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._coords: np.ndarray | None = None      # (n, 2)
        self._labels: np.ndarray | None = None      # (n,) int
        self._colors: dict[int, tuple] = {}         # cluster_id -> (R, G, B)
        self._hidden_clusters: set[int] = set()     # cluster_ids to skip in paint
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)

    # ------------------------------------------------------------------
    def set_data(
        self,
        coords: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_colors: dict[int, tuple],
    ) -> None:
        self._coords = coords.astype(np.float32)
        self._labels = cluster_labels
        self._colors = dict(cluster_colors)
        self._hidden_clusters = set()
        self.update()

    def update_colors(self, cluster_colors: dict[int, tuple]) -> None:
        self._colors = dict(cluster_colors)
        self.update()

    def set_hidden_clusters(self, hidden_ids: set[int]) -> None:
        self._hidden_clusters = set(hidden_ids)
        self.update()

    def clear(self) -> None:
        self._coords = None
        self._labels = None
        self._colors = {}
        self._hidden_clusters = set()
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#1e1e1e"))

        if self._coords is None or len(self._coords) == 0:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data")
            painter.end()
            return

        pad = 12
        plot_w = w - pad * 2
        plot_h = h - pad * 2
        if plot_w <= 0 or plot_h <= 0:
            painter.end()
            return

        xs = self._coords[:, 0]
        ys = self._coords[:, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        x_span = (x_max - x_min) or 1.0
        y_span = (y_max - y_min) or 1.0
        # 3 % padding in data space
        px = x_span * 0.03;  py = y_span * 0.03
        x_lo = x_min - px;   x_hi = x_max + px
        y_lo = y_min - py;   y_hi = y_max + py
        x_range = x_hi - x_lo
        y_range = y_hi - y_lo

        def to_screen(xi, yi):
            sx = pad + (xi - x_lo) / x_range * plot_w
            sy = pad + plot_h - (yi - y_lo) / y_range * plot_h
            return sx, sy

        r = self._POINT_RADIUS
        for lbl_id in np.unique(self._labels):
            if int(lbl_id) in self._hidden_clusters:
                continue
            rgb = self._colors.get(int(lbl_id), (180, 180, 180))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(*rgb)))
            mask = self._labels == lbl_id
            for xi, yi in self._coords[mask]:
                sx, sy = to_screen(float(xi), float(yi))
                painter.drawEllipse(QPointF(sx, sy), r, r)

        painter.end()


# ---------------------------------------------------------------------------
# Public tab widget
# ---------------------------------------------------------------------------
class ScatterPlotTab(QWidget):
    """Centre-panel tab wrapping a _ScatterCanvas."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._placeholder = QLabel(f"Run clustering to generate {title} plot.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-style: italic; background: #1e1e1e;")

        self._canvas = _ScatterCanvas()
        self._canvas.setVisible(False)

        layout.addWidget(self._placeholder)
        layout.addWidget(self._canvas)

    # ------------------------------------------------------------------
    def set_data(
        self,
        coords: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_colors: dict[int, tuple],
    ) -> None:
        self._placeholder.setVisible(False)
        self._canvas.setVisible(True)
        self._canvas.set_data(coords, cluster_labels, cluster_colors)

    def update_colors(self, cluster_colors: dict[int, tuple]) -> None:
        self._canvas.update_colors(cluster_colors)

    def set_hidden_clusters(self, hidden_ids: set[int]) -> None:
        self._canvas.set_hidden_clusters(hidden_ids)

    def clear(self) -> None:
        self._canvas.clear()
        self._canvas.setVisible(False)
        self._placeholder.setVisible(True)
