"""Which cells of a label map lie in an area of the image.

Segmentation and cell positivity can work on the full image, the part shown in
the viewer, the selected region or all regions. A cell belongs to the area when
its centroid does. A centroid (cy, cx) in pixel indices is the point
(cx + 0.5, cy + 0.5) in image coordinates, where pixel centres sit and where
region outlines are drawn.
"""

from __future__ import annotations

import math

import numpy as np

#: The area modes, as passed in a run's ``region_mode`` parameter.
AREA_MODES = ("full", "visible", "selected_region", "regions")


def label_centroids(labels: np.ndarray):
    """Label IDs present in ``labels`` and their centroids (row, col), in pixel indices."""
    n = int(labels.max()) + 1
    count = np.zeros(n, dtype=np.float64)
    sum_y = np.zeros(n, dtype=np.float64)
    sum_x = np.zeros(n, dtype=np.float64)
    h, w = labels.shape
    cols = np.arange(w, dtype=np.float64)
    step = max(1, 4_000_000 // max(w, 1))   # rows per block, ~4M pixels
    for y0 in range(0, h, step):
        block = labels[y0:y0 + step]
        flat = block.ravel()
        rows = np.repeat(np.arange(y0, y0 + block.shape[0], dtype=np.float64), w)
        count += np.bincount(flat, minlength=n)
        sum_y += np.bincount(flat, weights=rows, minlength=n)
        sum_x += np.bincount(flat, weights=np.tile(cols, block.shape[0]), minlength=n)
    ids = np.flatnonzero(count[1:]) + 1
    return ids, sum_y[ids] / count[ids], sum_x[ids] / count[ids]


class PolygonArea:
    """Point-in-polygon tests for many centroids against a set of outlines.

    The polygons are rasterised once over their bounding box, with the pixels
    along each outline marked separately: a centroid whose four surrounding
    pixels are all clearly inside or all clearly outside takes that answer, and
    only the few near an outline get an exact test. A point is inside when it is
    inside any of the polygons (each with the odd-even rule).
    """

    def __init__(self, polygons):
        """``polygons``: a list of point lists [(x, y), ...] in image coordinates."""
        self.polygons = [list(p) for p in polygons if len(p) >= 3]
        if not self.polygons:
            raise ValueError("no polygon with at least three points")
        xs = [x for p in self.polygons for x, _ in p]
        ys = [y for p in self.polygons for _, y in p]
        self.top, self.bottom = int(math.floor(min(ys))), int(math.ceil(max(ys))) + 1
        self.left, self.right = int(math.floor(min(xs))), int(math.ceil(max(xs))) + 1
        self._mask = None       # 1 inside a polygon, 2 on an outline

    def _build_mask(self):
        import cv2
        mask = np.zeros((self.bottom - self.top, self.right - self.left), dtype=np.uint8)
        # cv2 puts pixel (r, c)'s centre at (c, r); sub-pixel vertices via a 4-bit
        # fixed-point shift. fillPoly also fills pixels the outline merely passes
        # through, so the outline is drawn over it (2 px wide) to mark those
        # pixels as undecided.
        rings = [np.round((np.asarray(p, dtype=np.float64)
                           - (self.left + 0.5, self.top + 0.5)) * 16).astype(np.int32)
                 for p in self.polygons]
        for ring in rings:
            cv2.fillPoly(mask, [ring], 1, lineType=cv2.LINE_8, shift=4)
        for ring in rings:
            cv2.polylines(mask, [ring], True, 2, thickness=2, lineType=cv2.LINE_8, shift=4)
        return mask

    def contains(self, cy, cx) -> np.ndarray:
        """Which centroids (pixel-index row, col arrays) lie inside the polygons."""
        cy = np.asarray(cy, dtype=np.float64)
        cx = np.asarray(cx, dtype=np.float64)
        if self._mask is None:
            self._mask = self._build_mask()
        mask = self._mask
        mh, mw = mask.shape
        py, px = cy - self.top, cx - self.left          # in mask pixel-centre coordinates
        y0 = np.floor(py).astype(np.int64)
        x0 = np.floor(px).astype(np.int64)
        corners = []
        for yy, xx in ((y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)):
            ok = (yy >= 0) & (yy < mh) & (xx >= 0) & (xx < mw)
            v = np.zeros(len(py), dtype=np.uint8)
            v[ok] = mask[yy[ok], xx[ok]]
            corners.append(v)
        inside = corners[0] == 1
        sure = ((corners[0] != 2) & (corners[0] == corners[1])
                & (corners[0] == corners[2]) & (corners[0] == corners[3]))
        unsure = np.flatnonzero(~sure)
        if len(unsure):
            from PySide6.QtCore import QPointF, Qt
            from PySide6.QtGui import QPolygonF
            polys = [QPolygonF([QPointF(x, y) for x, y in p]) for p in self.polygons]
            for i in unsure:
                p = QPointF(float(cx[i]) + 0.5, float(cy[i]) + 0.5)
                inside[i] = any(q.containsPoint(p, Qt.FillRule.OddEvenFill) for q in polys)
        return inside


def cells_in_area(labels: np.ndarray, mode: str, viewport=None, polygons=None):
    """The cells of ``labels`` in an area, and the box that holds all their pixels.

    ``mode`` is one of AREA_MODES. "visible" needs ``viewport`` as
    (left, top, right, bottom) in image coordinates; "selected_region" and
    "regions" need ``polygons`` (point lists in image coordinates).

    Returns ``(ids, box)``: the label IDs, and ``(y0, y1, x0, x1)`` bounding every
    pixel of those cells, or ``None`` when there are none.
    """
    if mode not in AREA_MODES:
        raise ValueError(f"unknown area mode {mode!r}")
    if mode == "full":
        counts = np.bincount(labels.ravel())
        ids = np.flatnonzero(counts[1:]) + 1
        if not len(ids):
            return ids, None
        rows = np.flatnonzero(labels.any(axis=1))
        cols = np.flatnonzero(labels.any(axis=0))
        return ids, (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)

    ids, cy, cx = label_centroids(labels)
    if mode == "visible":
        left, top, right, bottom = viewport
        px, py = cx + 0.5, cy + 0.5
        inside = (px >= left) & (px < right) & (py >= top) & (py < bottom)
    else:
        inside = PolygonArea(polygons).contains(cy, cx)
    ids = ids[inside]
    if not len(ids):
        return ids, None

    selected = np.zeros(int(labels.max()) + 1, dtype=bool)
    selected[ids] = True
    h, w = labels.shape
    rows_any = np.zeros(h, dtype=bool)
    cols_any = np.zeros(w, dtype=bool)
    step = max(1, 4_000_000 // max(w, 1))
    for y0 in range(0, h, step):
        m = selected[labels[y0:y0 + step]]
        rows_any[y0:y0 + m.shape[0]] = m.any(axis=1)
        cols_any |= m.any(axis=0)
    rows = np.flatnonzero(rows_any)
    cols = np.flatnonzero(cols_any)
    return ids, (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
