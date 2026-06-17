"""Small geometry helpers shared by the image views."""

from __future__ import annotations

from PySide6.QtCore import QPointF


def clip_polygon_to_rect(points: list[QPointF], width: float, height: float) -> list[QPointF]:
    """Clip a closed polygon to the image rectangle [0, width] x [0, height].

    Uses the Sutherland–Hodgman algorithm: the polygon is clipped in turn
    against each of the four image edges (left, right, top, bottom). Where the
    contour leaves and re-enters the image the boundary is followed along the
    edge, and corner vertices are inserted automatically when the entry and exit
    happen on different sides.

    `points` is expected to be a closed ring (points[0] == points[-1]). Returns a
    closed ring clipped to the rectangle, or an empty list if the polygon lies
    entirely outside the image.
    """
    if not points:
        return []

    # Work on the open ring (drop the duplicate closing vertex if present).
    ring = [(p.x(), p.y()) for p in points]
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        return []

    x0, y0, x1, y1 = 0.0, 0.0, float(width), float(height)

    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def clip_edge(poly, inside, intersect):
        if not poly:
            return poly
        out = []
        prev = poly[-1]
        prev_in = inside(prev)
        for cur in poly:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    # One point of each clipped segment is inside and the other outside the
    # relevant axis line, so the divisor below is always non-zero.
    poly = clip_edge(ring,                       # left:   x >= x0
                     lambda p: p[0] >= x0,
                     lambda a, b: lerp(a, b, (x0 - a[0]) / (b[0] - a[0])))
    poly = clip_edge(poly,                       # right:  x <= x1
                     lambda p: p[0] <= x1,
                     lambda a, b: lerp(a, b, (x1 - a[0]) / (b[0] - a[0])))
    poly = clip_edge(poly,                       # top:    y >= y0
                     lambda p: p[1] >= y0,
                     lambda a, b: lerp(a, b, (y0 - a[1]) / (b[1] - a[1])))
    poly = clip_edge(poly,                       # bottom: y <= y1
                     lambda p: p[1] <= y1,
                     lambda a, b: lerp(a, b, (y1 - a[1]) / (b[1] - a[1])))

    if len(poly) < 3:
        return []

    result = [QPointF(x, y) for (x, y) in poly]
    result.append(QPointF(result[0]))  # re-close the ring
    return result
