"""Helpers for importing polygon regions from GeoJSON."""

from __future__ import annotations

from typing import Any


def geojson_region_parts(
    feature: dict[str, Any], fallback_name: str
) -> list[tuple[str, list[Any]]]:
    """Return one named exterior ring for each polygon in a GeoJSON feature.

    A MultiPolygon represents several separate regions.  When it contains more
    than one valid polygon, suffix every part with a human-readable number so
    none of those regions is silently discarded or ambiguously named.
    """
    geometry = feature.get("geometry") or {}
    properties = feature.get("properties") or {}
    coordinates = geometry.get("coordinates") or []

    if geometry.get("type") == "Polygon":
        polygons = [coordinates]
    elif geometry.get("type") == "MultiPolygon":
        polygons = coordinates
    else:
        return []

    exterior_rings = []
    for polygon in polygons:
        if not isinstance(polygon, (list, tuple)) or not polygon:
            continue
        exterior_ring = polygon[0]
        if isinstance(exterior_ring, (list, tuple)) and len(exterior_ring) >= 3:
            exterior_rings.append(exterior_ring)

    raw_name = properties.get("name")
    base_name = str(raw_name).strip() if raw_name is not None else ""
    if not base_name:
        base_name = fallback_name

    if len(exterior_rings) == 1:
        return [(base_name, exterior_rings[0])]
    return [
        (f"{base_name} {part_number}", ring)
        for part_number, ring in enumerate(exterior_rings, start=1)
    ]
