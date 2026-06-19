"""
Opal Studio project I/O — save / load a full session as a SpatialData-style
Zarr store, *without* depending on the ``spatialdata`` / ``geopandas`` /
``pyarrow`` libraries (none of which install in the pinned opal-all env).

The store mirrors the on-disk layout that :class:`opal_studio.image_loader.ZarrV3Array`
already reads, so generated images and label maps are written as genuine
Zarr V3 arrays (zstd-compressed, ``c/<c>/<y>/<x>`` chunk keys).  Tables are
written as AnnData-in-Zarr (the same encoding SpatialData uses).  Shapes
(vector regions) are written as a small JSON document, because real
``shapes`` elements need geopandas/geoparquet which is unavailable here.

The **original image is never copied** — only referenced.  ``attrs.source_image``
records both an absolute and a workspace-relative path plus enough metadata
to validate the reference on reload.

Layout
------
    <store>.zarr/
        zarr.json                     # root group: attrs = {source_image, opal_studio}
        images/
            derived/                  # generated channels (filter / average / …)
                zarr.json  +  0/      # OME multiscales group + level-0 V3 array
            brightfield/              # generated brightfield, c=3 (R,G,B)
        labels/
            <mask name>/              # one V3 array per cell / region / type mask
        shapes/
            regions/shapes.json       # vector regions (drawn / imported)
        tables/
            cells/                    # AnnData-in-Zarr  (per-cell table)
        opal_aux/
            <key>/                    # auxiliary arrays (clustering state, LUTs)

Public API
----------
    save_project(store_path, doc)         -> None
    load_project(store_path)              -> ProjectDocument
    PROJECT_SCHEMA_VERSION
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import numcodecs as _numcodecs
    _ZSTD = _numcodecs.Zstd(level=5)
except ImportError:  # pragma: no cover - numcodecs ships with the env
    _numcodecs = None
    _ZSTD = None

from opal_studio.image_loader import ZarrV3Array

PROJECT_SCHEMA_VERSION = 1

# Default chunk size (Y, X) for written arrays. One channel per c-chunk.
_DEFAULT_CHUNK_YX = (1024, 1024)

_ILLEGAL_DIR_CHARS = '<>:"/\\|?*'


def _safe_dirname(key: str, used: set[str]) -> str:
    """Map an arbitrary logical key to a unique, filesystem-safe directory name.

    The true key is preserved separately in element metadata, so this is only
    cosmetic / collision-avoidance for the on-disk name.
    """
    base = "".join("_" if (c in _ILLEGAL_DIR_CHARS or ord(c) < 32) else c
                   for c in str(key)).strip().rstrip(".") or "element"
    name = base
    i = 1
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


# ──────────────────────────────────────────────────────────────────────────────
# Project document — the in-memory representation save/load exchange with the UI
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProjectDocument:
    """Everything needed to persist / restore an Opal Studio session.

    All numpy image/label arrays are 2-D (Y, X) except ``images`` payloads
    which are (C, Y, X).  All ``session`` / ``source_image`` content must be
    JSON-serialisable.
    """
    source_image: dict = field(default_factory=dict)
    session: dict = field(default_factory=dict)
    # name -> {"data": (C,Y,X) ndarray, "channel_labels": [str, ...]}
    images: dict = field(default_factory=dict)
    # name -> (Y, X) integer ndarray
    labels: dict = field(default_factory=dict)
    # name -> {"<id>": [ring, ...], ...}  where ring = [[x, y], ...]
    shapes: dict = field(default_factory=dict)
    # key -> ndarray (any shape); stored/restored verbatim
    aux: dict = field(default_factory=dict)
    # optional AnnData (per-cell table)
    table: Any = None


# ──────────────────────────────────────────────────────────────────────────────
# Low-level Zarr V3 array writer (symmetric with image_loader.ZarrV3Array)
# ──────────────────────────────────────────────────────────────────────────────

def _write_v3_array(array_dir: Path, data: np.ndarray,
                    chunk_yx: tuple[int, int] = _DEFAULT_CHUNK_YX) -> dict:
    """Write *data* (C, Y, X) as a Zarr V3 array and return its metadata dict."""
    if _ZSTD is None:
        raise RuntimeError("numcodecs is required to write Opal Studio projects")
    if data.ndim != 3:
        raise ValueError(f"_write_v3_array expects (C, Y, X), got shape {data.shape}")

    # Force little-endian, C-contiguous so raw chunk bytes match the reader.
    dt = data.dtype.newbyteorder("<")
    data = np.ascontiguousarray(data, dtype=dt)
    C, H, W = data.shape

    cy = min(int(chunk_yx[0]), H) or 1
    cx = min(int(chunk_yx[1]), W) or 1
    chunk_shape = (1, cy, cx)

    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [int(C), int(H), int(W)],
        "data_type": np.dtype(data.dtype).name,
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": [1, cy, cx]}},
        "chunk_key_encoding": {"name": "default",
                               "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 5, "checksum": False}},
        ],
        "attributes": {},
    }

    array_dir.mkdir(parents=True, exist_ok=True)
    (array_dir / "zarr.json").write_text(json.dumps(meta, indent=2))

    n_ty = (H + cy - 1) // cy
    n_tx = (W + cx - 1) // cx
    for c in range(C):
        for ty in range(n_ty):
            y0, y1 = ty * cy, min((ty + 1) * cy, H)
            for tx in range(n_tx):
                x0, x1 = tx * cx, min((tx + 1) * cx, W)
                block = np.ascontiguousarray(data[c, y0:y1, x0:x1])
                raw = _ZSTD.encode(block.tobytes())
                chunk_path = array_dir / "c" / str(c) / str(ty) / str(tx)
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                with open(chunk_path, "wb") as fh:
                    fh.write(raw)
    return meta


def _read_v3_array_full(array_dir: Path) -> np.ndarray:
    """Read a Zarr V3 array written by :func:`_write_v3_array` into memory."""
    meta_path = array_dir / "zarr.json"
    with open(meta_path) as fh:
        meta = json.load(fh)
    arr = ZarrV3Array(array_dir, meta)
    C, H, W = arr.shape
    out = np.empty((C, H, W), dtype=arr.dtype)
    for c in range(C):
        out[c] = arr[c, 0:H, 0:W]
    return out


def _write_group(group_dir: Path, attributes: Optional[dict] = None) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    meta = {"zarr_format": 3, "node_type": "group",
            "attributes": attributes or {}}
    (group_dir / "zarr.json").write_text(json.dumps(meta, indent=2))


def _write_image_element(images_dir: Path, dirname: str, key: str,
                         data: np.ndarray, channel_labels: list[str]) -> None:
    """Write an OME-Zarr-style multiscale image group with a single level."""
    elem_dir = images_dir / dirname
    omero_channels = [{"label": str(lbl)} for lbl in channel_labels]
    group_attrs = {
        "opal_studio": {"key": key},
        "ome": {
            "version": "0.4",
            "multiscales": [{
                "name": dirname,
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [{
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}],
                }],
            }],
            "omero": {"channels": omero_channels},
        }
    }
    _write_group(elem_dir, group_attrs)
    _write_v3_array(elem_dir / "0", data)


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def save_project(store_path: str | Path, doc: ProjectDocument) -> None:
    """Write *doc* to a SpatialData-style Zarr store at *store_path*.

    An existing store at the path is replaced.
    """
    store_path = Path(store_path)
    if store_path.exists():
        if not (store_path / "zarr.json").exists() and any(store_path.iterdir()):
            raise FileExistsError(
                f"{store_path} exists and is not an Opal Studio project store")
        shutil.rmtree(store_path)

    # ── Root group with attrs (source_image + opal_studio session) ──────────
    root_attrs = {
        "source_image": doc.source_image,
        "opal_studio": {
            "schema_version": PROJECT_SCHEMA_VERSION,
            **doc.session,
        },
    }
    _write_group(store_path, root_attrs)

    # ── Images (derived channels, brightfield) ──────────────────────────────
    if doc.images:
        images_dir = store_path / "images"
        _write_group(images_dir)
        used: set[str] = set()
        for name, payload in doc.images.items():
            dirname = _safe_dirname(name, used)
            _write_image_element(
                images_dir, dirname, name,
                np.asarray(payload["data"]),
                payload.get("channel_labels", []),
            )

    # ── Labels (cell / region / type masks) ─────────────────────────────────
    if doc.labels:
        labels_dir = store_path / "labels"
        _write_group(labels_dir)
        used = set()
        for name, arr in doc.labels.items():
            arr = np.asarray(arr)
            dirname = _safe_dirname(name, used)
            _write_image_element(labels_dir, dirname, name,
                                 arr[np.newaxis, ...], [name])

    # ── Shapes (vector regions) ─────────────────────────────────────────────
    if doc.shapes:
        shapes_dir = store_path / "shapes"
        _write_group(shapes_dir)
        used = set()
        for name, polygons in doc.shapes.items():
            dirname = _safe_dirname(name, used)
            elem = shapes_dir / dirname
            elem.mkdir(parents=True, exist_ok=True)
            (elem / "shapes.json").write_text(json.dumps(
                {"format": "opal_polygons_v1", "key": name,
                 "polygons": polygons}))

    # ── Auxiliary arrays (clustering state, LUTs) ───────────────────────────
    if doc.aux:
        aux_dir = store_path / "opal_aux"
        _write_group(aux_dir)
        used = set()
        for key, arr in doc.aux.items():
            arr = np.asarray(arr)
            orig_shape = list(arr.shape)
            arr3d = _to_3d(arr)
            dirname = _safe_dirname(key, used)
            elem = aux_dir / dirname
            _write_v3_array(elem, arr3d)
            # remember original shape + logical key so we can restore on load
            meta_path = elem / "zarr.json"
            meta = json.loads(meta_path.read_text())
            meta["attributes"]["opal_orig_shape"] = orig_shape
            meta["attributes"]["opal_key"] = key
            meta_path.write_text(json.dumps(meta, indent=2))

    # ── Table (per-cell AnnData) ────────────────────────────────────────────
    if doc.table is not None:
        tables_dir = store_path / "tables"
        _write_group(tables_dir)
        doc.table.write_zarr(tables_dir / "cells")


def _to_3d(arr: np.ndarray) -> np.ndarray:
    """Promote a 1-D / 2-D array to (C, Y, X) for the V3 writer."""
    if arr.ndim == 1:
        return arr[np.newaxis, np.newaxis, :]
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported aux array ndim={arr.ndim}")


# ──────────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────────

def load_project(store_path: str | Path) -> ProjectDocument:
    """Read a store written by :func:`save_project` back into a ProjectDocument."""
    store_path = Path(store_path)
    root_meta_path = store_path / "zarr.json"
    if not root_meta_path.exists():
        raise FileNotFoundError(f"No zarr.json at {store_path} — not a project store")

    with open(root_meta_path) as fh:
        root_meta = json.load(fh)
    attrs = root_meta.get("attributes", {})
    opal = attrs.get("opal_studio", {})

    doc = ProjectDocument(
        source_image=attrs.get("source_image", {}),
        session=opal,
    )

    schema = opal.get("schema_version")
    if schema is not None and schema > PROJECT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Project schema v{schema} is newer than supported "
            f"v{PROJECT_SCHEMA_VERSION}; please update Opal Studio")

    # ── Images ──────────────────────────────────────────────────────────────
    images_dir = store_path / "images"
    if images_dir.is_dir():
        for elem in sorted(images_dir.iterdir()):
            if not (elem / "0" / "zarr.json").exists():
                continue
            key = _read_group_key(elem / "zarr.json", elem.name)
            doc.images[key] = {
                "data": _read_v3_array_full(elem / "0"),
                "channel_labels": _read_omero_labels(elem / "zarr.json"),
            }

    # ── Labels ──────────────────────────────────────────────────────────────
    labels_dir = store_path / "labels"
    if labels_dir.is_dir():
        for elem in sorted(labels_dir.iterdir()):
            if not (elem / "0" / "zarr.json").exists():
                continue
            key = _read_group_key(elem / "zarr.json", elem.name)
            doc.labels[key] = _read_v3_array_full(elem / "0")[0]

    # ── Shapes ──────────────────────────────────────────────────────────────
    shapes_dir = store_path / "shapes"
    if shapes_dir.is_dir():
        for elem in sorted(shapes_dir.iterdir()):
            sj = elem / "shapes.json"
            if not sj.exists():
                continue
            payload = json.loads(sj.read_text())
            doc.shapes[payload.get("key", elem.name)] = payload.get("polygons", {})

    # ── Auxiliary arrays ────────────────────────────────────────────────────
    aux_dir = store_path / "opal_aux"
    if aux_dir.is_dir():
        for elem in sorted(aux_dir.iterdir()):
            meta_path = elem / "zarr.json"
            if not meta_path.exists():
                continue
            arr = _read_v3_array_full(elem)
            elem_attrs = json.loads(meta_path.read_text()).get("attributes", {})
            orig = elem_attrs.get("opal_orig_shape")
            if orig is not None:
                arr = arr.reshape(tuple(orig))
            doc.aux[elem_attrs.get("opal_key", elem.name)] = arr

    # ── Table ───────────────────────────────────────────────────────────────
    table_dir = store_path / "tables" / "cells"
    if table_dir.is_dir():
        try:
            import anndata as ad
            doc.table = ad.read_zarr(table_dir)
        except Exception:
            doc.table = None

    return doc


def _read_group_key(group_meta_path: Path, fallback: str) -> str:
    try:
        meta = json.loads(group_meta_path.read_text())
        return meta["attributes"]["opal_studio"]["key"]
    except Exception:
        return fallback


def _read_omero_labels(group_meta_path: Path) -> list[str]:
    try:
        meta = json.loads(group_meta_path.read_text())
        channels = meta["attributes"]["ome"]["omero"]["channels"]
        return [str(c.get("label", f"Channel {i}")) for i, c in enumerate(channels)]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Source-image reference helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_source_reference(image_path: Path, store_path: Path, *,
                          backend: str, base_shape, axes: str,
                          channel_names: list[str],
                          image_name: Optional[str] = None,
                          slice_index: Optional[int] = None) -> dict:
    """Build the ``source_image`` attrs block (absolute + relative paths)."""
    image_path = Path(image_path).resolve()
    store_parent = Path(store_path).resolve().parent
    try:
        rel = os_relpath(image_path, store_parent)
    except Exception:
        rel = None
    ref = {
        "uri": str(image_path),
        "relpath": rel,
        "backend": backend,                       # "ome-tiff" | "spatialdata"
        "base_shape": [int(s) for s in base_shape],
        "axes": axes,
        "n_channels": len(channel_names),
        "channel_names": list(channel_names),
    }
    if image_name is not None:
        ref["image_name"] = image_name
    if slice_index is not None:
        ref["slice_index"] = int(slice_index)
    return ref


def resolve_source_path(ref: dict, store_path: Path) -> Optional[Path]:
    """Return the first existing path for a source reference, or None."""
    uri = ref.get("uri")
    if uri and Path(uri).exists():
        return Path(uri)
    rel = ref.get("relpath")
    if rel:
        candidate = (Path(store_path).resolve().parent / rel).resolve()
        if candidate.exists():
            return candidate
    return None


def os_relpath(target: Path, start: Path) -> str:
    import os
    return os.path.relpath(str(target), str(start))
