"""Opal Studio project I/O — save / load a session as a **spec-compliant
SpatialData** store (Zarr v3 + OME-NGFF ``0.5-dev-spatialdata``).

The on-disk layout matches what the ``spatialdata`` library (v0.7.x, store
format ``0.2``) produces, so the stores Opal writes can be opened by
``spatialdata`` / ``squidpy`` and friends — *without* Opal depending on the
``spatialdata`` package itself (it conflicts with the pinned ML env).  We drive
the underlying standards directly: Zarr v3 metadata is hand-written, shapes use
``geopandas`` GeoParquet, and the cell table uses ``anndata``.

Layout
------
    <store>.zarr/
        zarr.json                     # root group
                                      #   attrs.spatialdata_attrs {version 0.2, software}
                                      #   attrs.opal_studio       {source_image, session, ...}
                                      #   consolidated_metadata   (all v3 descendants, inlined)
        images/<name>/                # OME-NGFF multiscale image (c,y,x) + omero channels
            zarr.json  +  0/          #   spatialdata_attrs.version 0.3
        labels/<name>/                # OME-NGFF label raster (y,x), no channel axis
            zarr.json  +  0/
        shapes/<name>/                # ngff:shapes group + shapes.parquet (GeoParquet)
            zarr.json  +  shapes.parquet
        tables/cells/                 # anndata-in-zarr regions table (ngff:regions_table)
        opal_aux/<key>/               # Opal-private arrays (clustering state, LUTs) — custom

Versions
--------
    spatialdata container = 0.2     raster (images/labels) = 0.3
    shapes                = 0.3     tables                 = 0.2
    OME-NGFF              = 0.5-dev-spatialdata

Compliance note (table)
-----------------------
The cell table is written with ``anndata``; with the pinned ``anndata``/``zarr``
it serialises in anndata's own (Zarr v2) on-disk encoding nested inside the v3
store.  Opal reads it back directly; full Zarr-v3 consolidation of the table is
deferred until the env can ship a v3-capable ``anndata``/``zarr``.

Public API
----------
    save_project(store_path, doc)         -> None
    load_project(store_path)              -> ProjectDocument
    PROJECT_SCHEMA_VERSION
"""

from __future__ import annotations

import itertools
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

PROJECT_SCHEMA_VERSION = 2

# ── SpatialData / NGFF format version strings (match spatialdata v0.7.x) ───────
SPATIALDATA_CONTAINER_VERSION = "0.2"
RASTER_VERSION = "0.3"
SHAPES_VERSION = "0.3"
TABLES_VERSION = "0.2"
NGFF_VERSION = "0.5-dev-spatialdata"

# Default spatial chunk edge (Y/X). One channel per c-chunk for images.
_CHUNK_EDGE = 1024

# zstd codec block used by the hand-written anndata-v3 encoder.
_ANNDATA_ZSTD = {"name": "zstd", "configuration": {"level": 5, "checksum": False}}


def _software_version() -> str:
    try:
        import importlib.metadata as _md
        return f"opal-studio {_md.version('opal-studio')}"
    except Exception:
        return "opal-studio"


def _safe_dirname(key: str, used: set[str]) -> str:
    """Map a logical key to a unique, SpatialData-valid directory name.

    SpatialData validates element names against ``[A-Za-z0-9_.-]`` (alphanumeric,
    underscore, dot, hyphen) and rejects anything else — spaces included — so a
    store with e.g. a ``"Tumor cells"`` mask cannot be opened by the
    ``spatialdata`` library.  We therefore replace every disallowed character
    (not just Windows-illegal ones) with ``_``.  The true key is preserved
    verbatim in element metadata (``opal_studio.key``), so this is a lossless,
    on-disk-only mapping.
    """
    base = "".join(c if (c.isascii() and (c.isalnum() or c in "_.-")) else "_"
                   for c in str(key)).strip("_").rstrip(".") or "element"
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
# Low-level Zarr V3 array I/O (default chunk-key encoding, bytes+zstd codecs)
# ──────────────────────────────────────────────────────────────────────────────

def _pick_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Choose a chunk shape: 1 channel per c-chunk, spatial dims capped."""
    if len(shape) == 3:  # (C, Y, X)
        return (1, min(shape[1], _CHUNK_EDGE) or 1, min(shape[2], _CHUNK_EDGE) or 1)
    if len(shape) == 2:  # (Y, X)
        return (min(shape[0], _CHUNK_EDGE) or 1, min(shape[1], _CHUNK_EDGE) or 1)
    return tuple(max(s, 1) for s in shape)  # single chunk for 0/1-D and odd ranks


def _array_meta(shape: tuple[int, ...], dtype: np.dtype,
                chunk_shape: tuple[int, ...]) -> dict:
    return {
        "shape": [int(s) for s in shape],
        "data_type": np.dtype(dtype).name,
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": [int(c) for c in chunk_shape]}},
        "chunk_key_encoding": {"name": "default",
                               "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 5, "checksum": False}},
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }


def _write_v3_array(array_dir: Path, data: np.ndarray) -> dict:
    """Write *data* (any rank) as a Zarr V3 array; return its metadata dict."""
    if _ZSTD is None:
        raise RuntimeError("numcodecs is required to write Opal Studio projects")

    dt = data.dtype.newbyteorder("<")            # force little-endian on disk
    data = np.ascontiguousarray(data, dtype=dt)
    shape = data.shape
    chunk = _pick_chunks(shape)
    meta = _array_meta(shape, data.dtype, chunk)

    array_dir.mkdir(parents=True, exist_ok=True)
    (array_dir / "zarr.json").write_text(json.dumps(meta, indent=2))

    n_chunks = [(shape[d] + chunk[d] - 1) // chunk[d] for d in range(len(shape))]
    for idx in itertools.product(*[range(n) for n in n_chunks]):
        slices = tuple(slice(idx[d] * chunk[d], min((idx[d] + 1) * chunk[d], shape[d]))
                       for d in range(len(shape)))
        block_shape = tuple(s.stop - s.start for s in slices)
        if block_shape == tuple(chunk):
            block = np.ascontiguousarray(data[slices])
        else:
            # Boundary chunk: the Zarr spec requires every stored chunk to be the
            # full chunk_shape, with out-of-bounds elements set to fill_value (0).
            # Padding (vs. clipping) is what makes the store readable by standard
            # zarr-python / spatialdata. Compression keeps the padding ~free.
            block = np.zeros(chunk, dtype=dt)
            block[tuple(slice(0, b) for b in block_shape)] = data[slices]
        raw = _ZSTD.encode(np.ascontiguousarray(block).tobytes())
        chunk_path = array_dir.joinpath("c", *map(str, idx))
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.write_bytes(raw)
    return meta


def _read_v3_array(array_dir: Path) -> np.ndarray:
    """Read a Zarr V3 array written by :func:`_write_v3_array` into memory."""
    meta = json.loads((array_dir / "zarr.json").read_text())
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["data_type"]).newbyteorder("<")
    chunk = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])

    out = np.zeros(shape, dtype=dtype)
    n_chunks = [(shape[d] + chunk[d] - 1) // chunk[d] for d in range(len(shape))]
    full_size = int(np.prod(chunk))
    for idx in itertools.product(*[range(n) for n in n_chunks]):
        chunk_path = array_dir.joinpath("c", *map(str, idx))
        if not chunk_path.exists():
            continue                              # implicit fill_value (0)
        raw = _ZSTD.decode(chunk_path.read_bytes())
        slices = tuple(slice(idx[d] * chunk[d], min((idx[d] + 1) * chunk[d], shape[d]))
                       for d in range(len(shape)))
        block_shape = tuple(s.stop - s.start for s in slices)
        vals = np.frombuffer(raw, dtype=dtype)
        if vals.size == full_size:
            # Spec-compliant full chunk: reshape then trim to the valid region.
            block = vals.reshape(chunk)[tuple(slice(0, b) for b in block_shape)]
        else:
            # Legacy clipped edge chunk (older opal-studio stores).
            block = vals.reshape(block_shape)
        out[slices] = block
    return out.astype(out.dtype.newbyteorder("="), copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# Zarr V3 group + NGFF metadata helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_group(group_dir: Path, attributes: Optional[dict] = None) -> None:
    """Write a plain Zarr-v3 group.

    Sub-groups carry *no* ``consolidated_metadata`` key — only the store root
    does (a flat listing of every descendant).  This matches what SpatialData
    writes and lets readers that open a sub-group directly (e.g. anndata opening
    the ``tables/cells`` group) fall back to a normal on-disk hierarchy scan.
    """
    group_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "attributes": attributes or {},
        "zarr_format": 3,
        "node_type": "group",
    }
    (group_dir / "zarr.json").write_text(json.dumps(meta, indent=2))


def _ngff_axis(name: str, *, unit: bool) -> dict:
    ax = {"name": name, "type": "channel" if name == "c" else "space"}
    if unit and name != "c":
        ax["unit"] = "unit"
    return ax


def _coordinate_systems_axes(names: tuple[str, ...]) -> list[dict]:
    return [_ngff_axis(n, unit=True) for n in names]


def _identity_transform(names: tuple[str, ...]) -> dict:
    """NGFF identity transform between the element's CS and the 'global' CS."""
    return {
        "type": "identity",
        "input": {"name": "".join(names), "axes": _coordinate_systems_axes(names)},
        "output": {"name": "global", "axes": _coordinate_systems_axes(names)},
    }


def _raster_attrs(element_path: str, names: tuple[str, ...],
                  channel_labels: Optional[list[str]]) -> dict:
    """Element-group attributes for an image (c,y,x) or label (y,x) raster."""
    ome: dict = {}
    if channel_labels is not None:                # images carry omero channels
        ome["omero"] = {"channels": [{"label": str(lbl)} for lbl in channel_labels]}
    ome["version"] = NGFF_VERSION
    ome["multiscales"] = [{
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [{"type": "scale",
                                           "scale": [1.0] * len(names)}],
        }],
        "name": element_path,
        "axes": [_ngff_axis(n, unit=False) for n in names],
        "coordinateTransformations": [_identity_transform(names)],
    }]
    return {"ome": ome, "spatialdata_attrs": {"version": RASTER_VERSION}}


def _write_raster_element(parent_dir: Path, dirname: str, key: str,
                          data: np.ndarray, names: tuple[str, ...],
                          channel_labels: Optional[list[str]]) -> None:
    elem_dir = parent_dir / dirname
    attrs = _raster_attrs(f"/{parent_dir.name}/{dirname}", names, channel_labels)
    attrs["opal_studio"] = {"key": key}           # custom: preserve true key
    _write_group(elem_dir, attrs)
    _write_v3_array(elem_dir / "0", data)


def _write_shapes_element(parent_dir: Path, dirname: str, key: str,
                          polygons: dict) -> None:
    """Write a regions element: ngff:shapes group + GeoParquet of polygons."""
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    geoms: list = []
    index: list = []
    for sid, rings in polygons.items():
        polys = [Polygon([(float(x), float(y)) for x, y in ring])
                 for ring in rings if len(ring) >= 3]
        if not polys:
            continue
        geoms.append(polys[0] if len(polys) == 1 else MultiPolygon(polys))
        s = str(sid)
        index.append(int(s) if s.lstrip("-").isdigit() else s)

    elem_dir = parent_dir / dirname
    attrs = {
        "encoding-type": "ngff:shapes",
        "axes": ["x", "y"],
        "coordinateTransformations": [_identity_transform(("x", "y"))],
        "spatialdata_attrs": {"version": SHAPES_VERSION},
        "opal_studio": {"key": key},              # custom: preserve true key
    }
    _write_group(elem_dir, attrs)

    gdf = gpd.GeoDataFrame({"geometry": geoms}, index=index)
    gdf.to_parquet(elem_dir / "shapes.parquet")


# ──────────────────────────────────────────────────────────────────────────────
# AnnData → Zarr V3 (anndata's on-disk "encoding-type" format, hand-written)
#
# Mirrors what anndata writes into a Zarr-v3 group so that SpatialData / squidpy
# recognise the cell table.  Covered encodings: anndata, dict, dataframe,
# categorical, array, string-array, string, numeric-scalar, null.
# ──────────────────────────────────────────────────────────────────────────────

def _aw_chunk(node_dir: Path, raw: bytes, ndim: int) -> None:
    if ndim == 0:                                 # scalar → single file "c"
        (node_dir / "c").write_bytes(raw)
    else:                                         # single chunk "c/0[/0...]"
        cp = node_dir.joinpath("c", *(["0"] * ndim))
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(raw)


def _aw_array_meta(shape, data_type, fill_value, codecs, attrs) -> dict:
    return {
        "shape": list(shape),
        "data_type": data_type,
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": list(shape)}},
        "chunk_key_encoding": {"name": "default",
                               "configuration": {"separator": "/"}},
        "fill_value": fill_value,
        "codecs": codecs,
        "attributes": attrs,
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }


def _aw_enc(et: str, ev: str) -> dict:
    return {"encoding-type": et, "encoding-version": ev}


def _aw_write_array(node_dir: Path, arr: np.ndarray, et: str = "array",
                    ev: str = "0.2.0") -> None:
    arr = np.ascontiguousarray(arr)
    if arr.dtype.itemsize == 1:                   # int8 / bool: no endian
        bytes_codec = {"name": "bytes"}
    else:
        arr = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("<"))
        bytes_codec = {"name": "bytes", "configuration": {"endian": "little"}}
    if arr.dtype.kind == "b":
        fill: Any = False
    elif arr.dtype.kind == "f":
        fill = 0.0
    else:
        fill = 0
    meta = _aw_array_meta(arr.shape, np.dtype(arr.dtype).name, fill,
                          [bytes_codec, dict(_ANNDATA_ZSTD)], _aw_enc(et, ev))
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "zarr.json").write_text(json.dumps(meta, indent=2))
    _aw_chunk(node_dir, _ZSTD.encode(arr.tobytes()), arr.ndim)


def _aw_write_string_array(node_dir: Path, values, *, scalar: bool = False,
                           et: str = "string-array", ev: str = "0.2.0") -> None:
    if scalar:
        encode_arr = np.array(["" if values is None else str(values)], dtype=object)
        shape: tuple = ()
        ndim = 0
    else:
        encode_arr = np.array(["" if v is None else str(v) for v in values],
                              dtype=object)
        shape = encode_arr.shape
        ndim = 1
    meta = _aw_array_meta(shape, "string", "",
                          [{"name": "vlen-utf8", "configuration": {}},
                           dict(_ANNDATA_ZSTD)], _aw_enc(et, ev))
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "zarr.json").write_text(json.dumps(meta, indent=2))
    raw = _ZSTD.encode(bytes(_numcodecs.VLenUTF8().encode(encode_arr)))
    _aw_chunk(node_dir, raw, ndim)


def _aw_write_group(node_dir: Path, et: str, ev: str, extra: Optional[dict] = None) -> None:
    attrs = _aw_enc(et, ev)
    if extra:
        attrs.update(extra)
    _write_group(node_dir, attrs)


def _aw_write_scalar(node_dir: Path, value: Any) -> None:
    if isinstance(value, str):
        _aw_write_string_array(node_dir, value, scalar=True, et="string", ev="0.2.0")
        return
    arr = np.asarray(value)                        # 0-d numeric / bool
    meta_arr = arr.reshape(())
    if meta_arr.dtype.itemsize == 1:
        bytes_codec = {"name": "bytes"}
    else:
        meta_arr = np.ascontiguousarray(meta_arr, dtype=meta_arr.dtype.newbyteorder("<"))
        bytes_codec = {"name": "bytes", "configuration": {"endian": "little"}}
    fill = False if meta_arr.dtype.kind == "b" else (0.0 if meta_arr.dtype.kind == "f" else 0)
    meta = _aw_array_meta((), np.dtype(meta_arr.dtype).name, fill,
                          [bytes_codec, dict(_ANNDATA_ZSTD)],
                          _aw_enc("numeric-scalar", "0.2.0"))
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "zarr.json").write_text(json.dumps(meta, indent=2))
    _aw_chunk(node_dir, _ZSTD.encode(meta_arr.tobytes()), 0)


def _aw_write_column(node_dir: Path, series) -> None:
    import pandas as pd
    if isinstance(series.dtype, pd.CategoricalDtype):
        _aw_write_group(node_dir, "categorical", "0.2.0",
                        {"ordered": bool(series.cat.ordered)})
        _aw_write_string_array(node_dir / "categories",
                               list(series.cat.categories))
        _aw_write_array(node_dir / "codes",
                        np.asarray(series.cat.codes.values))
    elif series.dtype == object or series.dtype.kind in ("U", "S"):
        _aw_write_string_array(node_dir, list(series.values))
    else:
        _aw_write_array(node_dir, np.asarray(series.values))


def _aw_write_dataframe(node_dir: Path, df) -> None:
    index_name = df.index.name or "_index"
    _aw_write_group(node_dir, "dataframe", "0.2.0",
                    {"column-order": [str(c) for c in df.columns],
                     "_index": str(index_name)})
    _aw_write_string_array(node_dir / str(index_name),
                           [str(v) for v in df.index])
    for col in df.columns:
        _aw_write_column(node_dir / str(col), df[col])


def _aw_write_mapping(node_dir: Path, mapping) -> None:
    _aw_write_group(node_dir, "dict", "0.1.0")
    for key, value in mapping.items():
        _aw_write_uns_value(node_dir / str(key), value)


def _aw_write_uns_value(node_dir: Path, value: Any) -> None:
    import pandas as pd
    if isinstance(value, dict):
        _aw_write_mapping(node_dir, value)
    elif isinstance(value, str):
        _aw_write_scalar(node_dir, value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.dtype.kind in ("U", "S", "O"):
            _aw_write_string_array(node_dir, [str(v) for v in arr.ravel()])
        else:
            _aw_write_array(node_dir, arr)
    elif isinstance(value, (bool, int, float, np.integer, np.floating, np.bool_)):
        _aw_write_scalar(node_dir, value)
    elif isinstance(value, pd.DataFrame):
        _aw_write_dataframe(node_dir, value)
    else:
        _aw_write_scalar(node_dir, str(value))


def _write_null(node_dir: Path) -> None:
    """Encode an absent element (e.g. raw=None) as anndata does: a null array."""
    meta = _aw_array_meta((), "bool", False,
                          [{"name": "bytes"}, dict(_ANNDATA_ZSTD)],
                          _aw_enc("null", "0.1.0"))
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "zarr.json").write_text(json.dumps(meta, indent=2))


def _write_anndata_v3(cells_dir: Path, adata: Any, table_attrs: dict) -> None:
    """Write *adata* as an anndata-encoded Zarr-v3 group (ngff:regions_table)."""
    root_attrs = _aw_enc("anndata", "0.1.0")
    root_attrs.update(table_attrs)                # SpatialData table-link attrs
    _write_group(cells_dir, root_attrs)

    _aw_write_array(cells_dir / "X", np.asarray(adata.X))
    _aw_write_dataframe(cells_dir / "obs", adata.obs)
    _aw_write_dataframe(cells_dir / "var", adata.var)

    for mname in ("obsm", "obsp", "varm", "varp", "layers"):
        m = getattr(adata, mname, None)
        node = cells_dir / mname
        _aw_write_group(node, "dict", "0.1.0")
        if m:
            for key, value in dict(m).items():
                _aw_write_array(node / str(key), np.asarray(value))

    _aw_write_mapping(cells_dir / "uns", dict(adata.uns))
    _write_null(cells_dir / "raw")


def _build_consolidated(store_path: Path) -> dict:
    """Inline every v3 descendant's metadata into the root consolidated block."""
    metadata: dict = {}
    for zj in sorted(store_path.rglob("zarr.json")):
        if zj.parent == store_path:
            continue                              # skip the root itself
        rel = zj.parent.relative_to(store_path).as_posix()
        metadata[rel] = json.loads(zj.read_text())
    return {"kind": "inline", "must_understand": False, "metadata": metadata}


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def save_project(store_path: str | Path, doc: ProjectDocument) -> None:
    """Write *doc* to a SpatialData (Zarr v3) store at *store_path*.

    An existing store at the path is replaced.
    """
    store_path = Path(store_path)
    if store_path.exists():
        if not (store_path / "zarr.json").exists() and any(store_path.iterdir()):
            raise FileExistsError(
                f"{store_path} exists and is not an Opal Studio / SpatialData store")
        shutil.rmtree(store_path)

    # ── Root group (spatialdata_attrs + Opal-private attrs) ──────────────────
    root_attrs = {
        "spatialdata_attrs": {
            "version": SPATIALDATA_CONTAINER_VERSION,
            "spatialdata_software_version": _software_version(),
        },
        "opal_studio": {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "source_image": doc.source_image,
            "session": doc.session,
        },
    }
    _write_group(store_path, root_attrs)

    # ── Images (derived channels, brightfield) ──────────────────────────────
    if doc.images:
        images_dir = store_path / "images"
        _write_group(images_dir)
        used: set[str] = set()
        for name, payload in doc.images.items():
            data = np.asarray(payload["data"])
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            _write_raster_element(images_dir, _safe_dirname(name, used), name,
                                  data, ("c", "y", "x"),
                                  list(payload.get("channel_labels", [])))

    # ── Labels (cell / region / type masks) ─────────────────────────────────
    if doc.labels:
        labels_dir = store_path / "labels"
        _write_group(labels_dir)
        used = set()
        for name, arr in doc.labels.items():
            arr = np.asarray(arr)
            _write_raster_element(labels_dir, _safe_dirname(name, used), name,
                                  arr, ("y", "x"), None)

    # ── Shapes (vector regions) ─────────────────────────────────────────────
    if doc.shapes:
        shapes_dir = store_path / "shapes"
        _write_group(shapes_dir)
        used = set()
        for name, polygons in doc.shapes.items():
            _write_shapes_element(shapes_dir, _safe_dirname(name, used), name,
                                  polygons)

    # ── Table (per-cell AnnData, ngff:regions_table) ────────────────────────
    if doc.table is not None:
        tables_dir = store_path / "tables"
        _write_group(tables_dir)
        sd = {}
        try:
            sd = dict(doc.table.uns.get("spatialdata_attrs", {}))
        except Exception:
            pass
        table_attrs = {
            "spatialdata-encoding-type": "ngff:regions_table",
            "region": sd.get("region"),
            "region_key": sd.get("region_key"),
            "instance_key": sd.get("instance_key"),
            "version": TABLES_VERSION,
        }
        _write_anndata_v3(tables_dir / "cells", doc.table, table_attrs)

    # ── Auxiliary arrays (clustering state, LUTs) — Opal-private group ───────
    if doc.aux:
        aux_dir = store_path / "opal_aux"
        _write_group(aux_dir)
        used = set()
        for key, arr in doc.aux.items():
            arr = np.asarray(arr)
            orig_shape = list(arr.shape)
            elem = aux_dir / _safe_dirname(key, used)
            _write_v3_array(elem, _to_3d(arr))
            meta_path = elem / "zarr.json"
            meta = json.loads(meta_path.read_text())
            meta["attributes"]["opal_orig_shape"] = orig_shape
            meta["attributes"]["opal_key"] = key
            meta_path.write_text(json.dumps(meta, indent=2))

    # ── Consolidated metadata (root inlines all v3 descendants) ─────────────
    root_meta = json.loads((store_path / "zarr.json").read_text())
    root_meta["consolidated_metadata"] = _build_consolidated(store_path)
    (store_path / "zarr.json").write_text(json.dumps(root_meta, indent=2))


def _to_3d(arr: np.ndarray) -> np.ndarray:
    """Promote a 1-D / 2-D array to (C, Y, X) for the array writer."""
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

    root_meta = json.loads(root_meta_path.read_text())
    attrs = root_meta.get("attributes", {})
    opal = attrs.get("opal_studio", {})

    schema = opal.get("schema_version")
    if schema is not None and schema > PROJECT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Project schema v{schema} is newer than supported "
            f"v{PROJECT_SCHEMA_VERSION}; please update Opal Studio")

    doc = ProjectDocument(
        source_image=opal.get("source_image", {}),
        session=opal.get("session", {}),
    )

    # ── Images ──────────────────────────────────────────────────────────────
    images_dir = store_path / "images"
    if images_dir.is_dir():
        for elem in sorted(images_dir.iterdir()):
            if not (elem / "0" / "zarr.json").exists():
                continue
            key = _read_element_key(elem / "zarr.json", elem.name)
            doc.images[key] = {
                "data": _read_v3_array(elem / "0"),
                "channel_labels": _read_omero_labels(elem / "zarr.json"),
            }

    # ── Labels ──────────────────────────────────────────────────────────────
    labels_dir = store_path / "labels"
    if labels_dir.is_dir():
        for elem in sorted(labels_dir.iterdir()):
            if not (elem / "0" / "zarr.json").exists():
                continue
            key = _read_element_key(elem / "zarr.json", elem.name)
            arr = _read_v3_array(elem / "0")
            doc.labels[key] = arr[0] if arr.ndim == 3 else arr

    # ── Shapes ──────────────────────────────────────────────────────────────
    shapes_dir = store_path / "shapes"
    if shapes_dir.is_dir():
        for elem in sorted(shapes_dir.iterdir()):
            if not (elem / "shapes.parquet").exists():
                continue
            key = _read_element_key(elem / "zarr.json", elem.name)
            doc.shapes[key] = _read_shapes_parquet(elem / "shapes.parquet")

    # ── Auxiliary arrays ────────────────────────────────────────────────────
    aux_dir = store_path / "opal_aux"
    if aux_dir.is_dir():
        for elem in sorted(aux_dir.iterdir()):
            meta_path = elem / "zarr.json"
            if not meta_path.exists():
                continue
            arr = _read_v3_array(elem)
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


def _read_shapes_parquet(path: Path) -> dict:
    """Read a GeoParquet shapes file back into {sid: [ring, ...]}."""
    import geopandas as gpd

    gdf = gpd.read_parquet(path)
    out: dict = {}
    for sid, geom in zip(gdf.index, gdf.geometry):
        if geom is None:
            continue
        if geom.geom_type == "Polygon":
            rings = [[[x, y] for x, y in geom.exterior.coords]]
        elif geom.geom_type == "MultiPolygon":
            rings = [[[x, y] for x, y in g.exterior.coords] for g in geom.geoms]
        else:
            continue
        out[str(sid)] = rings
    return out


def _read_element_key(group_meta_path: Path, fallback: str) -> str:
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
    """Build the ``source_image`` block (absolute + relative paths)."""
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
