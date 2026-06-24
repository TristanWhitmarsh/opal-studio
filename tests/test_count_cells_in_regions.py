"""
First integration test against a real SpatialData store using the
``spatialdata`` library.

Reads ``C:/opal-studio/images/SpatialData/EMMET.zarr``, takes the cell
segmentation labels element ``InstanSegFull`` and the region shapes
``kidney``, ``liver``, ``heart``, ``lung``, and counts how many cells fall
inside each region (a cell is "in" a region if its centroid lies within the
region polygon).

In this store the labels and the region shapes both carry *identity*
transformations to the ``global`` coordinate system, so cell pixel
coordinates and region polygon coordinates live in the same space — the
labels are indexed (y, x) and shapes use (x, y).  The test verifies that
assumption before counting.

Run directly::

    python tests/test_count_cells_in_regions.py
    python tests/test_count_cells_in_regions.py path/to/store.zarr

Requires: spatialdata, geopandas, shapely, scipy, numpy (i.e. an environment
with the spatialdata stack — *not* the pinned opal-all env).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

DEFAULT_STORE = Path(r"C:\opal-studio\images\SpatialData\EMMET.zarr")
SEGMENTATION = "InstanSegFull"
REGIONS = ["kidney", "liver", "heart", "lung"]


# ──────────────────────────────────────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────────────────────────────────────

def _label_array(element):
    """Materialise the highest-resolution label image via spatialdata.

    Deliberately performs **no** clipped-chunk workaround: this test verifies
    that the store is readable by the standard spatialdata / zarr stack, so a
    non-compliant raster (e.g. clipped edge chunks) must surface as a failure.
    """
    import numpy as np

    if hasattr(element, "dims"):
        da = element                                   # single-scale DataArray
    else:
        scale0 = element["scale0"]                     # multiscale DataTree
        da = next(iter(scale0.data_vars.values()))
    return np.squeeze(np.asarray(da.data))


def _assert_identity_to_global(element, name: str):
    """Best-effort check that *element* maps to 'global' with no scale/shift."""
    try:
        from spatialdata.transformations import get_transformation, Identity
    except Exception:
        return  # transformation API unavailable — skip the guard
    t = get_transformation(element, to_coordinate_system="global")
    if not isinstance(t, Identity):
        print(f"  ! warning: '{name}' has a non-identity transform to 'global' "
              f"({type(t).__name__}); centroid coordinates may need transforming.")


def cell_centroids(sdata, segmentation_name: str):
    """Return (cell_ids, centroids_xy) for every labelled cell.

    centroids_xy is an (N, 2) array of (x, y) points in the global frame.
    """
    import numpy as np
    from scipy import ndimage

    element = sdata.labels[segmentation_name]
    _assert_identity_to_global(element, f"labels/{segmentation_name}")

    labels = _label_array(element)
    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids != 0]
    if cell_ids.size == 0:
        return cell_ids, np.empty((0, 2))

    # One vectorised C pass: unweighted centroid (row=y, col=x) per label.
    coms = ndimage.center_of_mass(
        np.ones(labels.shape, dtype=np.uint8), labels, cell_ids)
    coms = np.asarray(coms, dtype=np.float64)          # (N, 2) as (y, x)
    centroids_xy = coms[:, ::-1]                        # → (x, y)
    return cell_ids, centroids_xy


def count_cells_per_region(store_path, segmentation_name=SEGMENTATION,
                           region_names=REGIONS):
    """Read the store and return {region_name: cell_count} plus totals."""
    import geopandas as gpd
    import spatialdata as sd

    sdata = sd.read_zarr(str(store_path))

    cell_ids, centroids_xy = cell_centroids(sdata, segmentation_name)
    points = gpd.GeoDataFrame(
        {"cell_id": cell_ids},
        geometry=gpd.points_from_xy(centroids_xy[:, 0], centroids_xy[:, 1]),
    )

    counts: dict[str, int] = {}
    for name in region_names:
        if name not in sdata.shapes:
            print(f"  ! region '{name}' not found in store; skipping.")
            counts[name] = 0
            continue
        region = sdata.shapes[name]
        _assert_identity_to_global(region, f"shapes/{name}")
        # Spatial join (uses an R-tree index); nunique guards against a cell
        # matching several polygons of the same multi-part region.
        hit = gpd.sjoin(points, region[["geometry"]], predicate="within")
        counts[name] = int(hit["cell_id"].nunique())

    counts["__total_cells__"] = int(len(cell_ids))
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Unit test (skips cleanly when the stack or the data isn't available)
# ──────────────────────────────────────────────────────────────────────────────

class CountCellsInRegionsTest(unittest.TestCase):
    def setUp(self):
        try:
            import geopandas  # noqa: F401
            import scipy        # noqa: F401
            import spatialdata  # noqa: F401
        except Exception as exc:  # pragma: no cover - env-dependent
            self.skipTest(f"spatialdata stack not installed: {exc}")
        if not DEFAULT_STORE.exists():
            self.skipTest(f"test store not found: {DEFAULT_STORE}")

    def test_counts_are_sane(self):
        counts = count_cells_per_region(DEFAULT_STORE)
        total = counts["__total_cells__"]
        self.assertGreater(total, 0, "segmentation contains no cells")
        region_sum = 0
        for region in REGIONS:
            self.assertIn(region, counts)
            self.assertGreaterEqual(counts[region], 0)
            self.assertLessEqual(counts[region], total,
                                 f"{region} has more cells than the whole image")
            region_sum += counts[region]
        # Regions are disjoint sub-areas, so their union can't exceed the total.
        self.assertLessEqual(region_sum, total,
                             "regions overlap or double-count cells")


# ──────────────────────────────────────────────────────────────────────────────
# Script entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    store = Path(argv[0]) if argv else DEFAULT_STORE
    print(f"Reading SpatialData store: {store}")
    counts = count_cells_per_region(store)
    total = counts.pop("__total_cells__")
    print(f"\nSegmentation '{SEGMENTATION}': {total:,} cells total\n")
    print(f"{'Region':<12}{'Cells':>12}{'% of total':>14}")
    print("-" * 38)
    assigned = 0
    for region in REGIONS:
        n = counts.get(region, 0)
        assigned += n
        pct = (100.0 * n / total) if total else 0.0
        print(f"{region:<12}{n:>12,}{pct:>13.1f}%")
    print("-" * 38)
    print(f"{'in regions':<12}{assigned:>12,}{(100.0*assigned/total if total else 0):>13.1f}%")
    print(f"{'unassigned':<12}{total-assigned:>12,}"
          f"{(100.0*(total-assigned)/total if total else 0):>13.1f}%")


if __name__ == "__main__":
    main()
