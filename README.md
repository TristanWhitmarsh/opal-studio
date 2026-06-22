# Opal Studio

<img src="screenshot.jpg" width="100%">

**Opal Studio** is a cross-platform viewer and analysis application for highly multiplexed imaging data, including Imaging Mass Cytometry (IMC), large OME-TIFF files, pyramid TIFF data, and SpatialData/Zarr V3 image directories.

The application combines fast multi-channel image rendering with practical workflows for preprocessing, segmentation, mask refinement, cell positivity, phenotype gating, clustering, and export.

## Quick Start

### Install From Source

```bash
git clone https://github.com/TristanWhitmarsh/opal-studio.git
cd opal-studio
conda create -n opal-env python=3.9
conda activate opal-env
pip install -r requirements.txt
pip install --no-deps -e .
python -m opal_studio
```

You can also launch an installed package with:

```bash
opal-studio
```

### Create A Desktop Launcher

```bash
python -m opal_studio --create-launcher
```

On Windows this creates an `Opal Studio.lnk` shortcut on the desktop. On Linux it creates an `OpalStudio.desktop` launcher.

### University Server / Darkroom Setup

First-time installation, when a new wheel is released:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/scratch.space/envs/opal-env-j4
PIP_REQUIRE_VIRTUALENV=0 pip install opal_studio-0.1.3-py3-none-any.whl
python -m opal_studio --create-launcher
```

Launching after installation:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /storage/scratch.space/envs/opal-env-j4
python -m opal_studio --create-launcher
```

Then double-click the **Opal Studio** icon on the desktop.

## Supported Data

Opal Studio can open:

- OME-TIFF and TIFF image files, including multichannel and pyramidal data.
- SpatialData directories containing Zarr V3 image groups under `images/`.
- RGB-style TIFF series for brightfield/H&E viewing.
- Imported mask and cell label maps from OME-TIFF files.
- Phenotyping definitions from CSV files.

For OME-TIFF, channel names are read from OME metadata when available. For SpatialData, channel names are read from OME-Zarr metadata and can be enriched from `extras/mcd_schema.xml` when present.

## Project Format (SpatialData)

`File > Save Project` writes the whole session as a spec-compliant **SpatialData**
store (Zarr v3, OME-NGFF `0.5-dev-spatialdata`, store format `0.2`). The stores
Opal writes can be opened directly by the `spatialdata` / `squidpy` ecosystem —
Opal does not depend on the `spatialdata` package, it drives the underlying
standards (Zarr v3, GeoParquet, AnnData) directly.

Segmentation masks become `labels`, hand-drawn regions become `shapes`, processed
channels and the generated brightfield become `images`, and the per-cell table
becomes `tables` (linked to the cell mask via `region` / `region_key` /
`instance_key`, so e.g. `obs['region'].value_counts()` gives cells-per-region).

```
project.zarr/
├── zarr.json                      # root group: spatialdata_attrs (v0.2) +
│                                  #   opal_studio {source_image, session} +
│                                  #   consolidated_metadata (flat list of all nodes)
│
├── images/
│   ├── zarr.json                  # group
│   ├── derived/                   # one OME-NGFF image element per key
│   │   ├── zarr.json              #   ome (multiscales; axes c,y,x; omero channels;
│   │   │                          #        coordinateTransformations) + spatialdata_attrs v0.3
│   │   └── 0/                     # scale-0 array (C,Y,X)
│   │       ├── zarr.json
│   │       └── c/ …               # chunk files  (c/<c>/<y>/<x>)
│   └── brightfield/
│       ├── zarr.json
│       └── 0/ { zarr.json, c/… }
│
├── labels/
│   ├── zarr.json                  # group
│   └── Cell Mask/                 # OME-NGFF label element (axes y,x, no channel)
│       ├── zarr.json              #   ome + spatialdata_attrs v0.3
│       └── 0/ { zarr.json, c/… }  # scale-0 array (Y,X)
│
├── shapes/
│   ├── zarr.json                  # group
│   └── Region 1/                  # ngff:shapes element
│       ├── zarr.json              #   encoding-type ngff:shapes, axes, transforms, v0.3
│       └── shapes.parquet         # GeoParquet (WKB polygons) — NOT a zarr node
│
├── tables/
│   ├── zarr.json                  # group
│   └── cells/                     # anndata-encoded Zarr-v3 group (ngff:regions_table v0.2)
│       ├── zarr.json              #   encoding-type anndata + region/region_key/instance_key
│       ├── X/            { zarr.json, c/… }        # array
│       ├── obs/                                    # dataframe
│       │   ├── zarr.json          #   column-order, _index
│       │   ├── _index/ { zarr.json, c/… }          # string-array
│       │   ├── cell_id/ { zarr.json, c/… }         # array
│       │   ├── area_px/ { zarr.json, c/… }         # array
│       │   └── region/                             # categorical
│       │       ├── zarr.json
│       │       ├── categories/ { zarr.json, c/… }  # string-array
│       │       └── codes/      { zarr.json, c/… }  # array
│       ├── var/         { zarr.json, marker/ }     # dataframe
│       ├── obsm/        { zarr.json, spatial/ }    # dict → arrays
│       ├── layers/      { zarr.json, positive/ }   # dict → arrays
│       ├── uns/                                    # dict
│       │   ├── zarr.json
│       │   └── spatialdata_attrs/ { region, region_key, instance_key }  # string scalars
│       ├── obsp/ varm/ varp/  { zarr.json }        # empty dict groups
│       └── raw/         { zarr.json }              # null
│
└── opal_aux/                      # Opal-private (ignored by SpatialData readers)
    ├── zarr.json                  # group
    └── cluster_labels/            # plain Zarr-v3 array (+ opal_key, opal_orig_shape attrs)
        ├── zarr.json
        └── c/ …
```

Notes:

- `images`, `labels`, `shapes`, and `tables` are standard SpatialData elements.
  `opal_aux` and the `opal_studio` root attributes are Opal-private extras that
  SpatialData readers ignore.
- Element directory names are filesystem-sanitized; the true layer name is
  preserved in each element's attributes.
- Only the **root** `zarr.json` carries `consolidated_metadata`; every other group
  omits it (so readers that open a sub-group directly fall back to a disk scan).
- `shapes.parquet` is a regular GeoParquet file, deliberately not part of the zarr
  hierarchy.
- Element names must be unique across element types (a SpatialData requirement).

## Application Layout

The main window has three working areas:

- **Left panel**: layer management for Channels, Masks, Positivity, Types, and Regions.
- **Center tabs**: Image, Phenotyping, Heatmap, t-SNE, and UMAP.
- **Right panel**: collapsible operation sections for Pre-processing, Segmentation, Mask Processing, Cell positivity, and Cell identification.

The status bar shows cursor position and the selected channel value while hovering over the image.

## Core Functionality

### Image Viewing

- Lazy, tiled rendering for large images.
- Pyramid-aware zooming so low-resolution levels are used when zoomed out.
- Background rendering with cached tiles to keep panning and zooming responsive.
- Mouse wheel zoom and left/middle mouse drag panning.
- Multi-channel compositing with per-channel color, alpha, and display limits.
- Global brightness control.
- Mask overlays with opacity and optional vector contours.
- Per-cell positivity overlays and phenotype/cluster type overlays.

### Layer Management

The left panel separates generated and source layers:

- **Channels**: raw and processed image channels. Toggle visibility, change color, adjust alpha, adjust intensity limits, and show/hide all source channels.
- **Masks**: segmentation masks. Toggle raster overlay and contour visibility, adjust mask opacity, and delete generated masks.
- **Positivity**: marker positivity cell layers. Positive and negative cells share the same label map but use a positivity lookup table for display.
- **Types**: phenotype or cluster masks. Adjust shared type opacity and show/hide all type layers.
- **Regions**: hand-drawn polygon regions used for selected-region segmentation.

Generated processed channels, masks, positivity layers, type masks, and regions can be selected from this panel and reused by later steps.

### Region Drawing

Use the **Regions** tab in the left panel to draw analysis regions:

1. Click the draw button.
2. Drag on the Image tab to trace a polygon.
3. Release to create a region layer.
4. Select the region layer before running segmentation in **Selected region** mode.

The simplification control reduces polygon point density. Existing region vertices can be dragged while draw mode is active.

## Recommended Workflow

1. **Open data** with `File > Open Image...` or `File > Open SpatialData...`.
2. **Set up display** in the Channels tab: choose visible markers, colors, alpha, brightness, and intensity limits.
3. **Preprocess channels** if needed: merge markers, remove hot pixels, subtract background, rescale intensity, or create CLAHE-enhanced channels.
4. **Draw regions** if you want to test or restrict analysis to a tissue area.
5. **Run segmentation** on a full image, visible viewport, or selected region.
6. **Refine masks** with size filtering, CellSampler mask fusion, or label expansion.
7. **Call marker positivity** with AI or threshold-based per-cell intensity calls.
8. **Define phenotypes** in the Phenotyping tab using marker positive/negative rules.
9. **Identify cells** to create phenotype type masks, or run clustering to discover unsupervised cell populations.
10. **Inspect analysis views** in Heatmap, t-SNE, and UMAP.
11. **Export results** as OME-TIFF masks/cells, GeoJSON contours, and CSV phenotyping definitions.

## File Menu

| Menu action | Purpose |
| --- | --- |
| `Open Image...` | Open OME-TIFF/TIFF image data. |
| `Open SpatialData...` | Open a SpatialData root directory. |
| `Load Masks...` | Import label masks from OME-TIFF/TIFF. |
| `Load Cells...` | Import cell/positivity label maps from OME-TIFF/TIFF. |
| `Load Phenotyping...` | Import phenotype definitions from CSV. |
| `Save Masks...` | Export mask layers as OME-TIFF. |
| `Save Cells...` | Export cell/positivity layers as OME-TIFF. |
| `Save Contours...` | Export selected mask/cell contours as GeoJSON. |
| `Save Phenotyping...` | Export phenotype definitions as CSV. |

Mask and cell OME-TIFF exports are written as `CYX` data with channel names preserved in OME metadata.

## Pre-processing

Open **Pre-processing** in the right panel.

### Merge

The **Merge** tab averages two selected image channels and creates a new processed channel named from the source pair.

### Filter

The **Filter** tab creates a new processed channel from one source or processed channel. Available filters:

- **Median**: median filtering with a disk footprint.
- **Opening**: morphological opening for small-object/noise suppression.
- **CLAHE**: percentile normalization followed by contrast-limited adaptive histogram equalization.
- **Subtract Background**: Gaussian smoothing plus rolling-ball background subtraction.
- **Remove Hotpixels**: hot-pixel removal with threshold, pass count, and filter size controls.
- **Intensity Rescale**: percentile-based intensity rescaling.

Processed channels appear in the Channels tab and can be used for segmentation, positivity, and clustering.

## Segmentation

Open **Segmentation** in the right panel. Choose a region mode, a target mode, and one segmentation engine.

### Region Modes

- **Full image**: segment the entire image.
- **Visible region**: segment only the current canvas viewport, useful for fast parameter testing.
- **Selected region**: segment inside the selected polygon region. Only detections whose centroids fall inside the polygon are kept.

### Target Modes

- **New mask**: create a new mask layer.
- **Overwrite selected mask**: update an existing selected mask. For region-based overwrite, existing cells in the affected area are removed and new detections are merged back in.

### Engines

| Engine | Inputs and controls | Typical use |
| --- | --- | --- |
| **Watershed** | One channel, Voronoi or Gaussian labeller, spot sigma, outline sigma, threshold, minimum mean intensity. | Fast classical nuclei/cell segmentation and parameter testing. |
| **InstanSeg** | One channel, model name, pixel size, optional hole filling and largest-component cleanup. | Fast learned nuclei/cell segmentation. |
| **Mesmer** | Nuclear channel, optional membrane channel, DeepCell/default or local `.keras` model, nuclear or whole-cell compartment, pixel size, watershed post-processing. | Nuclear or whole-cell segmentation for multiplexed imaging. |
| **StarDist** | One channel, pretrained or local model, probability threshold, NMS threshold. | Nuclear segmentation with star-convex objects. |
| **Cellpose** | One channel, nuclei/cyto/cyto2 or local model, diameter, cell probability threshold, flow threshold. | Flexible cell or nuclei segmentation. |
| **Omnipose** | One channel, specialized Omnipose/custom model, diameter, mask threshold, flow threshold. | Bacteria, elongated objects, plant cells, worms, and other non-round shapes. |

Deep-learning engines run in a separate worker process to reduce TensorFlow/PyTorch conflicts. Local model folders are auto-discovered under `opal_studio/models/<engine>/` when present.

## Mask Processing

Open **Mask Processing** in the right panel.

| Tab | Function |
| --- | --- |
| **Filter** | Remove labels below a minimum area or above a maximum area. |
| **Sampler** | Merge multiple masks with CellSampler/Ubermasking. Strategies include largest cell count, highest Jaccard, and minimum area variance. |
| **Expand** | Expand labels by a chosen number of pixels. Binary Mask mode uses watershed-style separation lines; Label Map mode preserves integer labels with label expansion. |

Expanded binary masks keep their internal label map so threshold positivity and clustering can still operate per cell.

## Cell Positivity

Open **Cell positivity** in the right panel after creating or importing a cell mask.

### AI Positivity

The **AI** tab runs the packaged marker-positivity model against every non-mask image channel. For each channel it creates a Positivity layer that stores positive/negative calls per cell while preserving the original cell label IDs and contours.

### Threshold Positivity

The **Thresholds** tab computes per-cell mean intensity for every image channel:

1. Select a mask.
2. Click **Get Thresholds**.
3. Opal Studio computes per-cell means and an Otsu threshold for each channel.
4. Positivity layers are created immediately for all channels.
5. Use the channel dropdown, numeric threshold field, or threshold slider to adjust a channel interactively.

The count label shows positive cells over total signal-bearing cells for the selected marker.

## Phenotyping And Cell Identification

Use the **Phenotyping** center tab to define cell types:

1. Enter a cell type name and click **Add Cell Type**.
2. Click table cells to cycle marker rules through blank, `Pos`, and `Neg`.
3. Double-click a column header to rename a cell type.
4. Right-click a column header to delete a cell type.

Then open **Cell identification > Gating** and click **Identify Cells**. Opal Studio combines the marker positivity layers with the phenotype table and creates one Type mask per matching cell type. Cells that do not match any defined type are added to an `Unknown` type mask.

Phenotyping definitions can be saved and loaded as CSV files.

## Clustering

Open **Cell identification > Clustering** for unsupervised population discovery.

Inputs and options:

- Select the mask that defines individual cells.
- Choose which image or processed channels to include.
- Choose a normalization method: Yeo-Johnson, arcsinh, log-z, z-score, min-max, or none.
- Optionally enable PCA. If the PCA component count is blank, Opal Studio uses parallel analysis; DBSCAN uses PCA automatically.
- Choose a clustering method: Leiden, Louvain, PhenoGraph, FlowSOM, KMeans, Hierarchical, or DBSCAN.

Outputs:

- Type masks for each cluster, plus a grey Noise mask for DBSCAN noise when present.
- A Heatmap tab showing per-cluster mean channel intensity.
- t-SNE and UMAP plots colored by cluster.
- Clustering metrics including cell count, cluster count, PCA details, silhouette score, Davies-Bouldin index, Calinski-Harabasz index, and cluster sizes.

Cluster names can be edited in the Heatmap tab. Type mask color changes in the left panel are synchronized to the t-SNE and UMAP plots.

## Model Selection Notes

For IMC datasets, start with models trained or tuned for IMC when available. If no custom model is available, a practical workflow is:

1. Test quickly with **Watershed** or **Visible region** mode.
2. Try **InstanSeg** or **StarDist** for nuclei-rich marker channels.
3. Use **Mesmer** when nuclear and membrane/cytoplasm channels are available.
4. Use **Cellpose** or **Omnipose** when object morphology differs from round nuclei.
5. Refine with size filtering, CellSampler, and expansion before positivity or clustering.

Indicative speed from local high-resolution testing:

| Segmentation Engine | Approximate Speed |
| --- | ---: |
| Watershed | 1 sec |
| InstanSeg | 10 sec |
| Mesmer | 26 sec |
| Cellpose | 14 sec |
| Omnipose | 25 sec |
| StarDist | 60 sec |

Typical IMC segmentation quality depends strongly on staining, tissue, resolution, and model weights. In earlier local testing, the rough ordering was:

```text
StarDist > InstanSeg > Mesmer > Cellpose > Watershed > Omnipose
```

Treat this as a starting point rather than a rule.

## Tips

- Use **Visible region** segmentation to tune parameters before running a full image.
- Use **Overwrite selected mask** when iterating on a segmentation to avoid clutter.
- Draw and select a region before using **Selected region** mode.
- Use processed channels as segmentation inputs when raw channels are noisy or low contrast.
- Keep one clean cell label mask for downstream positivity, phenotyping, and clustering.
- Save masks/cells before closing if you want to reuse generated label maps in another session.

## License

Opal Studio is licensed under the **MIT License with the Commons Clause**.

- Free to use for research, internal analysis, and development.
- Free to inspect, modify, and build upon.
- You may not sell Opal Studio or offer it as a paid hosted service, paid software product, or commercial service whose value derives substantially from Opal Studio.

See `LICENSE` for the full license text.
