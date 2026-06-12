"""
Main application window — assembles the three-panel layout.
Handles image loading, coordinate mapping, and background operations.
"""

from __future__ import annotations

import sys
import os
import threading
import multiprocessing
from pathlib import Path
import random
import colorsys
import xml.etree.ElementTree as ET
import numpy as np
import tifffile
from PySide6.QtCore import Qt, Slot, QPoint, QSize, Signal, QPointF, QTimer
from PySide6.QtGui import (
     QImage, QPainter, QPixmap, QColor, QIcon, QPolygonF, QAction
)
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QSplitter, QWidget,
    QHBoxLayout, QVBoxLayout, QStatusBar, QMessageBox, QTabWidget,
    QApplication, QScrollBar, QLabel
)

from opal_studio.channel_model import Channel, ChannelListModel
from opal_studio.image_loader import (
    ImageData, open_image, open_spatialdata_collection,
    spatialdata_channel_maxima, get_tile, _get_yx
)
from opal_studio.widgets.channel_panel import ChannelPanel
from opal_studio.widgets.image_canvas import ImageCanvas
from opal_studio.widgets.operations_panel import OperationsPanel
from opal_studio.widgets.phenotyping_tab import PhenotypingTab
from opal_studio.widgets.clustering_heatmap_tab import ClusteringHeatmapTab
from opal_studio.widgets.scatter_plot_tab import ScatterPlotTab
from opal_studio.widgets.brightfield_view import BrightfieldView

import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
from skimage.morphology import white_tophat, opening, closing, disk

def expand_labels_watershed(labels, expansion_pixels=6):
    """
    Expand labeled regions using watershed, ensuring touching cells are separated.
    """
    # 1. Create the expansion mask
    mask = labels > 0
    distances = distance_transform_edt(~mask)
    expansion_mask = distances <= expansion_pixels

    # 2. Define the Topography
    elevation = distances

    # 3. Create Markers
    boundaries = find_boundaries(labels, mode='inner')
    markers = labels.copy()
    markers[boundaries] = 0
    
    # 3b. Safety: Resurrection for tiny cells
    unique_labels = np.unique(labels)
    unique_markers = np.unique(markers)
    if len(unique_labels) != len(unique_markers):
        lost_cells = np.setdiff1d(unique_labels, unique_markers)
        lost_cells = lost_cells[lost_cells != 0]
        
        if len(lost_cells) > 0:
            for cell_id in lost_cells:
                coords = np.argwhere(labels == cell_id)
                if len(coords) > 0:
                    center = coords.mean(axis=0).astype(int)
                    markers[center[0], center[1]] = cell_id

    # 4. Apply watershed with explicit types for Cython compatibility
    from skimage.segmentation import watershed
    expanded_labels = watershed(
        elevation.astype(np.float32),
        markers=markers.astype(np.int32),
        mask=expansion_mask,
        watershed_line=True
    )
    
    return expanded_labels


def expand_labels_labelmap(labels, expansion_pixels=6):
    """
    Expand labeled regions by simple dilation, preserving each cell's integer label value.
    Touching cells overwrite each other based on proximity (Voronoi-style via distance transform),
    with no watershed separation lines.
    """
    from skimage.segmentation import expand_labels
    return expand_labels(labels, distance=expansion_pixels)


class MainWindow(QMainWindow):
    """Top-level window for Opal Studio."""

    # Signals for cross-thread communication
    segmentationResultReady = Signal(object, str, bool, object, object, str, bool, int, object, bool, object) # labels, name, is_cell_mask, color, contour_data, source_marker, is_type_mask, target_idx, pos_lut, random_colors, aux_labels
    segmentationError = Signal(str)
    segmentationTileReady = Signal(object)  # dict with tile info from child process
    preprocessingResultReady = Signal(str, str, object, float, float) # original_name, suffix, data, min, max
    preprocessingError = Signal(str)
    operationProgress = Signal(int, int)
    operationFinished = Signal(int)
    thresholdMeansReady = Signal(object, object, object, int)  # labels, cell_means dict, all_thresholds dict, mask_model_idx
    clusteringHeatmapReady = Signal(object, object, object)  # cluster_ids, channel_names, heatmap_data
    clusteringDimReductionReady = Signal(object, object, object, object)  # tsne_coords, umap_coords, cluster_labels, cluster_colors
    clusteringMetricsReady = Signal(str)
    clusterTypesIdentified = Signal(object)  # dict[int, str]  cluster_id -> type_name
    brightfieldResultReady = Signal(object)  # (H, W, 3) uint8 ndarray

    def __init__(self):
        super().__init__()
        print("DEBUG: MainWindow initializing...")
        self.setWindowTitle("Opal Studio")
        self.resize(1400, 900)

        # Data model
        self._channel_model = ChannelListModel()
        self._image: ImageData | None = None
        self._spatialdata_collection = None
        self._spatialdata_slice_index = 0
        self._pending_spatialdata_slice = 0

        # Components
        self._channel_panel = ChannelPanel(self._channel_model)
        self._canvas = ImageCanvas(self._channel_model)
        self._ops_panel = OperationsPanel(self._channel_model, self)
        self._phenotyping_tab = PhenotypingTab(self._channel_model)
        self._clustering_heatmap_tab = ClusteringHeatmapTab()
        self._tsne_tab = ScatterPlotTab("t-SNE")
        self._umap_tab = ScatterPlotTab("UMAP")
        self._brightfield_view = BrightfieldView(self._channel_model)
        self._active_cluster_ids: list[int] = []
        self._cluster_cell_ids: "np.ndarray | None" = None
        self._cluster_labels_arr: "np.ndarray | None" = None
        self._cluster_working_labels: "np.ndarray | None" = None

        # Signals
        self._ops_panel.runSegmentationRequested.connect(self._start_segmentation)
        self._ops_panel.runPreprocessingRequested.connect(self._run_preprocessing)
        self._ops_panel.runBrightfieldRequested.connect(self._run_brightfield)
        self.brightfieldResultReady.connect(self._on_brightfield_complete)
        self._ops_panel.runMaskProcessingRequested.connect(self._run_mask_expansion)
        self._ops_panel.runCellPositivityRequested.connect(self._run_cell_positivity)
        self._ops_panel.runCellIdentificationRequested.connect(self._run_cell_identification)
        self._ops_panel.runClusteringRequested.connect(self._run_clustering)
        self._ops_panel.runClusterCellIdentificationRequested.connect(self._run_cluster_cell_identification)
        self._ops_panel.runThresholdComputeRequested.connect(self._run_threshold_compute)
        self._ops_panel.applyThresholdRequested.connect(self._apply_threshold_positivity)
        self.operationProgress.connect(self._ops_panel.set_progress_info)
        self.operationFinished.connect(self._on_operation_complete)
        self.thresholdMeansReady.connect(self._on_threshold_means_ready)
        self.clusteringHeatmapReady.connect(self._on_clustering_heatmap_ready)
        self.clusterTypesIdentified.connect(self._on_cluster_types_identified)
        self.clusteringDimReductionReady.connect(self._on_dim_reduction_ready)
        self.clusteringMetricsReady.connect(self._ops_panel.set_clustering_metrics)
        self._clustering_heatmap_tab.clusterRenamed.connect(self._on_cluster_renamed)
        self._channel_model.dataChanged.connect(self._on_channel_data_changed)
        
        self.segmentationResultReady.connect(self._on_segmentation_complete)
        self.segmentationError.connect(self._on_segmentation_error)
        self.segmentationTileReady.connect(self._on_segmentation_tile_ready)
        self._ops_panel.cancelSegmentationRequested.connect(self._cancel_segmentation)
        self.preprocessingResultReady.connect(self._on_preprocessing_complete)
        self.preprocessingError.connect(self._on_segmentation_error)
        self._canvas.pixelHovered.connect(self._on_pixel_hovered)
        self._segmentation_proc = None
        self._seg_preview_channel_idx = -1
        self._seg_cancelled = False

        # Connect drawing signals between panel and canvas / brightfield view
        self._channel_panel._draw_btn.toggled.connect(self._canvas.set_draw_mode)
        self._channel_panel._draw_btn.toggled.connect(self._brightfield_view.set_draw_mode)
        self._channel_panel._simplification_spin.valueChanged.connect(self._canvas.set_simplification_epsilon)
        self._channel_panel._simplification_spin.valueChanged.connect(self._brightfield_view.set_simplification_epsilon)
        self._canvas.regionDrawn.connect(self._on_region_drawn)
        self._brightfield_view.regionDrawn.connect(self._on_region_drawn)

        # Sync viewport between multiplex canvas and brightfield view
        self._viewport_syncing = False
        self._canvas.viewportChanged.connect(self._sync_canvas_to_brightfield)
        self._brightfield_view.viewportChanged.connect(self._sync_brightfield_to_canvas)

        # Layout
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._channel_panel)
        
        self._center_tabs = QTabWidget()

        self._image_tab = QWidget()
        image_layout = QVBoxLayout(self._image_tab)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)
        image_layout.addWidget(self._canvas, 1)

        self._slice_controls = QWidget()
        slice_layout = QHBoxLayout(self._slice_controls)
        slice_layout.setContentsMargins(8, 4, 8, 4)
        self._slice_label = QLabel("Slice 1 / 1")
        self._slice_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._slice_scrollbar.setRange(1, 1)
        self._slice_scrollbar.setSingleStep(1)
        self._slice_scrollbar.setPageStep(1)
        self._slice_scrollbar.valueChanged.connect(self._on_spatialdata_slice_changed)
        slice_layout.addWidget(self._slice_label)
        slice_layout.addWidget(self._slice_scrollbar, 1)
        self._slice_controls.hide()
        image_layout.addWidget(self._slice_controls)

        self._slice_load_timer = QTimer(self)
        self._slice_load_timer.setSingleShot(True)
        self._slice_load_timer.setInterval(120)
        self._slice_load_timer.timeout.connect(self._load_pending_spatialdata_slice)
        
        # Inject an invisible native icon to force the tab bar to draw taller, preventing cut-off text
        spacer_pixmap = QPixmap(1, 24)
        spacer_pixmap.fill(Qt.GlobalColor.transparent)
        spacer_icon = QIcon(spacer_pixmap)
        
        self._center_tabs.setIconSize(QSize(1, 24))
        self._center_tabs.addTab(self._image_tab, spacer_icon, "Image")
        self._center_tabs.addTab(self._brightfield_view, spacer_icon, "Brightfield")
        self._center_tabs.addTab(self._phenotyping_tab, spacer_icon, "Phenotyping")
        self._center_tabs.addTab(self._clustering_heatmap_tab, spacer_icon, "Heatmap")
        self._center_tabs.addTab(self._tsne_tab, spacer_icon, "t-SNE")
        self._center_tabs.addTab(self._umap_tab, spacer_icon, "UMAP")
        self._splitter.addWidget(self._center_tabs)
        
        self._splitter.addWidget(self._ops_panel)
        
        self._splitter.setStretchFactor(0, 0) # channel
        self._splitter.setStretchFactor(1, 1) # canvas
        self._splitter.setStretchFactor(2, 0) # operations
        self._splitter.setSizes([300, 630, 350])
        self._splitter.setCollapsible(0, False)  # channel panel cannot vanish
        self._splitter.setCollapsible(2, False)  # operations panel cannot vanish
        self._channel_panel.setMinimumWidth(180)
        self._ops_panel.setMinimumWidth(220)
        
        self.setCentralWidget(self._splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._setup_menus()

    def _setup_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        
        open_act = QAction("&Open Image…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        open_sdata_act = QAction("Open &SpatialData…", self)
        open_sdata_act.setShortcut("Ctrl+Shift+O")
        open_sdata_act.triggered.connect(self._on_open_spatialdata)
        file_menu.addAction(open_sdata_act)

        file_menu.addSeparator()

        load_masks_act = QAction("&Load Masks…", self)
        load_masks_act.triggered.connect(lambda: self._on_import_masks(target="mask"))
        file_menu.addAction(load_masks_act)

        load_contours_act = QAction("Load &Contours…", self)
        load_contours_act.triggered.connect(self._on_import_contours)
        file_menu.addAction(load_contours_act)

        load_positivity_act = QAction("Load &Positivity…", self)
        load_positivity_act.triggered.connect(lambda: self._on_import_masks(target="cell"))
        file_menu.addAction(load_positivity_act)

        load_types_act = QAction("Load &Types…", self)
        load_types_act.triggered.connect(self._on_import_types)
        file_menu.addAction(load_types_act)

        load_regions_act = QAction("Load &Regions…", self)
        load_regions_act.triggered.connect(self._on_import_regions)
        file_menu.addAction(load_regions_act)

        load_phenos_act = QAction("Load P&henotyping…", self)
        load_phenos_act.triggered.connect(self._on_import_phenotypes)
        file_menu.addAction(load_phenos_act)

        file_menu.addSeparator()

        save_masks_act = QAction("&Save Masks…", self)
        save_masks_act.triggered.connect(lambda: self._on_export_masks(target="mask"))
        file_menu.addAction(save_masks_act)

        save_contours_act = QAction("Save Con&toursS…", self)
        save_contours_act.triggered.connect(self._on_export_contours)
        file_menu.addAction(save_contours_act)

        save_positivity_act = QAction("Save &Positivity…", self)
        save_positivity_act.triggered.connect(lambda: self._on_export_masks(target="cell"))
        file_menu.addAction(save_positivity_act)

        save_types_act = QAction("Save &Types…", self)
        save_types_act.triggered.connect(self._on_export_types)
        file_menu.addAction(save_types_act)

        save_regions_act = QAction("Save &Regions…", self)
        save_regions_act.triggered.connect(self._on_export_regions)
        file_menu.addAction(save_regions_act)

        save_phenos_act = QAction("Save P&henotyping…", self)
        save_phenos_act.triggered.connect(self._on_export_phenotypes)
        file_menu.addAction(save_phenos_act)

        save_brightfield_act = QAction("Save &Brightfield…", self)
        save_brightfield_act.triggered.connect(self._on_export_brightfield)
        file_menu.addAction(save_brightfield_act)

        save_cells_act = QAction("Save Cell Data…", self)
        save_cells_act.triggered.connect(self._on_export_cells)
        file_menu.addAction(save_cells_act)
        
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.ome.tiff *.tiff *.tif *.png *.jpg);;All files (*)")
        if path: self._load_image(path)

    def _on_open_spatialdata(self):
        """Let the user pick a SpatialData root directory, then load it."""
        path = QFileDialog.getExistingDirectory(self, "Open SpatialData Directory", "")
        if path:
            self._load_spatialdata(path)

    def _load_spatialdata(self, path: str):
        """Open a SpatialData directory and set it as the active image."""
        try:
            collection = open_spatialdata_collection(path)
            img = collection.open_image(0)
            self._image = img
            self._spatialdata_collection = collection
            self._spatialdata_slice_index = 0
            self._pending_spatialdata_slice = 0

            from opal_studio.channel_model import generate_spaced_colors
            palette = generate_spaced_colors(len(img.channel_names))
            
            # Efficiently compute per-channel maxima from coarsest level once
            self._status.showMessage("Computing channel ranges...")
            maxima = spatialdata_channel_maxima(img)
            
            channels = []
            for i, name in enumerate(img.channel_names):
                rgb = palette[i]
                channels.append(Channel(
                    name=name, color=QColor(*rgb),
                    visible=True, 
                    data_min=0.0, 
                    data_max=float(maxima[i]), 
                    index=i
                ))
            self._channel_model.set_channels(channels)

            self._canvas.set_image(img)
            self._ops_panel.reset()
            self._phenotyping_tab.clear()
            self._clustering_heatmap_tab.clear()

            slice_count = len(collection)
            self._slice_scrollbar.blockSignals(True)
            self._slice_scrollbar.setRange(1, slice_count)
            self._slice_scrollbar.setValue(1)
            self._slice_scrollbar.blockSignals(False)
            self._slice_label.setText(
                f"Slice 1 / {slice_count}: {collection.image_names[0]}"
            )
            self._slice_controls.setVisible(slice_count > 1)
            self._status.showMessage(
                f"Loaded SpatialData: {Path(path).name} (slice 1 of {slice_count})",
                5000,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Could not load SpatialData: {e}")

    @Slot(int)
    def _on_spatialdata_slice_changed(self, value: int):
        collection = self._spatialdata_collection
        if collection is None:
            return
        index = value - 1
        if not 0 <= index < len(collection):
            return

        self._pending_spatialdata_slice = index
        self._slice_label.setText(
            f"Slice {value} / {len(collection)}: {collection.image_names[index]}"
        )
        if index == self._spatialdata_slice_index:
            self._slice_load_timer.stop()
            return

        # Coalesce rapid scrollbar changes. Opening a slice below only builds
        # lazy metadata objects; pixel chunks remain on the render thread.
        self._slice_load_timer.start()

    @Slot()
    def _load_pending_spatialdata_slice(self):
        collection = self._spatialdata_collection
        if collection is None:
            return
        index = self._pending_spatialdata_slice
        if index == self._spatialdata_slice_index:
            return

        try:
            img = collection.open_image(index)
            self._image = img
            self._spatialdata_slice_index = index

            from opal_studio.channel_model import generate_spaced_colors
            palette = generate_spaced_colors(len(img.channel_names))
            previous = [
                self._channel_model.channel(i)
                for i in range(min(self._channel_model.rowCount(), len(img.channel_names)))
            ]
            channels = []
            for i, name in enumerate(img.channel_names):
                old = previous[i] if i < len(previous) else None
                if old is not None and old.index == i and not (
                    old.is_mask or old.is_cell_mask or old.is_type_mask
                    or old.is_processed or old.is_region
                ):
                    channels.append(Channel(
                        name=name,
                        color=QColor(old.color),
                        visible=old.visible,
                        selected=old.selected,
                        range_min=old.range_min,
                        range_max=old.range_max,
                        data_min=old.data_min,
                        data_max=old.data_max,
                        index=i,
                        alpha=old.alpha,
                    ))
                else:
                    channels.append(Channel(
                        name=name,
                        color=QColor(*palette[i]),
                        visible=True,
                        data_min=0.0,
                        data_max=1.0,
                        index=i,
                    ))

            self._channel_model.set_channels(channels)
            self._canvas.set_image(img, load_overview=False)
            self._ops_panel.reset()
            self._phenotyping_tab.clear()
            self._clustering_heatmap_tab.clear()
            self._status.showMessage(
                f"Loaded slice {index + 1} of {len(collection)}: "
                f"{collection.image_names[index]}",
                3000,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self._slice_scrollbar.blockSignals(True)
            self._slice_scrollbar.setValue(self._spatialdata_slice_index + 1)
            self._slice_scrollbar.blockSignals(False)
            QMessageBox.critical(self, "Error", f"Could not load SpatialData slice: {e}")

    def _load_image(self, path: str):
        try:
            self._slice_load_timer.stop()
            self._spatialdata_collection = None
            self._slice_controls.hide()
            img = open_image(path)
            self._image = img
            
            if img.is_rgb:
                self._channel_model.set_channels([])
            else:
                from opal_studio.channel_model import generate_spaced_colors
                palette = generate_spaced_colors(len(img.channel_names))
                channels = []
                for i, name in enumerate(img.channel_names):
                    dmin, dmax = self._quick_percentile_range(img, i)
                    rgb = palette[i]
                    channels.append(Channel(
                        name=name, color=QColor(*rgb),
                        visible=True, data_min=float(dmin), data_max=float(dmax), index=i
                    ))
                self._channel_model.set_channels(channels)

            self._canvas.set_image(img)
            self._ops_panel.reset()
            self._phenotyping_tab.clear()
            self._clustering_heatmap_tab.clear()
            self._status.showMessage(f"Loaded: {Path(path).name}", 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Could not load image: {e}")

    def _on_export_masks(self, target="mask"):
        if not self._image: return
        
        # 1. Collect all matching masks
        masks = []
        names = []
        label = "Masks" if target == "mask" else "Cells"
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            # Filter by target
            is_match = False
            if target == "mask" and ch.is_mask and not ch.is_cell_mask:
                is_match = True
            elif target == "cell" and ch.is_cell_mask:
                is_match = True
                
            if is_match and ch.mask_data is not None:
                prefix = "Cell: " if ch.is_cell_mask else "Mask: "
                masks.append(ch.mask_data)
                names.append(prefix + ch.name)
        
        if not masks:
            QMessageBox.information(self, "Export", f"No {label.lower()} found to export.")
            return
            
        # 2. Ask for filename
        path, _ = QFileDialog.getSaveFileName(self, f"Export {label}", "", "OME-TIFF (*.ome.tif *.ome.tiff)")
        if not path: return
        if not path.lower().endswith(".ome.tif") and not path.lower().endswith(".ome.tiff"):
            path += ".ome.tif"
            
        # 3. Write OME-TIFF
        try:
            data = np.stack(masks) # (C, H, W)
            # Ensure uint16 or uint32 if labels are large
            if data.max() < 65535:
                data = data.astype(np.uint16)
            else:
                data = data.astype(np.uint32)
                
            tifffile.imwrite(
                path,
                data,
                ome=True,
                metadata={'axes': 'CYX', 'Channel': {'Name': names}},
                compression='zlib'
            )
            self._status.showMessage(f"Exported {len(masks)} {label.lower()} to {Path(path).name}", 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Could not export {label.lower()}: {e}")

    def _on_import_masks(self, target="mask"):
        label = "Masks" if target == "mask" else "Cells"
        path, _ = QFileDialog.getOpenFileName(self, f"Import {label}", "", "Images (*.ome.tif *.ome.tiff *.tif *.tiff)")
        if not path: return
        
        try:
            with tifffile.TiffFile(path) as tif:
                series = tif.series[0]
                data = series.asarray() # (C, H, W) or (H, W)
                
                # Handle single channel case
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                
                # Extract names from OME-XML
                names = []
                try:
                    ome_xml = tif.ome_metadata
                    if ome_xml:
                        root = ET.fromstring(ome_xml)
                        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                        channels = root.findall(".//ome:Channel", ns)
                        if not channels:
                            channels = root.findall(".//{*}Channel")
                        for ch_xml in channels:
                            names.append(ch_xml.get("Name") or ch_xml.get("ID"))
                except:
                    pass
                
                if len(names) < data.shape[0]:
                    for i in range(len(names), data.shape[0]):
                        names.append(f"Imported Mask {i}")
                
                # Add to model
                imported_count = 0
                for i in range(data.shape[0]):
                    name = names[i]
                    # Determine if it's a cell mask or normal mask
                    is_cell = name.startswith("Cell: ")
                    is_reg_mask = name.startswith("Mask: ")
                    
                    # If prefix is missing, use the target requested by the user
                    final_is_cell = is_cell
                    if not is_cell and not is_reg_mask:
                        final_is_cell = (target == "cell")
                    
                    # Clean up name
                    clean_name = name
                    if is_cell: clean_name = name[len("Cell: "):]
                    elif is_reg_mask: clean_name = name[len("Mask: "):]
                    
                    mask_data = data[i].astype(np.int32)
                    contour_data = self._get_contour_data(mask_data)
                    
                    new_ch = Channel(
                        name=clean_name,
                        color=QColor(255, 255, 255),
                        visible=True,
                        is_mask=(not final_is_cell),
                        is_cell_mask=final_is_cell,
                        mask_data=mask_data,
                        contour_data=contour_data,
                        index=-1
                    )
                    self._channel_model.add_channel(new_ch)
                    imported_count += 1
                
            self._status.showMessage(f"Imported {imported_count} {label.lower()}", 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Import Error", f"Could not import {label.lower()}: {e}")

    def _on_export_phenotypes(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Phenotypes", "", "CSV Files (*.csv)")
        if not path: return
        if not path.lower().endswith(".csv"): path += ".csv"
        
        try:
            self._phenotyping_tab.save_to_csv(path)
            self._status.showMessage(f"Exported phenotypes to {Path(path).name}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not export phenotypes: {e}")

    def _on_import_phenotypes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Phenotypes", "", "CSV Files (*.csv)")
        if not path: return
        
        try:
            self._phenotyping_tab.load_from_csv(path)
            self._status.showMessage(f"Imported phenotypes from {Path(path).name}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Could not import phenotypes: {e}")

    def _on_export_contours(self):
        if not self._image: return
        
        ch = self._channel_model.selected_channel()
        if not ch or not ch.contour_data:
            QMessageBox.warning(self, "Export Contours", "Please select a computed mask or cell channel with contours.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Export Contours", f"{ch.name}_contours.geojson", "GeoJSON (*.geojson)")
        if not path: return
        if not path.endswith(".geojson"): path += ".geojson"
        
        try:
            import json
            features = []
            
            for label_id, data in ch.contour_data.items():
                polygons = data.get("polygons", [])
                if not polygons: continue
                
                coordinates = []
                for poly in polygons:
                    ring = []
                    # QPolygonF contains QPointF points
                    for i in range(poly.count()):
                        pt = poly.at(i)
                        ring.append([pt.x(), pt.y()])
                        
                    # GeoJSON rings must be closed (first and last coordinate identical)
                    if ring and ring[0] != ring[-1]:
                        ring.append(ring[0])
                        
                    if len(ring) >= 4: # Minimum 4 points for a valid closed polygon (triangle: a, b, c, a)
                        coordinates.append(ring)
                        
                if not coordinates: continue
                
                # QuPath standard expects "detection" or "annotation". "detection" is standard for cells.
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "objectType": "detection",
                        "name": f"{label_id}"
                    }
                }
                
                if len(coordinates) > 1:
                    # If multiple disconnected parts, represent as MultiPolygon
                    feature["geometry"]["type"] = "MultiPolygon"
                    feature["geometry"]["coordinates"] = [[ring] for ring in coordinates]
                    
                features.append(feature)
                
            geojson_col = {
                "type": "FeatureCollection",
                "features": features
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(geojson_col, f)
                
            self._status.showMessage(f"Exported contours to {Path(path).name}", 5000)
            
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Could not export contours: {e}")

    def _on_import_contours(self):
        if not self._image: return

        path, _ = QFileDialog.getOpenFileName(self, "Load Contours", "", "GeoJSON (*.geojson);;All files (*)")
        if not path: return

        try:
            import json
            import cv2
            from PySide6.QtGui import QPolygonF
            from opal_studio.image_loader import _get_yx

            with open(path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)

            features = geojson_data.get("features", [])
            if not features:
                QMessageBox.information(self, "Load Contours", "No features found in GeoJSON file.")
                return

            h, w = _get_yx(self._image.base_shape, self._image.axes, self._image.is_rgb)
            labels = np.zeros((h, w), dtype=np.int32)
            contour_data = {}

            label_id = 1
            for feat in features:
                geom = feat.get("geometry", {})
                geom_type = geom.get("type", "")
                coords = geom.get("coordinates", [])

                if geom_type == "Polygon" and coords:
                    rings = [coords[0]]
                elif geom_type == "MultiPolygon" and coords:
                    rings = [sub[0] for sub in coords if sub]
                else:
                    continue

                polygons = []
                all_xs, all_ys = [], []
                for ring_coords in rings:
                    if len(ring_coords) < 3:
                        continue
                    # Coordinates are stored at pixel-centres (+0.5 offset from _get_contour_data).
                    # Subtract 0.5 before rounding so each centre maps back to its origin pixel.
                    pts_np = np.array([[c[0] - 0.5, c[1] - 0.5] for c in ring_coords], dtype=np.float64)
                    pts_np = np.round(pts_np).astype(np.int32)
                    pts_np[:, 0] = np.clip(pts_np[:, 0], 0, w - 1)
                    pts_np[:, 1] = np.clip(pts_np[:, 1], 0, h - 1)
                    cv2.fillPoly(labels, [pts_np], label_id)

                    points = [QPointF(c[0], c[1]) for c in ring_coords]
                    # Ensure closed ring in QPolygonF (consistent with _get_contour_data)
                    if points[0].x() != points[-1].x() or points[0].y() != points[-1].y():
                        points.append(points[0])
                    polygons.append(QPolygonF(points))
                    all_xs.extend(c[0] for c in ring_coords)
                    all_ys.extend(c[1] for c in ring_coords)

                if polygons:
                    contour_data[label_id] = {
                        "polygons": polygons,
                        "bbox": [min(all_ys), min(all_xs), max(all_ys), max(all_xs)]
                    }
                    label_id += 1

            if label_id == 1:
                QMessageBox.information(self, "Load Contours", "No valid polygons found in the GeoJSON file.")
                return

            name = self._channel_model.get_unique_name(Path(path).stem)
            new_ch = Channel(
                name=name,
                color=QColor(255, 255, 255),
                visible=True,
                is_mask=True,
                mask_data=labels,
                contour_data=contour_data,
                index=-1
            )
            self._channel_model.add_channel(new_ch)
            self._status.showMessage(f"Loaded {label_id - 1} contours as mask from {Path(path).name}", 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Import Error", f"Could not load contours: {e}")

    def _on_export_regions(self):
        if not self._image: return

        regions = []
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if getattr(ch, 'is_region', False) and ch.contour_data:
                regions.append(ch)

        if not regions:
            QMessageBox.information(self, "Export", "No regions found to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Regions", "regions.geojson", "GeoJSON (*.geojson)")
        if not path: return
        if not path.endswith(".geojson"): path += ".geojson"

        try:
            import json
            features = []

            for ch in regions:
                for label_id, data in ch.contour_data.items():
                    polygons = data.get("polygons", [])
                    if not polygons: continue

                    coordinates = []
                    for poly in polygons:
                        ring = []
                        for j in range(poly.count()):
                            pt = poly.at(j)
                            ring.append([pt.x(), pt.y()])
                        # GeoJSON rings must be explicitly closed (first coord == last coord).
                        # Strip any existing closing duplicate first, then re-append, to avoid
                        # a double-closing point when the stored QPolygonF is already closed.
                        if len(ring) >= 2 and ring[0][0] == ring[-1][0] and ring[0][1] == ring[-1][1]:
                            ring = ring[:-1]
                        if len(ring) < 3:
                            continue
                        ring.append(ring[0])  # explicit closure
                        coordinates.append(ring)

                    if not coordinates: continue

                    feature = {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": coordinates},
                        "properties": {"objectType": "annotation", "name": ch.name}
                    }
                    if len(coordinates) > 1:
                        feature["geometry"]["type"] = "MultiPolygon"
                        feature["geometry"]["coordinates"] = [[ring] for ring in coordinates]
                    features.append(feature)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump({"type": "FeatureCollection", "features": features}, f)

            self._status.showMessage(f"Saved {len(features)} regions to {Path(path).name}", 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Could not save regions: {e}")

    def _on_import_regions(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Regions", "", "GeoJSON (*.geojson);;All files (*)")
        if not path: return

        try:
            import json
            from PySide6.QtGui import QPolygonF
            from opal_studio.channel_model import generate_spaced_colors

            with open(path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)

            features = geojson_data.get("features", [])
            if not features:
                QMessageBox.information(self, "Load Regions", "No features found in GeoJSON file.")
                return

            existing_regions = sum(
                1 for i in range(self._channel_model.rowCount())
                if getattr(self._channel_model.channel(i), 'is_region', False)
            )
            colors = generate_spaced_colors(len(features) + existing_regions + 1)

            imported_count = 0
            for feat in features:
                geom = feat.get("geometry", {})
                props = feat.get("properties", {})
                name = props.get("name", f"Region {existing_regions + imported_count + 1}")
                geom_type = geom.get("type", "")
                coords = geom.get("coordinates", [])

                if geom_type == "Polygon" and coords:
                    ring_coords = coords[0]
                elif geom_type == "MultiPolygon" and coords:
                    ring_coords = coords[0][0]
                else:
                    continue

                points = [QPointF(c[0], c[1]) for c in ring_coords]
                # Normalise: strip any existing closing duplicate, then always re-append it.
                # This matches the format that _simplify_contour produces for drawn regions,
                # so drawPolyline renders a visually closed shape.
                if len(points) >= 2 and points[0].x() == points[-1].x() and points[0].y() == points[-1].y():
                    points = points[:-1]
                if len(points) < 3:
                    continue
                points.append(points[0])  # explicit closure — first == last

                qpoly = QPolygonF(points)
                xs = [pt.x() for pt in points]
                ys = [pt.y() for pt in points]
                contour_data = {1: {"polygons": [qpoly], "bbox": [min(ys), min(xs), max(ys), max(xs)]}}

                rgb = colors[(existing_regions + imported_count) % len(colors)]
                new_ch = Channel(
                    name=self._channel_model.get_unique_name(name),
                    color=QColor(*rgb),
                    visible=True,
                    is_region=True,
                    contour_data=contour_data,
                    index=-1
                )
                self._channel_model.add_channel(new_ch)
                imported_count += 1

            self._status.showMessage(f"Loaded {imported_count} regions from {Path(path).name}", 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Import Error", f"Could not load regions: {e}")

    def _on_export_types(self):
        if not self._image: return

        masks = []
        names = []
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_type_mask and ch.mask_data is not None:
                masks.append(ch.mask_data)
                names.append("Type: " + ch.name)

        if not masks:
            QMessageBox.information(self, "Export", "No cell type masks found to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Types", "", "OME-TIFF (*.ome.tif *.ome.tiff)")
        if not path: return
        if not path.lower().endswith(".ome.tif") and not path.lower().endswith(".ome.tiff"):
            path += ".ome.tif"

        try:
            data = np.stack(masks).astype(np.uint8)
            tifffile.imwrite(
                path,
                data,
                ome=True,
                metadata={'axes': 'CYX', 'Channel': {'Name': names}},
                compression='zlib'
            )
            self._status.showMessage(f"Saved {len(masks)} type masks to {Path(path).name}", 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Could not save types: {e}")

    def _on_import_types(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Types", "", "Images (*.ome.tif *.ome.tiff *.tif *.tiff)")
        if not path: return

        try:
            from opal_studio.channel_model import generate_spaced_colors

            with tifffile.TiffFile(path) as tif:
                data = tif.series[0].asarray()
                if data.ndim == 2:
                    data = data[np.newaxis, ...]

                names = []
                try:
                    ome_xml = tif.ome_metadata
                    if ome_xml:
                        root = ET.fromstring(ome_xml)
                        channels = root.findall(".//ome:Channel", {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"})
                        if not channels:
                            channels = root.findall(".//{*}Channel")
                        for ch_xml in channels:
                            names.append(ch_xml.get("Name") or ch_xml.get("ID"))
                except Exception:
                    pass

                for i in range(len(names), data.shape[0]):
                    names.append(f"Type {i}")

            colors = generate_spaced_colors(data.shape[0])

            imported_count = 0
            for i in range(data.shape[0]):
                name = names[i]
                clean_name = name[len("Type: "):] if name.startswith("Type: ") else name
                new_ch = Channel(
                    name=clean_name,
                    color=QColor(*colors[i % len(colors)]),
                    visible=True,
                    is_type_mask=True,
                    mask_data=data[i].astype(np.int32),
                    index=-1
                )
                self._channel_model.add_channel(new_ch)
                imported_count += 1

            self._status.showMessage(f"Loaded {imported_count} type masks from {Path(path).name}", 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Import Error", f"Could not load types: {e}")

    @staticmethod
    def _quick_percentile_range(img: ImageData, channel: int) -> tuple[float, float]:
        if not img.levels: return 0.0, 1.0
        coarsest = img.levels[-1]
        h, w = _get_yx(coarsest.shape, img.axes, img.is_rgb)
        try:
            data = get_tile(img, coarsest.index, channel, slice(0, h), slice(0, w))
            low, high = np.percentile(data, (5, 95))
            if high <= low: return float(np.min(data)), float(np.max(data))
            return float(low), float(high)
        except: return 0.0, 1.0

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _run_preprocessing(self, params: dict):
        if not self._image: return
        def _run():
            try:
                if params.get("is_merge"):
                    idx1 = params["channel1_index"]
                    idx2 = params["channel2_index"]
                    ch1 = self._channel_model.channel(idx1)
                    ch2 = self._channel_model.channel(idx2)
                    
                    if ch1.is_processed and ch1.processed_data is not None:
                        data1 = ch1.processed_data.astype(np.float32)
                    else:
                        data1 = self._image.get_full_channel_data(ch1.index, level=0).astype(np.float32)
                        
                    if ch2.is_processed and ch2.processed_data is not None:
                        data2 = ch2.processed_data.astype(np.float32)
                    else:
                        data2 = self._image.get_full_channel_data(ch2.index, level=0).astype(np.float32)
                        
                    x = (data1 + data2) / 2.0
                    suffix = "_Merge"
                    original_name = f"{ch1.name}_{ch2.name}"
                else:
                    idx = params["channel_index"]
                    ch = self._channel_model.channel(idx)
                    original_name = ch.name
                    
                    # Check if we should use memory-cached data or disk data
                    if ch.is_processed and ch.processed_data is not None:
                        data = ch.processed_data.astype(np.float32)
                    else:
                        data = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)
                    
                    if params.get("is_filter"):
                        filter_type = params["filter_type"]
                        
                        if filter_type == 'median':
                            filter_value = params["filter_value"]
                            suffix = "_Median"
                            # Use a disk/elliptical footprint for smoother results
                            footprint = disk(max(1, filter_value // 2))
                            x = ndi.median_filter(data, footprint=footprint, mode='reflect')
                        elif filter_type == 'opening':
                            filter_value = params["filter_value"]
                            suffix = "_Open"
                            # The user expects this filter to perform noise removal (subtraction).
                            # Mathematically, this is an Opening operation.
                            x = opening(data, footprint=disk(max(1, filter_value // 2)))
                        elif filter_type == 'clahe':
                            from skimage import exposure
                            suffix = "_Equal"
                            
                            # 1. Get raw values at percentiles for rescaling
                            p_low_val = np.percentile(data, params["p_low"])
                            p_high_val = np.percentile(data, params["p_high"])
                            diff = p_high_val - p_low_val if p_high_val > p_low_val else 1.0
                            
                            # 2. Normalize to [0, 1]
                            x = (data - p_low_val) / diff
                            x = np.clip(x, 0.0, 1.0)
                            
                            # 3. Apply CLAHE
                            if params["apply_clahe"]:
                                x = exposure.equalize_adapthist(x, kernel_size=params["clahe_kernel"], clip_limit=params["clahe_clip"]).astype(np.float32)
                            
                            # 4. Rescale back to original intensity range
                            x = x * diff + p_low_val
                        elif filter_type == 'subtract background':
                            from scipy.ndimage import gaussian_filter
                            from skimage.restoration import rolling_ball
                            suffix = "_BkgSub"
                            sigma = params["sigma"]
                            radius = params["radius"]
                            image_gauss = gaussian_filter(data, sigma)
                            bkg = rolling_ball(image_gauss, radius=radius)
                            x = data - bkg
                        elif filter_type == 'remove hotpixels':
                            from opal_studio.remove_hotpixels import run as remove_hotpixels_run
                            x = remove_hotpixels_run(
                                data,
                                threshold=params["threshold"],
                                npass=params["npass"],
                                filter_size=params["filter_size"],
                            )
                            suffix = "_Hotpix"
                        elif filter_type == 'intensity rescale':
                            from skimage import exposure
                            p1 = params["p1"]
                            p2 = params["p2"]
                            p2_val, p98_val = np.percentile(data, (p1, p2))
                            x = exposure.rescale_intensity(data, in_range=(p2_val, p98_val))
                            suffix = "_Rescale"
                        else:
                            suffix = f"_{filter_type.capitalize()}"
                            x = data
                    else:
                        from csbdeep.utils import normalize
                        from skimage import exposure
                        suffix = "_Equal"
                        
                        # 1. Get raw values at percentiles for rescaling
                        p_low_val = np.percentile(data, params["p_low"])
                        p_high_val = np.percentile(data, params["p_high"])
                        diff = p_high_val - p_low_val if p_high_val > p_low_val else 1.0
                        
                        # 2. Normalize to [0, 1]
                        x = (data - p_low_val) / diff
                        x = np.clip(x, 0.0, 1.0)
                        
                        # 3. Apply CLAHE
                        if params["apply_clahe"]:
                            x = exposure.equalize_adapthist(x, kernel_size=params["clahe_kernel"], clip_limit=params["clahe_clip"]).astype(np.float32)
                        
                        # 4. Rescale back to original intensity range
                        x = x * diff + p_low_val
                
                # Use full range for pre-processed image rendering
                data_min, data_max = float(np.min(x)), float(np.max(x))
                
                self.preprocessingResultReady.emit(original_name, suffix, x, data_min, data_max)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.preprocessingError.emit(str(e))
        threading.Thread(target=_run, daemon=True).start()

    def _on_preprocessing_complete(self, original_name, suffix, data, data_min, data_max):
        self._ops_panel.stop_loading()
        new_name = self._channel_model.get_unique_name(f"{original_name}{suffix}", always_suffix=True)
        new_ch = Channel(
            name=new_name, color=QColor(255, 255, 255), visible=True,
            data_min=data_min, data_max=data_max,
            is_processed=True, processed_data=data, index=-1
        )
        self._channel_model.add_channel(new_ch)
        self._status.showMessage(f"Generated: {new_name}", 3000)

    # ------------------------------------------------------------------
    # Brightfield generation
    # ------------------------------------------------------------------

    def _run_brightfield(self, params: dict):
        if not self._image: return

        def _run():
            try:
                import json
                from multiplex2brightfield import convert
                from multiplex2brightfield.configuration_presets import GetConfiguration

                config_json = params.get("config_json")
                if config_json:
                    try:
                        config = json.loads(config_json)
                    except json.JSONDecodeError as e:
                        self.segmentationError.emit(f"Invalid config JSON: {e}")
                        return
                else:
                    preset = params.get("preset", "H&E")
                    config = GetConfiguration(preset)

                channel_names = []
                channels_data = []
                for i in range(self._channel_model.rowCount()):
                    ch = self._channel_model.channel(i)
                    if ch.is_mask or ch.is_cell_mask or ch.is_type_mask or getattr(ch, 'is_region', False):
                        continue
                    if ch.is_processed and ch.processed_data is not None:
                        data = ch.processed_data.astype(np.float32)
                    else:
                        data = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)
                    channels_data.append(data)
                    channel_names.append(ch.name)

                if not channels_data:
                    self.segmentationError.emit("No image channels found for brightfield generation.")
                    return

                # convert expects (n_slices, n_channels, H, W); single slice = (1, C, H, W)
                imc_image = np.stack(channels_data, axis=0)[np.newaxis]

                result = convert(
                    imc_image,
                    output_filename=None,
                    channel_names=channel_names,
                    config=config,
                    AI_enhancement=False,
                )
                # result: (1, 3, H, W) -> (H, W, 3) uint8
                rgb = np.transpose(result[0], (1, 2, 0)).astype(np.uint8)
                self.brightfieldResultReady.emit(rgb)

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    @Slot(object)
    def _on_brightfield_complete(self, rgb_array):
        self._brightfield_rgb = rgb_array
        self._ops_panel.stop_loading()
        self._brightfield_view.set_image(rgb_array)
        for i in range(self._center_tabs.count()):
            if self._center_tabs.tabText(i) == "Brightfield":
                self._center_tabs.setCurrentIndex(i)
                break
        self._status.showMessage("Brightfield image generated", 3000)

    def _on_export_brightfield(self):
        rgb = getattr(self, "_brightfield_rgb", None)
        if rgb is None:
            QMessageBox.information(self, "Export", "No brightfield image has been generated yet.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Brightfield", "", "OME-TIFF (*.ome.tif *.ome.tiff)")
        if not path:
            return
        if not path.lower().endswith(".ome.tif") and not path.lower().endswith(".ome.tiff"):
            path += ".ome.tif"

        try:
            tifffile.imwrite(
                path,
                rgb,
                ome=True,
                photometric="rgb",
                metadata={"axes": "YXS"},
            )
            self._status.showMessage(f"Brightfield saved to {Path(path).name}", 5000)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Could not save brightfield: {e}")

    # ------------------------------------------------------------------
    # Cell data export
    # ------------------------------------------------------------------

    def _build_cell_export_dataframe(self):
        """Collect all per-cell data into a pandas DataFrame plus a metadata dict.

        Returns (df, metadata) or (None, {}) if no cell label map is found.
        """
        import pandas as pd
        from skimage.measure import regionprops_table

        # Resolve label map: prefer the one used for clustering, else any cell mask
        working_labels = self._cluster_working_labels
        if working_labels is None:
            for i in range(self._channel_model.rowCount()):
                ch = self._channel_model.channel(i)
                if ch.is_cell_mask and ch.mask_data is not None:
                    lm = ch.mask_data
                    if lm.max() <= 1:
                        from skimage.measure import label as _label
                        lm = _label(lm).astype(np.int32)
                    working_labels = lm.astype(np.int32)
                    break

        if working_labels is None or working_labels.max() == 0:
            return None, {}

        # Cell IDs, centroids, areas
        props = regionprops_table(
            working_labels, properties=("label", "centroid", "area")
        )
        cell_ids = props["label"]           # actual label values
        cx = props["centroid-1"]            # column → x
        cy = props["centroid-0"]            # row    → y
        areas = props["area"]

        pixel_size = self._ops_panel.get_pixel_size()

        data: dict = {
            "cell_id": cell_ids.astype(np.int32),
            "x_px":    cx.astype(np.float32),
            "y_px":    cy.astype(np.float32),
            "area_px": areas.astype(np.int32),
        }
        if pixel_size > 0:
            data["x_um"]     = (cx    * pixel_size     ).astype(np.float32)
            data["y_um"]     = (cy    * pixel_size     ).astype(np.float32)
            data["area_um2"] = (areas * pixel_size ** 2).astype(np.float32)

        df = pd.DataFrame(data)

        # Cluster assignments
        if self._cluster_cell_ids is not None and self._cluster_labels_arr is not None:
            max_label = int(working_labels.max())
            id_to_cluster = np.full(max_label + 1, -1, dtype=np.int32)
            for j, cid in enumerate(self._cluster_cell_ids):
                if cid <= max_label:
                    id_to_cluster[cid] = self._cluster_labels_arr[j]
            cluster_ids = id_to_cluster[cell_ids]
            df["cluster_id"] = cluster_ids
            names_map = self._clustering_heatmap_tab.get_cluster_names()
            df["cluster_name"] = pd.Categorical([
                names_map.get(int(c), f"Cluster {c}") if c >= 0 else "Noise"
                for c in cluster_ids
            ])

        # Phenotypes from type-mask channels
        type_masks: list[tuple[str, np.ndarray]] = []
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_type_mask and ch.mask_data is not None:
                type_masks.append((ch.name, ch.mask_data))
        if type_masks:
            cell_phenotype: dict[int, str] = {}
            for type_name, type_data in type_masks:
                region_ids = np.unique(working_labels[type_data > 0])
                for cid in region_ids[region_ids > 0]:
                    cell_phenotype[int(cid)] = type_name
            df["phenotype"] = pd.Categorical([
                cell_phenotype.get(int(c), "Unknown") for c in cell_ids
            ])

        # Per-marker means and positivity, interleaved per marker
        thresh_tab = self._ops_panel._thresh_tab
        pos_luts: dict[str, np.ndarray] = {}
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_cell_mask and ch.pos_lut is not None:
                key = ch.source_marker if ch.source_marker else ch.name
                pos_luts[key] = ch.pos_lut

        marker_names: list[str] = []
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_mask or ch.is_cell_mask or ch.is_type_mask or getattr(ch, "is_region", False):
                continue
            has_mean = i in thresh_tab._cell_means
            has_pos  = ch.name in pos_luts
            if not has_mean and not has_pos:
                continue
            marker_names.append(ch.name)
            if has_mean:
                lut = thresh_tab._cell_means[i]
                safe = np.clip(cell_ids, 0, len(lut) - 1)
                df[f"{ch.name}_mean"] = lut[safe].astype(np.float32)
            if has_pos:
                lut = pos_luts[ch.name]
                safe = np.clip(cell_ids, 0, len(lut) - 1)
                df[f"{ch.name}_positive"] = (lut[safe] == 2)

        thresholds: dict[str, float] = {}
        for k, v in thresh_tab._thresholds.items():
            try:
                thresholds[self._channel_model.channel(k).name] = float(v)
            except Exception:
                pass

        metadata = {
            "pixel_size_um": float(pixel_size),
            "n_cells": len(cell_ids),
            "marker_names": marker_names,
            "thresholds": thresholds,
        }
        return df, metadata

    def _on_export_cells(self):
        df, metadata = self._build_cell_export_dataframe()
        if df is None:
            QMessageBox.information(self, "Export",
                "No cell data available. Run segmentation and cell identification first.")
            return

        parquet_filter = "Parquet (*.parquet)"
        h5ad_filter    = "AnnData (*.h5ad)"
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Cell Data", "", f"{parquet_filter};;{h5ad_filter}")
        if not path:
            return

        try:
            if selected_filter == h5ad_filter or path.lower().endswith(".h5ad"):
                self._save_cells_h5ad(path, df, metadata)
            else:
                if not path.lower().endswith(".parquet"):
                    path += ".parquet"
                df.to_parquet(path, index=False)
                self._status.showMessage(
                    f"Exported {len(df)} cells to {Path(path).name}", 5000)
        except ImportError as e:
            QMessageBox.critical(self, "Missing Dependency", str(e))
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Could not export cell data: {e}")

    def _save_cells_h5ad(self, path: str, df, metadata: dict):
        import anndata as ad
        import pandas as pd
        import datetime

        if not path.lower().endswith(".h5ad"):
            path += ".h5ad"

        marker_names = metadata.get("marker_names", [])
        pixel_size   = metadata.get("pixel_size_um", 1.0)

        mean_cols = [f"{m}_mean" for m in marker_names if f"{m}_mean" in df.columns]
        var_names = [c[:-5] for c in mean_cols]
        X = df[mean_cols].values.astype(np.float32) if mean_cols else np.zeros((len(df), 0), dtype=np.float32)

        obs_keys = ["cell_id", "x_px", "y_px", "area_px"]
        for k in ("x_um", "y_um", "area_um2", "phenotype", "cluster_id", "cluster_name"):
            if k in df.columns:
                obs_keys.append(k)
        obs = df[obs_keys].copy()
        obs.index = obs["cell_id"].astype(str)

        var = pd.DataFrame(index=pd.Index(var_names, name="marker"))

        adata = ad.AnnData(X=X, obs=obs, var=var)

        if mean_cols:
            adata.layers["intensity"] = X
        pos_cols = [f"{m}_positive" for m in var_names if f"{m}_positive" in df.columns]
        if pos_cols and len(pos_cols) == len(var_names):
            adata.layers["positive"] = df[pos_cols].values.astype(bool)

        xy_cols = ("x_um", "y_um") if "x_um" in df.columns else ("x_px", "y_px")
        adata.obsm["spatial"] = df[list(xy_cols)].values.astype(np.float32)

        adata.uns["opal_studio"] = {
            "pixel_size_um": pixel_size,
            "thresholds":    metadata.get("thresholds", {}),
            "export_date":   datetime.datetime.now().isoformat(),
        }

        adata.write_h5ad(path)
        self._status.showMessage(
            f"Exported {len(df)} cells to {Path(path).name}", 5000)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _start_segmentation(self, params: dict):
        if not self._image: return
        if hasattr(self, "_segmentation_thread") and self._segmentation_thread.is_alive():
            QMessageBox.warning(self, "Busy", "A task is already running.")
            return

        def _run():
            try:
                method = params.get("method", "stardist")
                indices = params["channel_indices"]
                region_mode = params.get("region_mode", "full")
                target_mode = params.get("target_mode", "new")
                target_mask_index = params.get("target_mask_index", None)
                method_names = {
                    "stardist": "StarDist",
                    "cellpose": "Cellpose",
                    "omnipose": "Omnipose",
                    "instanseg": "InstanSeg",
                    "watershed": "Watershed",
                    "mesmer": "Mesmer"
                }
                override_name = method_names.get(method, "Mask")
                x = None
                contour_data = None
                input_channels_data = []
                print(f"[Segmentation] Model: {method} | Region: {region_mode} | Target: {target_mode}")
                
                v_top, v_bottom, v_left, v_right = 0, 0, 0, 0
                full_shape = None

                # ---- Selected region: resolve bounding box and polygon --------
                region_polygon = None   # QPolygonF
                r_top = r_bottom = r_left = r_right = 0
                if region_mode == "selected_region":
                    region_ch_idx = params.get("region_channel_index")
                    if region_ch_idx is None:
                        raise ValueError("No region channel index provided for 'selected_region' mode.")
                    region_ch = self._channel_model.channel(region_ch_idx)
                    if not region_ch.contour_data:
                        raise ValueError("The selected region has no polygon data.")
                    # Region channels store their polygon in contour_data[1]["polygons"][0]
                    first_entry = region_ch.contour_data.get(1, {})
                    polys = first_entry.get("polygons", [])
                    if not polys:
                        raise ValueError("The selected region polygon is empty.")
                    region_polygon = polys[0]
                    bbox = first_entry.get("bbox")  # [y0, x0, y1, x1]
                    if bbox:
                        r_top, r_left, r_bottom, r_right = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    else:
                        xs = [region_polygon.at(i).x() for i in range(region_polygon.count())]
                        ys = [region_polygon.at(i).y() for i in range(region_polygon.count())]
                        r_top, r_bottom = int(min(ys)), int(max(ys))
                        r_left, r_right = int(min(xs)), int(max(xs))
                
                for idx in indices:
                    ch = self._channel_model.channel(idx)
                    if ch.is_processed and ch.processed_data is not None:
                        raw = ch.processed_data.astype(np.float32)
                    else:
                        raw = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)
                    
                    full_shape = raw.shape
                    if region_mode == "visible":
                        v_top = int(max(0, self._canvas._viewport.top()))
                        v_left = int(max(0, self._canvas._viewport.left()))
                        v_bottom = int(min(full_shape[0], self._canvas._viewport.bottom()))
                        v_right = int(min(full_shape[1], self._canvas._viewport.right()))
                        if v_top >= v_bottom or v_left >= v_right:
                            raise ValueError("Visible region is completely outside the image.")
                        raw = raw[v_top:v_bottom, v_left:v_right]
                    elif region_mode == "selected_region":
                        r_top = int(max(0, r_top))
                        r_left = int(max(0, r_left))
                        r_bottom = int(min(full_shape[0], r_bottom))
                        r_right = int(min(full_shape[1], r_right))
                        if r_top >= r_bottom or r_left >= r_right:
                            raise ValueError("Selected region bounding box is outside the image.")
                        raw = raw[r_top:r_bottom, r_left:r_right]

                    # Normalize for deep learning models to prevent NMS hangs or junk results
                    # Subsample for percentile calculation if image is large
                    if raw.size > 1000000:
                        subsample = raw[::int(np.sqrt(raw.size/1000000)), ::int(np.sqrt(raw.size/1000000))]
                        p = np.percentile(subsample, (1, 99.8))
                    else:
                        p = np.percentile(raw, (1, 99.8))
                    data = np.clip((raw - p[0]) / (p[1] - p[0] + 1e-6), 0, 1)
                    input_channels_data.append(data)
                    x = data if x is None else x + data
                
                def _cell_centroid_in_polygon(labels_crop, poly, offset_y, offset_x):
                    """
                    Keep only cells whose centroid (in full-image coords) lies inside `poly`.
                    Returns a filtered label array (same shape as labels_crop).
                    """
                    from scipy.ndimage import find_objects
                    from PySide6.QtCore import QPointF, Qt

                    cell_ids = np.unique(labels_crop)
                    cell_ids = cell_ids[cell_ids > 0]
                    if len(cell_ids) == 0:
                        return labels_crop

                    locs = find_objects(labels_crop)
                    ids_to_keep = []
                    for cell_id in cell_ids:
                        loc = locs[cell_id - 1]
                        if loc is None:
                            continue
                        binary = (labels_crop[loc] == cell_id)
                        ys, xs = np.where(binary)
                        if len(ys) == 0:
                            continue
                        cy = float(np.mean(ys)) + loc[0].start + offset_y
                        cx = float(np.mean(xs)) + loc[1].start + offset_x
                        if poly.containsPoint(QPointF(cx, cy), Qt.FillRule.OddEvenFill):
                            ids_to_keep.append(cell_id)

                    if len(ids_to_keep) == len(cell_ids):
                        return labels_crop
                    keep_set = set(ids_to_keep)
                    mask_keep = np.vectorize(lambda v: v if v in keep_set else 0)(labels_crop).astype(labels_crop.dtype)
                    return mask_keep

                def process_and_emit(out_labels, out_name, out_is_cell):
                    if out_labels is None: return
                    target_idx = -1
                    if target_mode == "overwrite" and target_mask_index is not None:
                        tgt_ch = self._channel_model.channel(target_mask_index)
                        if getattr(tgt_ch, 'is_cell_mask', False) == out_is_cell:
                            target_idx = target_mask_index

                    existing_labels = None
                    if target_idx != -1:
                        target_ch = self._channel_model.channel(target_idx)
                        existing_labels = target_ch.mask_data.copy() if target_ch.mask_data is not None else None

                    if region_mode == "visible":
                        from skimage.segmentation import clear_border
                        if out_labels.max() > 0:
                            out_labels = clear_border(out_labels)

                        if existing_labels is not None:
                            # 1. remove cells in existing_labels that are inside the viewport but don't touch viewport bounds
                            viewport_ext = existing_labels[v_top:v_bottom, v_left:v_right]
                            if viewport_ext.shape[0] > 0 and viewport_ext.shape[1] > 0:
                                top_b = viewport_ext[0, :]
                                bot_b = viewport_ext[-1, :]
                                left_b = viewport_ext[:, 0]
                                right_b = viewport_ext[:, -1]
                                border_ids = np.unique(np.concatenate([top_b, bot_b, left_b, right_b]))
                                all_ids = np.unique(viewport_ext)
                                ids_to_remove = np.setdiff1d(all_ids, border_ids)
                                ids_to_remove = ids_to_remove[ids_to_remove > 0]
                                if len(ids_to_remove) > 0:
                                    mask_to_remove = np.isin(existing_labels, ids_to_remove)
                                    existing_labels[mask_to_remove] = 0

                            # 2. Add out_labels into existing_labels offsetting IDs
                            if out_labels.max() > 0:
                                max_id = existing_labels.max()
                                mask_new = out_labels > 0
                                existing_labels[v_top:v_bottom, v_left:v_right][mask_new] = out_labels[mask_new] + max_id
                            final_labels = existing_labels
                        else:
                            # New mask, apply zeros outside
                            full_labels = np.zeros(full_shape, dtype=out_labels.dtype)
                            full_labels[v_top:v_bottom, v_left:v_right] = out_labels
                            final_labels = full_labels

                    elif region_mode == "selected_region":
                        from PySide6.QtCore import Qt
                        # Filter: keep only cells whose centroid is inside the polygon
                        if out_labels.max() > 0 and region_polygon is not None:
                            out_labels = _cell_centroid_in_polygon(out_labels, region_polygon, r_top, r_left)

                        if existing_labels is not None:
                            # Remove existing cells whose centroids are inside the region polygon
                            if region_polygon is not None:
                                from PySide6.QtCore import QPointF
                                from scipy.ndimage import find_objects
                                locs = find_objects(existing_labels)
                                ids_in_region = []
                                for cell_id_m1, loc in enumerate(locs):
                                    if loc is None:
                                        continue
                                    cell_id = cell_id_m1 + 1
                                    binary = (existing_labels[loc] == cell_id)
                                    ys, xs = np.where(binary)
                                    if len(ys) == 0:
                                        continue
                                    cy = float(np.mean(ys)) + loc[0].start
                                    cx = float(np.mean(xs)) + loc[1].start
                                    if region_polygon.containsPoint(QPointF(cx, cy), Qt.FillRule.OddEvenFill):
                                        ids_in_region.append(cell_id)
                                if ids_in_region:
                                    mask_remove = np.isin(existing_labels, ids_in_region)
                                    existing_labels[mask_remove] = 0

                            # Merge new detections into existing mask
                            if out_labels.max() > 0:
                                max_id = existing_labels.max()
                                mask_new = out_labels > 0
                                existing_labels[r_top:r_bottom, r_left:r_right][mask_new] = out_labels[mask_new] + max_id
                            final_labels = existing_labels
                        else:
                            # New mask: place results at region bounding box, zeros elsewhere
                            full_labels = np.zeros(full_shape, dtype=out_labels.dtype)
                            if out_labels.max() > 0:
                                full_labels[r_top:r_bottom, r_left:r_right] = out_labels
                            final_labels = full_labels
                    else:
                        final_labels = out_labels

                    contour_data = self._get_contour_data(final_labels.astype(np.int32))
                    self.segmentationResultReady.emit(final_labels, out_name, out_is_cell, None, contour_data, "", False, target_idx, None, True, None)

                if method == "watershed":
                    from opal_studio.watershed import _voronoi_otsu_labeling, _gauss_otsu_labeling

                    labeller = params.get("labeller", "voronoi")
                    spot_sigma = params.get("spot_sigma", 2)
                    outline_sigma = params.get("outline_sigma", 2)
                    threshold = params.get("threshold", 1)
                    min_mean_intensity = params.get("min_mean_intensity", 0)

                    # Pre-process: Gaussian blur + Laplacian edge enhancement
                    nuclei_gauss = ndi.gaussian_filter(x, sigma=1.5)
                    nuclei_gauss = ndi.convolve(
                        nuclei_gauss, np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                    )

                    if labeller == "voronoi":
                        labels = _voronoi_otsu_labeling(
                            nuclei_gauss,
                            spot_sigma=spot_sigma,
                            outline_sigma=outline_sigma,
                            threshold=threshold,
                        )
                    elif labeller == "gauss":
                        labels = _gauss_otsu_labeling(
                            nuclei_gauss,
                            outline_sigma=outline_sigma,
                            threshold=threshold,
                        )

                    if min_mean_intensity > x.min():
                        cell_ids = np.unique(labels)
                        cell_ids = cell_ids[cell_ids > 0]
                        if len(cell_ids) > 0:
                            means = ndi.mean(x, labels=labels, index=cell_ids)
                            if not isinstance(means, (list, np.ndarray)):
                                means = [means]
                            mapping = np.zeros(int(labels.max()) + 1, dtype=labels.dtype)
                            for cid, mean_val in zip(cell_ids, means):
                                if mean_val >= min_mean_intensity:
                                    mapping[cid] = cid
                            labels = mapping[labels]
                    
                    process_and_emit(labels, override_name, False)
                else:
                    # Run deep learning models in a separate process
                    from opal_studio.segmentation_engine import run_segmentation_task_pipe
                    
                    ctx = multiprocessing.get_context("spawn")
                    parent_conn, child_conn = ctx.Pipe()
                    stop_event = ctx.Event()

                    worker_params = params.copy()
                    worker_params["override_name"] = override_name
                    worker_params["crop_offset_y"] = v_top if region_mode == "visible" else (r_top if region_mode == "selected_region" else 0)
                    worker_params["crop_offset_x"] = v_left if region_mode == "visible" else (r_left if region_mode == "selected_region" else 0)
                    worker_params["full_shape"] = list(full_shape) if full_shape is not None else None
                    worker_params["target_mode"] = target_mode
                    worker_params["target_mask_index"] = target_mask_index

                    proc = ctx.Process(target=run_segmentation_task_pipe, args=(child_conn, worker_params, input_channels_data, stop_event))
                    proc.start()
                    child_conn.close()
                    self._segmentation_proc = proc
                    self._seg_cancelled = False

                    try:
                        while True:
                            try:
                                msg = parent_conn.recv()
                            except EOFError:
                                if not self._seg_cancelled:
                                    raise Exception("Worker process terminated unexpectedly.")
                                break
                            msg_type = msg.get("type")
                            if msg_type == "tile_update":
                                self.segmentationTileReady.emit(msg)
                            elif msg_type == "result":
                                for out_labels, out_name, out_is_cell in msg.get("results", []):
                                    process_and_emit(out_labels, out_name, out_is_cell)
                                break
                            elif msg_type == "cancelled":
                                break
                            elif msg_type == "error":
                                raise Exception(f"Worker Error: {msg.get('error')}\n{msg.get('traceback')}")
                    finally:
                        proc.join(timeout=5)
                        parent_conn.close()
                        self._segmentation_proc = None
                if not self._seg_cancelled:
                    print(f"[Segmentation] Result emitted.")
            except Exception as e:
                import traceback; traceback.print_exc()
                if not self._seg_cancelled:
                    self.segmentationError.emit(str(e))
                else:
                    self.segmentationError.emit("")  # triggers stop_loading without dialog

        self._segmentation_thread = threading.Thread(target=_run, daemon=True)
        self._segmentation_thread.start()

    @Slot()
    def _cancel_segmentation(self):
        self._seg_cancelled = True
        proc = self._segmentation_proc
        if proc is not None and proc.is_alive():
            proc.terminate()
        self._segmentation_proc = None
        self._seg_preview_channel_idx = -1
        self._ops_panel.stop_loading()

    @Slot(object)
    def _on_segmentation_tile_ready(self, info: dict):
        y0 = info.get('y0', 0)
        x0 = info.get('x0', 0)
        y1 = info.get('y1', 0)
        x1 = info.get('x1', 0)
        tile_labels = info.get('tile_labels')
        full_shape_list = info.get('full_shape')
        name = info.get('name', 'Preview')
        is_cell_mask = info.get('is_cell_mask', False)
        tile_idx = info.get('tile_idx', 0)
        n_tiles = info.get('n_tiles', 1)
        target_mode = info.get('target_mode', 'new')
        target_mask_index = info.get('target_mask_index')

        if tile_labels is None or full_shape_list is None:
            return

        full_shape = tuple(full_shape_list)
        self._ops_panel.set_progress_info(tile_idx + 1, n_tiles)

        if target_mode == 'overwrite' and target_mask_index is not None:
            try:
                ch = self._channel_model.channel(target_mask_index)
                if ch.mask_data is not None:
                    ch.mask_data[y0:y1, x0:x1] = tile_labels
                    idx_qt = self._channel_model.index(target_mask_index)
                    self._channel_model.dataChanged.emit(idx_qt, idx_qt, [])
                    self._channel_model.channels_changed.emit()
            except Exception:
                pass
            return

        preview_idx = self._seg_preview_channel_idx
        ch = None
        if preview_idx >= 0:
            try:
                candidate = self._channel_model.channel(preview_idx)
                if candidate.mask_data is not None and candidate.mask_data.shape == full_shape:
                    ch = candidate
            except Exception:
                preview_idx = -1

        if ch is None:
            new_ch = Channel(
                name=self._channel_model.get_unique_name(name),
                color=QColor(255, 255, 255),
                visible=True,
                is_mask=True,
                is_cell_mask=is_cell_mask,
                mask_data=np.zeros(full_shape, dtype=np.int32),
                contour_data={},
                index=-1,
            )
            self._channel_model.add_channel(new_ch)
            preview_idx = self._channel_model.rowCount() - 1
            self._seg_preview_channel_idx = preview_idx
            idx_qt = self._channel_model.index(preview_idx)
            self._channel_model.setData(idx_qt, True, ChannelListModel.SelectedRole)
            ch = new_ch

        ch.mask_data[y0:y1, x0:x1] = tile_labels
        idx_qt = self._channel_model.index(preview_idx)
        self._channel_model.dataChanged.emit(idx_qt, idx_qt, [])
        self._channel_model.channels_changed.emit()

    def _on_segmentation_complete(self, labels, name, is_cell_mask, color, contour_data, source_marker, is_type_mask, target_idx, pos_lut=None, random_colors=True, aux_labels=None):
        # Type masks arrive one-per-cluster/type while the background thread is still running.
        # operationFinished handles stop_loading() for those batches.
        if not is_type_mask:
            self._ops_panel.stop_loading()

        # If a tile-progress preview channel was created, use it as the final target
        # so contour data replaces the raw tile preview in-place.
        if not is_type_mask and target_idx < 0 and self._seg_preview_channel_idx >= 0:
            target_idx = self._seg_preview_channel_idx
        if not is_type_mask:
            self._seg_preview_channel_idx = -1

        # For cell masks with no explicit target, overwrite any existing mask
        # for the same marker rather than accumulating duplicates.
        if is_cell_mask and target_idx < 0 and source_marker:
            for i in range(self._channel_model.rowCount()):
                ch = self._channel_model.channel(i)
                if ch.is_cell_mask and ch.source_marker == source_marker:
                    target_idx = i
                    break

        if target_idx >= 0:
            # Update existing
            try:
                ch = self._channel_model.channel(target_idx)
                ch.mask_data = labels
                ch.contour_data = contour_data
                ch.pos_lut = pos_lut
                ch.random_contour_colors = random_colors
                if aux_labels is not None:
                    ch.processed_data = aux_labels
                idx_qt = self._channel_model.index(target_idx)
                self._channel_model.dataChanged.emit(idx_qt, idx_qt, [])
                self._channel_model.channels_changed.emit()
                self._status.showMessage(f"Updated: {ch.name}", 5000)
                return
            except Exception:
                pass

        # Create new
        mask_name = self._channel_model.get_unique_name(name if name else "Mask")
        row_color = color if color else QColor(255, 255, 255)
        
        new_ch = Channel(
            name=mask_name, color=row_color, visible=True,
            is_mask=(not is_cell_mask and not is_type_mask), 
            is_cell_mask=is_cell_mask,
            is_type_mask=is_type_mask,
            mask_data=labels,
            contour_data=contour_data,
            pos_lut=pos_lut,
            processed_data=aux_labels,
            random_contour_colors=random_colors,
            source_marker=source_marker,
            index=-1
        )
        self._channel_model.add_channel(new_ch)
        
        # Select it!
        new_idx = self._channel_model.rowCount() - 1
        idx_qt = self._channel_model.index(new_idx)
        self._channel_model.setData(idx_qt, True, ChannelListModel.SelectedRole)

        self._status.showMessage(f"Task Complete: {mask_name}", 5000)

    @Slot(str)
    def _on_segmentation_error(self, message):
        self._ops_panel.stop_loading()
        self._seg_preview_channel_idx = -1
        if message:
            QMessageBox.critical(self, "Task Error", message)

    @Slot(dict)
    def _run_mask_expansion(self, params):
        tool = params.get("tool", "expansion_watershed")
        
        if tool == "cell_sampler":
            mask_indices = params.get("mask_indices", [])
            labels_list = []
            for idx in mask_indices:
                ch = self._channel_model.channel(idx)
                if ch.mask_data is not None:
                    labels_list.append(ch.mask_data)
                    
            if len(labels_list) < 2:
                self._ops_panel.stop_loading()
                return

            def _run_sampler():
                try:
                    from opal_studio.uber import UBM
                    import numpy as np
                    
                    carray = np.stack(labels_list)
                    merit = params.get("merit", "pop")
                    nsize = params.get("nsize", 40)
                    joint_mask, method_mask = UBM(carray).form_um(merit=merit, nsize=nsize)
                    mask_result = joint_mask.astype(np.int32)
                    
                    merit_suffixes = {"pop": "_Count", "j1": "_Jac", "cstd": "_Var"}
                    suffix = merit_suffixes.get(merit, f"_{merit}")
                    new_name = f"Sampled{suffix}"
                    
                    contour_data = self._get_contour_data(mask_result)
                    self.segmentationResultReady.emit(mask_result, new_name, False, None, contour_data, "", False, -1, None, True, None)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    self.segmentationError.emit(str(e))

            threading.Thread(target=_run_sampler, daemon=True).start()
            return

        mask_idx = params.get("mask_index")
        if mask_idx is None:
            self._ops_panel.stop_loading()
            return
            
        # In current model, mask_data is stored in .mask_data
        original_mask_ch = self._channel_model.channel(mask_idx)
        labels = original_mask_ch.mask_data
        if labels is None:
            self._ops_panel.stop_loading()
            return

        def _run():
            try:
                if tool == "expansion_watershed":
                    expansion_pixels = params["expansion_pixels"]
                    expanded = expand_labels_watershed(labels, expansion_pixels)
                    # Extract individual contours before binarizing
                    contour_data = self._get_contour_data(expanded.astype(np.int32))
                    mask_result = (expanded > 0).astype(np.int32)
                    new_name = f"{original_mask_ch.name}_Expand"
                    
                    # Store the labeled version in aux_labels so thresholding can find individual cells
                    # even though mask_data is binary.
                    self.segmentationResultReady.emit(mask_result, new_name, original_mask_ch.is_cell_mask, original_mask_ch.color, contour_data, "", original_mask_ch.is_type_mask, -1, None, False, expanded.astype(np.int32))
                    return
                elif tool == "expansion_labelmap":
                    expansion_pixels = params["expansion_pixels"]
                    expanded = expand_labels_labelmap(labels, expansion_pixels)
                    mask_result = expanded.astype(np.int32)
                    new_name = f"{original_mask_ch.name}_Expand"
                elif tool == "filter_size":
                    min_size = params["min_size"]
                    max_size = params["max_size"]
                    
                    sizes = np.bincount(labels.ravel())
                    sizes[0] = 0 # Ignore background
                    
                    mask_keep = (sizes >= min_size) & (sizes <= max_size)
                    
                    valid_labels = np.where(mask_keep)[0]
                    mapper = np.zeros(len(sizes), dtype=np.int32)
                    mapper[valid_labels] = valid_labels
                    mask_result = mapper[labels]
                    
                    new_name = f"{original_mask_ch.name}_Filter"
                
                contour_data = self._get_contour_data(mask_result)
                self.segmentationResultReady.emit(mask_result, new_name, original_mask_ch.is_cell_mask, original_mask_ch.color, contour_data, "", original_mask_ch.is_type_mask, -1, None, True, None)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    @Slot(dict)
    def _run_cell_positivity(self, params):
        if not self._image: return
        mask_idx = params["mask_index"]
        original_mask_ch = self._channel_model.channel(mask_idx)
        labels = original_mask_ch.mask_data
        if labels is None:
            self._ops_panel.stop_loading()
            return

        def _run():
            try:
                from opal_studio.image_loader import _get_yx
                from opal_studio.segmentation_engine import run_positivity_task
                
                print(f"[AI] Starting cell positivity detection for mask: {original_mask_ch.name}")
                
                target_channels = []
                for i in range(self._channel_model.rowCount()):
                    ch = self._channel_model.channel(i)
                    if not ch.is_mask and not ch.is_cell_mask:
                        target_channels.append(ch)
                
                S = len(target_channels)
                h, w = _get_yx(self._image.base_shape, self._image.axes, self._image.is_rgb)
                markers = np.zeros((h, w, S), dtype=np.float32)
                for z, ch in enumerate(target_channels):
                    if ch.is_processed and ch.processed_data is not None:
                        data = ch.processed_data.astype(np.float32)
                    else:
                        data = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)
                        
                    dh, dw = data.shape[:2]
                    copy_h = min(h, dh)
                    copy_w = min(w, dw)
                    markers[:copy_h, :copy_w, z] = data[:copy_h, :copy_w]

                # Run in worker process
                ctx = multiprocessing.get_context("spawn")
                queue = multiprocessing.Queue()
                
                proc = ctx.Process(target=run_positivity_task, args=(queue, params, {"labels": labels, "markers": markers}))
                proc.start()
                
                try:
                    worker_res = queue.get()
                except Exception:
                    raise Exception("Worker process failed to return result.")
                finally:
                    proc.join()
                
                if not worker_res.get("success"):
                    raise Exception(f"Worker Error: {worker_res.get('error')}\n{worker_res.get('traceback')}")
                
                # Extract individual contours using original labels
                contour_data = self._get_contour_data(labels)
                max_id = int(np.max(labels))

                for slice_out, z_idx in worker_res.get("results", []):
                    ch = target_channels[z_idx]
                    
                    # AI returns slice_out (0/1/2 map). Convert to pos_lut (ID -> state)
                    pos_lut = np.zeros(max_id + 1, dtype=np.int16)
                    mask_active = labels > 0
                    if np.any(mask_active):
                        pos_lut[labels[mask_active]] = slice_out[mask_active]

                    self.segmentationResultReady.emit(
                        labels, ch.name, True, ch.color, contour_data, 
                        ch.name, False, -1, pos_lut, False, None
                    )
                    self.operationProgress.emit(z_idx + 1, S)

                print("[AI] Cell positivity detection complete.")
                self.operationFinished.emit(S)

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

                print("[AI] Cell positivity detection complete.")
                self.operationFinished.emit(S)

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Threshold-based cell positivity
    # ------------------------------------------------------------------

    @Slot(dict)
    def _run_threshold_compute(self, params: dict):
        """Background thread: compute per-cell mean intensity for every image channel."""
        if not self._image:
            self._ops_panel.stop_loading()
            return

        mask_idx = params["mask_index"]
        method = params.get("method", "otsu")
        mask_ch = self._channel_model.channel(mask_idx)
        labels = mask_ch.mask_data
        if labels is None:
            self._ops_panel.stop_loading()
            return

        def _run():
            try:
                from opal_studio.image_loader import _get_yx
                from concurrent.futures import ThreadPoolExecutor
                from skimage.filters import (
                    threshold_otsu, threshold_triangle, threshold_li,
                    threshold_yen, threshold_isodata,
                )

                THRESH_METHODS = {
                    "otsu":     threshold_otsu,
                    "triangle": threshold_triangle,
                    "li":       threshold_li,
                    "yen":      threshold_yen,
                    "isodata":  threshold_isodata,
                }
                thresh_fn = THRESH_METHODS.get(method, threshold_otsu)

                # Handle binary masks (e.g. from Expansion) by using aux labels or re-labeling
                working_labels = labels
                if mask_ch.processed_data is not None:
                    working_labels = mask_ch.processed_data
                elif labels.max() == 1:
                    from skimage.measure import label
                    working_labels = label(labels).astype(np.int32)

                cell_ids = np.unique(working_labels)
                cell_ids = cell_ids[cell_ids > 0]
                if len(cell_ids) == 0:
                    self.segmentationError.emit("No cells found in the selected mask.")
                    return

                max_label = int(working_labels.max())
                h, w = _get_yx(self._image.base_shape, self._image.axes, self._image.is_rgb)

                # Collect image channels to process (skip mask channels)
                channel_indices = [
                    i for i in range(self._channel_model.rowCount())
                    if not (self._channel_model.channel(i).is_mask
                            or self._channel_model.channel(i).is_cell_mask
                            or self._channel_model.channel(i).is_type_mask)
                ]
                channels_snapshot = {i: self._channel_model.channel(i) for i in channel_indices}

                # Serialise file reads so we don't overwhelm I/O; compute in parallel
                _read_lock = threading.Lock()

                def _process_one(i):
                    ch = channels_snapshot[i]
                    if ch.is_processed and ch.processed_data is not None:
                        data = ch.processed_data.astype(np.float32)
                    else:
                        with _read_lock:
                            data = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)

                    dh, dw = data.shape[:2]
                    crop_h, crop_w = min(h, dh), min(w, dw)
                    lab_crop = working_labels[:crop_h, :crop_w]
                    dat_crop = data[:crop_h, :crop_w]

                    # bincount is ~3-5× faster than scipy.ndimage.mean for integer labels
                    flat_lbl = lab_crop.ravel()
                    flat_dat = dat_crop.ravel()
                    sums   = np.bincount(flat_lbl, weights=flat_dat.astype(np.float64), minlength=max_label + 1)
                    counts = np.bincount(flat_lbl, minlength=max_label + 1)
                    with np.errstate(invalid="ignore"):
                        means_lut = np.where(counts > 0, sums / counts, 0.0).astype(np.float32)

                    valid_means = means_lut[cell_ids]
                    valid_means = valid_means[valid_means > 0]
                    threshold = None
                    if valid_means.size > 1:
                        try:
                            val = thresh_fn(valid_means)
                            if not np.isnan(val):
                                threshold = float(val)
                        except Exception:
                            pass

                    return i, means_lut, threshold

                cell_means: dict[int, np.ndarray] = {}
                thresholds: dict[int, float] = {}

                n_workers = min(len(channel_indices), max(1, (os.cpu_count() or 4)))
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    for i, means_lut, threshold in pool.map(_process_one, channel_indices):
                        cell_means[i] = means_lut
                        if threshold is not None:
                            thresholds[i] = threshold

                self.thresholdMeansReady.emit(working_labels, cell_means, thresholds, mask_idx)

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    @Slot(object, object, object, int)
    def _on_threshold_means_ready(self, labels, cell_means, thresholds, mask_model_idx):
        """Deliver computed means to the Thresholds tab (main thread)."""
        self._ops_panel.stop_loading()
        self._ops_panel._thresh_tab.receive_means(labels, cell_means, thresholds, mask_model_idx)

    @Slot(dict)
    def _apply_threshold_positivity(self, params: dict):
        """
        Apply a threshold to the precomputed per-cell means and update (or create)
        the result channel.  Runs on the MAIN THREAD — the LUT lookup is O(H×W)
        and completes in <50 ms even for very large images.
        """
        ch_model_idx    = params["ch_model_index"]
        threshold       = params["threshold"]
        mask_idx        = params["mask_index"]
        ch_name         = params["ch_name"]
        ch_color        = params["ch_color"]
        target_ch_idx   = params["target_ch_index"]  # -1 = not yet created

        mask_ch = self._channel_model.channel(mask_idx)
        labels  = mask_ch.mask_data
        if labels is None:
            return

        # Source of truth for IDs: processed_data (if it's a label map) or connected components if binary
        working_labels = labels
        if mask_ch.processed_data is not None:
            working_labels = mask_ch.processed_data
        elif labels.max() == 1:
            from skimage.measure import label
            working_labels = label(labels).astype(np.int32)
        
        # Retrieve the per-cell means LUT that was already computed
        thresh_tab = self._ops_panel._thresh_tab
        means_lut  = thresh_tab._cell_means.get(ch_model_idx)
        if means_lut is None:
            return

        # --- Individual Cell identity ---
        # We store the ORIGINAL label map (with 10k unique IDs) in mask_data,
        # but store the positivity classification in a LUT (pos_lut).
        # This allows per-cell contours and individual cell manipulation.
        
        pos_lut = np.zeros(len(means_lut), dtype=np.int16)
        cell_ids = np.flatnonzero(means_lut > 0)
        if len(cell_ids):
            pos_lut[cell_ids] = np.where(means_lut[cell_ids] >= threshold, 2, 1).astype(np.int16)

        # If no registered target yet, check for an existing cell mask for this marker
        if target_ch_idx == -1:
            for i in range(self._channel_model.rowCount()):
                existing = self._channel_model.channel(i)
                if existing.is_cell_mask and existing.source_marker == ch_name:
                    target_ch_idx = i
                    thresh_tab.register_generated_channel(ch_model_idx, i)
                    break

        if target_ch_idx != -1:
            # Update existing channel in-place
            try:
                tgt_ch = self._channel_model.channel(target_ch_idx)
                tgt_ch.mask_data = working_labels
                tgt_ch.pos_lut = pos_lut
                idx_qt = self._channel_model.index(target_ch_idx)
                self._channel_model.dataChanged.emit(idx_qt, idx_qt, [])
                self._channel_model.channels_changed.emit()
                return
            except Exception:
                pass  # Channel may have been removed — fall through to create new

        # First time: create a new cell-mask channel
        contour_data = self._get_contour_data(labels)   # One contour per cell ID
        new_name = self._channel_model.get_unique_name(ch_name)
        new_ch = Channel(
            name=new_name,
            color=ch_color,
            visible=True,
            is_cell_mask=True,
            mask_data=working_labels,        # Use working labels
            pos_lut=pos_lut,         # Positivity states
            contour_data=contour_data,
            source_marker=ch_name,
            index=-1,
        )
        self._channel_model.add_channel(new_ch)
        new_idx = self._channel_model.rowCount() - 1

        # Register so future slider moves update in-place
        thresh_tab.register_generated_channel(ch_model_idx, new_idx)
        self._status.showMessage(f"Threshold positivity: {new_name}", 3000)

    @Slot(dict)
    def _run_cluster_cell_identification(self, params: dict):
        """Rename existing cluster channels to their identified cell types."""
        if not self._image:
            self._ops_panel.stop_loading()
            return

        if self._cluster_cell_ids is None or self._cluster_labels_arr is None:
            self._ops_panel.stop_loading()
            self.statusBar().showMessage("Run clustering first.", 5000)
            return

        definitions = self._phenotyping_tab.get_phenotype_definitions()
        if not definitions:
            self._ops_panel.stop_loading()
            self.statusBar().showMessage("No cell types defined in the Phenotyping tab.", 5000)
            return

        pos_cutoff = params.get("positive_fraction_cutoff", 0.60)
        neg_cutoff = params.get("negative_fraction_cutoff", 0.20)

        # Collect per-marker positivity LUTs on main thread (safe Qt model access)
        marker_luts: dict[str, np.ndarray] = {}
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_cell_mask and ch.pos_lut is not None:
                key = ch.source_marker if ch.source_marker else ch.name
                marker_luts[key] = ch.pos_lut

        if not marker_luts:
            self._ops_panel.stop_loading()
            self.statusBar().showMessage(
                "No positivity masks found. Run thresholding first.", 5000
            )
            return

        cell_ids       = self._cluster_cell_ids
        cluster_labels = self._cluster_labels_arr

        def _run():
            try:
                real_clusters = np.unique(cluster_labels)
                real_clusters = real_clusters[real_clusters >= 0]

                # Step 1: per-cluster, per-marker positivity fraction → state
                # State: 1 = positive, 2 = negative, 0 = neutral/mixed
                cluster_states: dict[int, dict[str, int]] = {}
                for cluster_id in real_clusters:
                    cells = cell_ids[cluster_labels == cluster_id]
                    states_for_cluster: dict[str, int] = {}
                    for marker_name, lut in marker_luts.items():
                        max_cell = int(cells.max()) if len(cells) else 0
                        if max_cell >= len(lut):
                            safe_lut = np.zeros(max_cell + 1, dtype=np.int16)
                            safe_lut[:len(lut)] = lut
                        else:
                            safe_lut = lut
                        cell_states = safe_lut[cells]
                        valid = cell_states[cell_states > 0]
                        if len(valid) == 0:
                            states_for_cluster[marker_name] = 0
                            continue
                        pos_fraction = float(np.sum(valid == 2)) / len(valid)
                        if pos_fraction >= pos_cutoff:
                            states_for_cluster[marker_name] = 1
                        elif pos_fraction <= neg_cutoff:
                            states_for_cluster[marker_name] = 2
                        else:
                            states_for_cluster[marker_name] = 0
                    cluster_states[int(cluster_id)] = states_for_cluster

                # Step 2: match each cluster to the first matching phenotype
                cluster_to_type: dict[int, str] = {}
                for cluster_id in real_clusters:
                    c_states = cluster_states[int(cluster_id)]
                    assigned = None
                    for type_name, criteria in definitions.items():
                        if not criteria:
                            continue
                        if all(c_states.get(m, 0) == req for m, req in criteria.items()):
                            assigned = type_name
                            break
                    cluster_to_type[int(cluster_id)] = assigned or "Unknown"

                self.clusterTypesIdentified.emit(cluster_to_type)

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(f"Cluster identification error: {e}")

        threading.Thread(target=_run, daemon=True).start()

    @Slot(object)
    def _on_cluster_types_identified(self, cluster_to_type: dict):
        """Rename existing cluster channels and heatmap columns to identified cell types."""
        # Snapshot current heatmap names before modifying anything
        current_names = self._clustering_heatmap_tab.get_cluster_names()

        for cluster_id, type_name in cluster_to_type.items():
            old_name = current_names.get(cluster_id, f"Cluster {cluster_id}")
            for i in range(self._channel_model.rowCount()):
                ch = self._channel_model.channel(i)
                if ch.is_type_mask and ch.name == old_name:
                    ch.name = type_name
                    idx_qt = self._channel_model.index(i)
                    self._channel_model.dataChanged.emit(idx_qt, idx_qt, [])
                    break

        self._clustering_heatmap_tab.rename_clusters(cluster_to_type)
        self._ops_panel.stop_loading()
        self.statusBar().showMessage(
            f"Identified {len(cluster_to_type)} clusters.", 5000
        )

    def _run_cell_identification(self):
        """Perform logical gating based on phenotyping definitions in a background thread."""
        if not self._image: return


        definitions = self._phenotyping_tab.get_phenotype_definitions()
        if not definitions:
            self._ops_panel.stop_loading()
            self.statusBar().showMessage("No cell types defined in the Phenotyping tab.", 5000)
            return

        # Build per-marker positivity maps: 0=background, 1=negative, 2=positive
        # ch.mask_data is the label map; ch.pos_lut maps label -> positivity state
        cell_positivity: dict[str, np.ndarray] = {}
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_cell_mask and ch.mask_data is not None and ch.pos_lut is not None:
                key = ch.source_marker if ch.source_marker else ch.name
                label_map = ch.mask_data
                lut = ch.pos_lut
                max_label = int(label_map.max())
                if max_label >= len(lut):
                    safe_lut = np.zeros(max_label + 1, dtype=np.int16)
                    safe_lut[:len(lut)] = lut
                else:
                    safe_lut = lut
                cell_positivity[key] = safe_lut[label_map]

        if not cell_positivity:
            self._ops_panel.stop_loading()
            self.statusBar().showMessage("No cell positivity masks found. Detect marker positivity first.", 5000)
            return

        def _run():
            try:
                some_map = next(iter(cell_positivity.values()))
                h, w = some_map.shape

                from opal_studio.channel_model import generate_spaced_colors
                colors = generate_spaced_colors(len(definitions) + 10)

                total_types = len(definitions)
                identified_count = 0
                any_identified_mask = np.zeros((h, w), dtype=bool)

                for i, (type_name, criteria) in enumerate(definitions.items()):
                    if not criteria:
                        self.operationProgress.emit(i+1, total_types)
                        continue

                    valid_mask = np.ones((h, w), dtype=bool)
                    criteria_met = True
                    for marker_name, required_state in criteria.items():
                        if marker_name in cell_positivity:
                            # pos_lut values: 1=negative, 2=positive
                            target_val = 2 if required_state == 1 else 1
                            valid_mask &= (cell_positivity[marker_name] == target_val)
                        else:
                            criteria_met = False
                            break

                    if criteria_met and np.any(valid_mask):
                        any_identified_mask |= valid_mask
                        type_data = np.zeros((h, w), dtype=np.uint8)
                        type_data[valid_mask] = 1 # Binary mask for this type
                        
                        color_rgb = colors[i % len(colors)]
                        row_color = QColor(*color_rgb)
                        
                        # We use the existing result handler to handle thread-safe model addition
                        # We pass is_cell_mask=False and use source_marker="" for types.
                        # Wait, we need it to be is_type_mask. 
                        # I'll update _on_segmentation_complete once more to handle this.
                        # Signal compatibility: is_cell_mask=False, source_marker="", is_type_mask=True
                        self.segmentationResultReady.emit(type_data, type_name, False, row_color, None, "", True, -1, None, False, None)
                        identified_count += 1
                    
                    self.operationProgress.emit(i+1, total_types)

                # Add Unknown phenotype for cells that matched nothing
                # some_map > 0 covers all pixels that belong to any cell (neg or pos)
                is_cell = (some_map > 0)
                unknown_mask = is_cell & (~any_identified_mask)
                if np.any(unknown_mask):
                    unknown_data = np.zeros((h, w), dtype=np.uint8)
                    unknown_data[unknown_mask] = 1
                    self.segmentationResultReady.emit(unknown_data, "Unknown", False, QColor(128, 128, 128), None, "", True, -1, None, False, None)
                    identified_count += 1

                self.operationFinished.emit(identified_count)

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(f"Identification Error: {e}")

        threading.Thread(target=_run, daemon=True).start()

    @Slot(dict)
    def _run_clustering(self, params: dict):
        """Perform cell clustering based on per-cell mean intensities."""
        if not self._image:
            self._ops_panel.stop_loading()
            return

        mask_idx = params["mask_index"]
        mask_ch = self._channel_model.channel(mask_idx)
        labels = mask_ch.mask_data
        if labels is None:
            self._ops_panel.stop_loading()
            return

        # Remove any existing type masks (clusters) from previous runs
        for i in range(self._channel_model.rowCount() - 1, -1, -1):
            ch = self._channel_model.channel(i)
            if ch.is_type_mask:
                self._channel_model.remove_channel(i)
        self._clustering_heatmap_tab.clear()
        self._tsne_tab.clear()
        self._umap_tab.clear()

        def _run():
            try:
                from opal_studio.image_loader import _get_yx
                from opal_studio.clustering import (
                    normalize_data, run_leiden, run_louvain, run_dbscan,
                    run_kmeans, run_phenograph, run_flowsom, run_hierarchical,
                )
                from opal_studio.channel_model import generate_spaced_colors

                # Resolve label map: use aux labels if available, or re-label binary
                working_labels = labels
                if mask_ch.processed_data is not None:
                    working_labels = mask_ch.processed_data
                elif labels.max() == 1:
                    from skimage.measure import label
                    working_labels = label(labels)
                
                working_labels = working_labels.astype(np.int32)
                cell_ids = np.unique(working_labels)
                cell_ids = cell_ids[cell_ids > 0]
                if len(cell_ids) == 0:
                    self.segmentationError.emit("No cells found in the selected mask.")
                    return

                max_label = int(working_labels.max())
                h, w = _get_yx(self._image.base_shape, self._image.axes, self._image.is_rgb)

                # ── Compute per-cell mean intensity for every image channel ──
                channel_names = []
                means_columns = []
                selected_channels = set(params.get("selected_channels") or [])

                for i in range(self._channel_model.rowCount()):
                    ch = self._channel_model.channel(i)
                    if ch.is_mask or ch.is_cell_mask or ch.is_type_mask or ch.is_region:
                        continue
                    if selected_channels and i not in selected_channels:
                        continue

                    if ch.is_processed and ch.processed_data is not None:
                        data = ch.processed_data.astype(np.float32)
                    else:
                        data = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)

                    dh, dw = data.shape[:2]
                    crop_h, crop_w = min(h, dh), min(w, dw)
                    lab_crop = working_labels[:crop_h, :crop_w]
                    dat_crop = data[:crop_h, :crop_w]

                    means_list = ndi.mean(dat_crop, labels=lab_crop, index=cell_ids)
                    if not isinstance(means_list, np.ndarray):
                        means_list = np.array([means_list])
                    means_list = np.nan_to_num(means_list)

                    channel_names.append(ch.name)
                    means_columns.append(means_list)

                if len(means_columns) == 0:
                    self.segmentationError.emit("No image channels found for clustering.")
                    return

                # Shape: (n_cells, n_channels)
                cell_means = np.column_stack(means_columns).astype(np.float32)

                # ── Normalize ────────────────────────────────────────────────
                norm_method = params.get("normalization", "zscore")
                cell_means_norm = normalize_data(
                    cell_means,
                    method=norm_method,
                    cofactor=params.get("cofactor", 5),
                    skewness_threshold=params.get("skewness_threshold", 1),
                )

                # ── PCA (always for DBSCAN, optional for others) ─────────────
                method = params["method"]
                use_pca = params.get("use_pca", False) or method == "dbscan"
                pca_n_used: int | None = None
                pca_auto = False
                if use_pca:
                    from sklearn.decomposition import PCA as _PCA
                    from opal_studio.dimensionality_reduction import parallel_analysis_n_components
                    override = params.get("pca_components")  # None = use PA
                    if override is None:
                        n_comp = parallel_analysis_n_components(cell_means_norm)
                        pca_auto = True
                    else:
                        n_comp = override
                    n_comp = min(n_comp, cell_means_norm.shape[1], cell_means_norm.shape[0] - 1)
                    n_comp = max(1, n_comp)
                    cell_means_clust = _PCA(n_components=n_comp, random_state=42).fit_transform(cell_means_norm)
                    pca_n_used = n_comp
                else:
                    cell_means_clust = cell_means_norm

                # ── Cluster ──────────────────────────────────────────────────
                if method == "leiden":
                    cluster_labels, n_clusters = run_leiden(
                        cell_means_clust,
                        resolution=params.get("resolution", 0.5),
                    )
                elif method == "louvain":
                    cluster_labels, n_clusters = run_louvain(
                        cell_means_clust,
                        resolution=params.get("resolution", 0.5),
                    )
                elif method == "phenograph":
                    cluster_labels, n_clusters = run_phenograph(
                        cell_means_clust,
                        k=params.get("k", 30),
                    )
                elif method == "flowsom":
                    cluster_labels, n_clusters = run_flowsom(
                        cell_means_clust,
                        xdim=params.get("xdim", 10),
                        ydim=params.get("ydim", 10),
                        n_clusters=params.get("n_clusters", 10),
                    )
                elif method == "dbscan":
                    cluster_labels, n_clusters = run_dbscan(
                        cell_means_clust,
                        eps=params.get("eps", None),
                        min_samples=params.get("min_samples", 10),
                    )
                elif method == "kmeans":
                    cluster_labels, n_clusters = run_kmeans(
                        cell_means_clust,
                        n_clusters=params.get("n_clusters", 5),
                    )
                elif method == "hierarchical":
                    cluster_labels, n_clusters = run_hierarchical(
                        cell_means_clust,
                        n_clusters=params.get("n_clusters", 5),
                        linkage=params.get("linkage", "ward"),
                        metric=params.get("metric", "euclidean"),
                    )
                else:
                    self.segmentationError.emit(f"Unknown clustering method: {method}")
                    return

                # ── Build a type mask per cluster and emit ────────────────────
                unique_clusters = np.unique(cluster_labels)
                # For DBSCAN, -1 means noise
                real_clusters = unique_clusters[unique_clusters >= 0]
                colors = generate_spaced_colors(len(real_clusters) + 5)

                # Build a cell_id -> cluster_label lookup
                # cell_ids[j] has cluster cluster_labels[j]
                id_to_cluster = np.full(max_label + 1, -1, dtype=np.int32)
                for j, cid in enumerate(cell_ids):
                    id_to_cluster[cid] = cluster_labels[j]

                for ci, cluster_id in enumerate(real_clusters):
                    # Binary mask: pixels belonging to cells in this cluster
                    member_ids = cell_ids[cluster_labels == cluster_id]
                    member_set = np.zeros(max_label + 1, dtype=bool)
                    member_set[member_ids] = True
                    type_data = member_set[working_labels].astype(np.uint8)

                    cluster_name = f"Cluster {cluster_id}"
                    color_rgb = colors[ci % len(colors)]
                    row_color = QColor(*color_rgb)

                    self.segmentationResultReady.emit(
                        type_data, cluster_name, False, row_color, None, "", True, -1, None, False, None
                    )
                    self.operationProgress.emit(ci + 1, len(real_clusters))

                # Handle DBSCAN noise as a separate "Noise" type
                noise_ids = cell_ids[cluster_labels == -1] if -1 in cluster_labels else np.array([])
                if len(noise_ids) > 0:
                    noise_set = np.zeros(max_label + 1, dtype=bool)
                    noise_set[noise_ids] = True
                    noise_data = noise_set[working_labels].astype(np.uint8)
                    self.segmentationResultReady.emit(
                        noise_data, "Noise", False, QColor(128, 128, 128), None, "", True, -1, None, False, None
                    )

                # Store for cluster-based cell identification
                self._cluster_cell_ids     = cell_ids
                self._cluster_labels_arr   = cluster_labels
                self._cluster_working_labels = working_labels

                # ── Emit heatmap data ─────────────────────────────────────
                # Compute per-cluster mean of the ORIGINAL (pre-norm) data
                heatmap = np.zeros((len(real_clusters), len(channel_names)), dtype=np.float32)
                for ci, cluster_id in enumerate(real_clusters):
                    member_mask = cluster_labels == cluster_id
                    if np.any(member_mask):
                        heatmap[ci] = cell_means[member_mask].mean(axis=0)

                self.clusteringHeatmapReady.emit(
                    list(real_clusters), channel_names, heatmap
                )

                # ── Dimensionality reduction (t-SNE & UMAP) ──────────────────
                cluster_colors_map = {
                    int(cluster_id): colors[ci % len(colors)]
                    for ci, cluster_id in enumerate(real_clusters)
                }
                # Noise cluster (-1) gets grey
                if -1 in cluster_labels:
                    cluster_colors_map[-1] = (128, 128, 128)

                from opal_studio.dimensionality_reduction import run_tsne, run_umap
                tsne_coords = run_tsne(cell_means_clust)
                umap_coords = run_umap(cell_means_clust)

                self.clusteringDimReductionReady.emit(
                    tsne_coords, umap_coords, cluster_labels, cluster_colors_map
                )

                # ── Clustering quality metrics ────────────────────────────────
                from sklearn.metrics import (
                    silhouette_score, davies_bouldin_score, calinski_harabasz_score
                )
                valid_mask = cluster_labels >= 0
                X_valid = cell_means_clust[valid_mask]
                y_valid = cluster_labels[valid_mask]
                n_unique = len(np.unique(y_valid))

                lines = []
                lines.append(f"Cells: {len(cell_ids):,}  |  Clusters: {len(real_clusters)}")
                if pca_n_used is not None:
                    pa_note = "Horn's PA, correlation shuffle" if pca_auto else "manual override"
                    lines.append(f"PCA: {pca_n_used}/{cell_means_norm.shape[1]} components ({pa_note})")
                if -1 in cluster_labels:
                    n_noise = int((cluster_labels == -1).sum())
                    lines.append(f"Noise (DBSCAN): {n_noise:,} cells ({100*n_noise/len(cell_ids):.1f}%)")
                lines.append("")
                lines.append("Quality Metrics")
                lines.append("-" * 32)

                if n_unique >= 2:
                    # Subsample silhouette for large datasets to stay responsive
                    MAX_SIL = 10_000
                    if len(X_valid) > MAX_SIL:
                        rng = np.random.default_rng(42)
                        idx = rng.choice(len(X_valid), MAX_SIL, replace=False)
                        sil = silhouette_score(X_valid[idx], y_valid[idx])
                        lines.append(f"Silhouette Score:          {sil:+.3f}  (subsample {MAX_SIL:,})")
                    else:
                        sil = silhouette_score(X_valid, y_valid)
                        lines.append(f"Silhouette Score:          {sil:+.3f}")
                    lines.append(f"  ↑ higher is better  (−1 to 1)")
                    db = davies_bouldin_score(X_valid, y_valid)
                    lines.append(f"Davies-Bouldin Index:      {db:.3f}")
                    lines.append(f"  ↓ lower is better")
                    ch = calinski_harabasz_score(X_valid, y_valid)
                    lines.append(f"Calinski-Harabasz Index: {ch:,.1f}")
                    lines.append(f"  ↑ higher is better")
                else:
                    lines.append("(need ≥ 2 clusters for quality metrics)")

                lines.append("")
                lines.append("Cluster Sizes")
                lines.append("-" * 32)
                n_total = len(cell_ids)
                for cluster_id in real_clusters:
                    n = int((cluster_labels == cluster_id).sum())
                    lines.append(f"  Cluster {cluster_id}: {n:>6,} cells  ({100*n/n_total:.1f}%)")

                self.clusteringMetricsReady.emit("\n".join(lines))

                self.operationFinished.emit(len(real_clusters))

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(f"Clustering Error: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _on_operation_complete(self, count):
        self._ops_panel.stop_loading()
        # count might be total markers or total types
        self.statusBar().showMessage(f"Task complete. Processed {count} items.", 5000)

    @Slot(object, object, object)
    def _on_clustering_heatmap_ready(self, cluster_ids, channel_names, heatmap_data):
        """Populate the clustering heatmap tab."""
        self._clustering_heatmap_tab.set_heatmap(cluster_ids, channel_names, heatmap_data)

    @Slot(object, object, object, object)
    def _on_dim_reduction_ready(self, tsne_coords, umap_coords, cluster_labels, cluster_colors):
        """Populate the t-SNE and UMAP scatter plot tabs."""
        # Store ordering for later color-sync when the user changes cluster colors
        self._active_cluster_ids = sorted(k for k in cluster_colors if k >= 0)
        if -1 in cluster_colors:
            self._active_cluster_ids.append(-1)

        self._tsne_tab.set_data(tsne_coords, cluster_labels, cluster_colors)
        if umap_coords is not None:
            self._umap_tab.set_data(umap_coords, cluster_labels, cluster_colors)

    @Slot(object, object, object)
    def _on_channel_data_changed(self, top_left, bottom_right, roles):
        from opal_studio.channel_model import ChannelListModel
        row = top_left.row()
        ch = self._channel_model.channel(row)
        if not ch or not ch.is_type_mask:
            return
        if not roles or ChannelListModel.ColorRole in roles:
            self._sync_scatter_colors()
        if not roles or ChannelListModel.VisibleRole in roles:
            self._sync_scatter_visibility()

    def _sync_scatter_visibility(self):
        """Hide/show clusters in scatter plots to match type mask visibility."""
        if not self._active_cluster_ids:
            return
        hidden: set[int] = set()
        type_idx = 0
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_type_mask:
                if type_idx < len(self._active_cluster_ids):
                    if not ch.visible:
                        hidden.add(self._active_cluster_ids[type_idx])
                type_idx += 1
        self._tsne_tab.set_hidden_clusters(hidden)
        self._umap_tab.set_hidden_clusters(hidden)

    def _sync_scatter_colors(self):
        """Rebuild the cluster→color map from current type mask channels and repaint."""
        if not self._active_cluster_ids:
            return
        new_colors: dict[int, tuple] = {}
        type_idx = 0
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_type_mask:
                if type_idx < len(self._active_cluster_ids):
                    cid = self._active_cluster_ids[type_idx]
                    new_colors[cid] = (ch.color.red(), ch.color.green(), ch.color.blue())
                type_idx += 1
        if new_colors:
            self._tsne_tab.update_colors(new_colors)
            self._umap_tab.update_colors(new_colors)

    @Slot(int, str, str)
    def _on_cluster_renamed(self, _cluster_id: int, old_name: str, new_name: str):
        """Rename the type mask channel that corresponds to a cluster."""
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_type_mask and ch.name == old_name:
                ch.name = new_name
                idx_qt = self._channel_model.index(i)
                self._channel_model.dataChanged.emit(idx_qt, idx_qt, [])
                break

    def _on_segmentation_error(self, message):
        self._ops_panel.stop_loading()
        QMessageBox.critical(self, "Segmentation Error", message)

    @Slot(int, int)
    def _on_pixel_hovered(self, x: int, y: int):
        if not self._image: return
        axes = self._image.axes
        from opal_studio.image_loader import _get_yx, get_tile
        h, w = _get_yx(self._image.base_shape, axes, self._image.is_rgb)
        
        ch = self._channel_model.selected_channel()
        if not ch:
            self._status.showMessage(f"X: {x}, Y: {y}")
            return
            
        if 0 <= y < h and 0 <= x < w:
            val = None
            if ch.is_processed and ch.processed_data is not None:
                val = ch.processed_data[y, x]
            elif (ch.is_mask or ch.is_cell_mask) and ch.mask_data is not None:
                val = ch.mask_data[y, x]
            elif getattr(ch, 'is_region', False):
                val = "Region"
            elif ch.index >= 0:
                try:
                    tile = get_tile(self._image, 0, ch.index, slice(y, y+1), slice(x, x+1))
                    if tile.size > 0:
                        if self._image.is_rgb:
                            val = f"RGB{tuple(tile[0, 0])}"
                        else:
                            val = tile[0, 0]
                except Exception:
                    pass
            
            if val is not None:
                self._status.showMessage(f"X: {x}, Y: {y} | [{ch.name}] Val: {val}")
            else:
                self._status.showMessage(f"X: {x}, Y: {y}")
        else:
            self._status.clearMessage()

    def _sync_canvas_to_brightfield(self, vp):
        if self._viewport_syncing:
            return
        self._viewport_syncing = True
        try:
            self._brightfield_view.set_image_viewport(vp)
        finally:
            self._viewport_syncing = False

    def _sync_brightfield_to_canvas(self, vp):
        if self._viewport_syncing:
            return
        self._viewport_syncing = True
        try:
            self._canvas.set_image_viewport(vp)
        finally:
            self._viewport_syncing = False

    @Slot(list)
    def _on_region_drawn(self, points: list[QPointF]):
        if not points:
            return
            
        xs = [pt.x() for pt in points]
        ys = [pt.y() for pt in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        from PySide6.QtGui import QPolygonF
        qpoly = QPolygonF(points)
        
        contour_data = {
            1: {
                "polygons": [qpoly],
                "bbox": [min_y, min_x, max_y, max_x]
            }
        }
        
        regions_count = sum(1 for ch in self._channel_model._channels if getattr(ch, 'is_region', False))
        
        from opal_studio.channel_model import generate_spaced_colors
        colors = generate_spaced_colors(regions_count + 1)
        rgb = colors[regions_count % len(colors)]
        row_color = QColor(*rgb)
        
        region_name = self._channel_model.get_unique_name("Region ")
        
        new_ch = Channel(
            name=region_name,
            color=row_color,
            visible=True,
            is_region=True,
            contour_data=contour_data,
            index=-1
        )
        self._channel_model.add_channel(new_ch)
        
        new_idx = self._channel_model.rowCount() - 1
        idx_qt = self._channel_model.index(new_idx)
        self._channel_model.setData(idx_qt, True, ChannelListModel.SelectedRole)
        
        self.statusBar().showMessage(f"Created region: {region_name}", 3000)

    def closeEvent(self, event):
        super().closeEvent(event)

    def _get_contour_data(self, labels: np.ndarray) -> dict:
        """Generate vector contours from a label image (Pre-cached as QPolygonF)."""
        try:
            import cv2
            objs = find_objects(labels)
            contour_data = {}
            
            for i, loc in enumerate(objs):
                if loc is None: continue
                label_id = i + 1
                
                crop = labels[loc]
                binary_crop = (crop == label_id).astype(np.uint8)
                
                # Use OpenCV for much faster contour finding
                cnts, _ = cv2.findContours(binary_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts: continue
                
                polygons = []
                for c in cnts:
                    if len(c) < 3: continue
                    qpoly = QPolygonF()
                    for pt in c:
                        px, py = pt[0]
                        # Offset back to original image space
                        oy = loc[0].start + py + 0.5
                        ox = loc[1].start + px + 0.5
                        qpoly.append(QPointF(ox, oy))
                    # Close the polygon
                    qpoly.append(qpoly.at(0))
                    polygons.append(qpoly)
                
                if polygons:
                    contour_data[label_id] = {
                        "polygons": polygons,
                        "bbox": [loc[0].start, loc[1].start, loc[0].stop, loc[1].stop] # [y0, x0, y1, x1]
                    }
            
            return contour_data
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Error generating vector contours: {e}")
            return {}
