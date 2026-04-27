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
    QHBoxLayout, QStatusBar, QMessageBox, QTabWidget, QApplication
)

from opal_studio.channel_model import Channel, ChannelListModel
from opal_studio.image_loader import ImageData, open_image, get_tile, _get_yx
from opal_studio.widgets.channel_panel import ChannelPanel
from opal_studio.widgets.image_canvas import ImageCanvas
from opal_studio.widgets.operations_panel import OperationsPanel
from opal_studio.widgets.phenotyping_tab import PhenotypingTab

import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.segmentation import watershed, find_boundaries
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
    preprocessingResultReady = Signal(str, str, object, float, float) # original_name, suffix, data, min, max
    preprocessingError = Signal(str)
    operationProgress = Signal(int, int)
    operationFinished = Signal(int)
    thresholdMeansReady = Signal(object, object, object, int)  # labels, cell_means dict, otsu dict, mask_model_idx

    def __init__(self):
        super().__init__()
        print("DEBUG: MainWindow initializing...")
        self.setWindowTitle("Opal Studio")
        self.resize(1400, 900)

        # Data model
        self._channel_model = ChannelListModel()
        self._image: ImageData | None = None

        # Components
        self._channel_panel = ChannelPanel(self._channel_model)
        self._canvas = ImageCanvas(self._channel_model)
        self._ops_panel = OperationsPanel(self._channel_model, self)
        self._phenotyping_tab = PhenotypingTab(self._channel_model)

        # Signals
        self._ops_panel.runSegmentationRequested.connect(self._start_segmentation)
        self._ops_panel.runPreprocessingRequested.connect(self._run_preprocessing)
        self._ops_panel.runMaskProcessingRequested.connect(self._run_mask_expansion)
        self._ops_panel.runCellPositivityRequested.connect(self._run_cell_positivity)
        self._ops_panel.runCellIdentificationRequested.connect(self._run_cell_identification)
        self._ops_panel.runThresholdComputeRequested.connect(self._run_threshold_compute)
        self._ops_panel.applyThresholdRequested.connect(self._apply_threshold_positivity)
        self.operationProgress.connect(self._ops_panel.set_progress_info)
        self.operationFinished.connect(self._on_operation_complete)
        self.thresholdMeansReady.connect(self._on_threshold_means_ready)
        
        self.segmentationResultReady.connect(self._on_segmentation_complete)
        self.segmentationError.connect(self._on_segmentation_error)
        self.preprocessingResultReady.connect(self._on_preprocessing_complete)
        self.preprocessingError.connect(self._on_segmentation_error)
        self._canvas.pixelHovered.connect(self._on_pixel_hovered)

        # Layout
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._channel_panel)
        
        self._center_tabs = QTabWidget()
        
        # Inject an invisible native icon to force the tab bar to draw taller, preventing cut-off text
        spacer_pixmap = QPixmap(1, 24)
        spacer_pixmap.fill(Qt.GlobalColor.transparent)
        spacer_icon = QIcon(spacer_pixmap)
        
        self._center_tabs.setIconSize(QSize(1, 24))
        self._center_tabs.addTab(self._canvas, spacer_icon, "Image")
        self._center_tabs.addTab(self._phenotyping_tab, spacer_icon, "Phenotyping")
        self._splitter.addWidget(self._center_tabs)
        
        self._splitter.addWidget(self._ops_panel)
        
        self._splitter.setStretchFactor(0, 0) # channel
        self._splitter.setStretchFactor(1, 1) # canvas
        self._splitter.setStretchFactor(2, 0) # operations
        self._splitter.setSizes([300, 680, 420])
        self._splitter.setCollapsible(2, False)
        
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

        file_menu.addSeparator()
        
        load_masks_act = QAction("&Load Masks…", self)
        load_masks_act.triggered.connect(lambda: self._on_import_masks(target="mask"))
        file_menu.addAction(load_masks_act)

        load_cells_act = QAction("Load &Cells…", self)
        load_cells_act.triggered.connect(lambda: self._on_import_masks(target="cell"))
        file_menu.addAction(load_cells_act)

        load_phenos_act = QAction("Load &Phenotyping…", self)
        load_phenos_act.triggered.connect(self._on_import_phenotypes)
        file_menu.addAction(load_phenos_act)

        file_menu.addSeparator()

        save_masks_act = QAction("&Save Masks…", self)
        save_masks_act.triggered.connect(lambda: self._on_export_masks(target="mask"))
        file_menu.addAction(save_masks_act)

        save_cells_act = QAction("Save &Cells…", self)
        save_cells_act.triggered.connect(lambda: self._on_export_masks(target="cell"))
        file_menu.addAction(save_cells_act)

        save_contours_act = QAction("Save Con&tours (GeoJSON)…", self)
        save_contours_act.triggered.connect(self._on_export_contours)
        file_menu.addAction(save_contours_act)

        save_phenos_act = QAction("Save &Phenotyping…", self)
        save_phenos_act.triggered.connect(self._on_export_phenotypes)
        file_menu.addAction(save_phenos_act)
        
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.ome.tiff *.tiff *.tif *.png *.jpg);;All files (*)")
        if path: self._load_image(path)

    def _load_image(self, path: str):
        try:
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
                        import cv2
                        filter_type = params["filter_type"]
                        filter_value = params["filter_value"]
                        
                        if filter_type == 'median':
                            suffix = "_Median"
                            # Use a disk/elliptical footprint for smoother results
                            footprint = disk(max(1, filter_value // 2))
                            x = ndi.median_filter(data, footprint=footprint, mode='reflect')
                        elif filter_type == 'opening':
                            suffix = "_Open"
                            # The user expects this filter to perform noise removal (subtraction).
                            # Mathematically, this is an Opening operation.
                            x = opening(data, footprint=disk(max(1, filter_value // 2)))
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
                    else:
                        final_labels = out_labels

                    contour_data = self._get_contour_data(final_labels.astype(np.int32))
                    self.segmentationResultReady.emit(final_labels, out_name, out_is_cell, None, contour_data, "", False, target_idx, None, True, None)

                if method == "watershed":
                    from skimage.filters import gaussian
                    from skimage.filters import threshold_local as sk_threshold_local
                    from skimage.filters import threshold_otsu as sk_threshold_otsu
                    from skimage.measure import label as sk_label
                    from skimage.morphology import local_maxima
                    from skimage.segmentation import expand_labels, watershed as sk_watershed

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
                        blurred_spots = gaussian(nuclei_gauss, spot_sigma)
                        spot_centroids = local_maxima(blurred_spots)
                        blurred_outline = gaussian(nuclei_gauss, outline_sigma)
                        blurred_outline = (
                            (blurred_outline - blurred_outline.min())
                            / (blurred_outline.max() - blurred_outline.min() + 1e-8)
                            * 255
                        )
                        sk_thresh = sk_threshold_local(blurred_outline, 101, offset=0)
                        thresh_offset = (threshold - 1.0) * 255 * 0.1
                        binary_otsu = blurred_outline > sk_thresh + thresh_offset
                        remaining_spots = spot_centroids * binary_otsu
                        labeled_spots = sk_label(remaining_spots)
                        labels = sk_watershed(binary_otsu, labeled_spots, mask=binary_otsu)
                    elif labeller == "gauss":
                        blurred_outline = gaussian(nuclei_gauss, outline_sigma)
                        sk_thresh = sk_threshold_otsu(blurred_outline)
                        binary_otsu = blurred_outline > threshold * sk_thresh
                        labels = sk_label(binary_otsu)

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
                    
                    worker_params = params.copy()
                    worker_params["override_name"] = override_name
                    
                    proc = ctx.Process(target=run_segmentation_task_pipe, args=(child_conn, worker_params, input_channels_data))
                    proc.start()
                    
                    try:
                        worker_res = parent_conn.recv()
                    except EOFError:
                        raise Exception("Worker process terminated unexpectedly.")
                    finally:
                        proc.join()
                        parent_conn.close()
                    
                    if not worker_res.get("success"):
                        raise Exception(f"Worker Error: {worker_res.get('error')}\n{worker_res.get('traceback')}")
                    
                    for out_labels, out_name, out_is_cell in worker_res.get("results", []):
                        process_and_emit(out_labels, out_name, out_is_cell)
                print(f"[Segmentation] Result emitted.")
            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

        self._segmentation_thread = threading.Thread(target=_run, daemon=True)
        self._segmentation_thread.start()

    def _on_segmentation_complete(self, labels, name, is_cell_mask, color, contour_data, source_marker, is_type_mask, target_idx, pos_lut=None, random_colors=True, aux_labels=None):
        self._ops_panel.stop_loading()
        
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
                    joint_mask, method_mask = UBM(carray).form_um(merit=merit, nsize=80)
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
        mask_ch = self._channel_model.channel(mask_idx)
        labels = mask_ch.mask_data
        if labels is None:
            self._ops_panel.stop_loading()
            return

        def _run():
            try:
                from opal_studio.image_loader import _get_yx
                from skimage.filters import threshold_otsu

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
                
                # Update labels sent to UI so _apply knows which one to use
                labels_for_ui = working_labels

                max_label = int(working_labels.max())

                cell_means: dict[int, np.ndarray] = {}
                otsu_thresholds: dict[int, float] = {}

                h, w = _get_yx(self._image.base_shape, self._image.axes, self._image.is_rgb)

                for i in range(self._channel_model.rowCount()):
                    ch = self._channel_model.channel(i)
                    if ch.is_mask or ch.is_cell_mask or ch.is_type_mask:
                        continue  # skip mask channels; include raw and processed image channels

                    if ch.is_processed and ch.processed_data is not None:
                        data = ch.processed_data.astype(np.float32)
                    else:
                        data = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)

                    # Crop/pad to label map size just in case
                    dh, dw = data.shape[:2]
                    crop_h, crop_w = min(h, dh), min(w, dw)
                    lab_crop = working_labels[:crop_h, :crop_w]
                    dat_crop = data[:crop_h, :crop_w]

                    # Vectorised: one scipy call returns mean for every label ID
                    means_list = ndi.mean(dat_crop, labels=lab_crop, index=cell_ids)
                    if not isinstance(means_list, np.ndarray):
                        means_list = np.array([means_list])
                    
                    # Clean NaNs (labels that weren't found in the crop)
                    means_list = np.nan_to_num(means_list)

                    # Build LUT indexed by label value (index 0 = background = 0)
                    means_lut = np.zeros(max_label + 1, dtype=np.float32)
                    means_lut[cell_ids] = means_list

                    cell_means[i] = means_lut

                    # Otsu on the per-cell means (ignore background zeros)
                    valid_means = means_lut[cell_ids]
                    valid_means = valid_means[valid_means > 0]
                    if valid_means.size > 1:
                        try:
                            val = threshold_otsu(valid_means)
                            if not np.isnan(val):
                                otsu_thresholds[i] = float(val)
                        except Exception:
                            pass
                
                # Deliver to main thread
                self.thresholdMeansReady.emit(working_labels, cell_means, otsu_thresholds, mask_idx)

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    @Slot(object, object, object, int)
    def _on_threshold_means_ready(self, labels, cell_means, otsu_thresholds, mask_model_idx):
        """Deliver computed means to the Thresholds tab (main thread)."""
        self._ops_panel.stop_loading()
        self._ops_panel._thresh_tab.receive_means(labels, cell_means, otsu_thresholds, mask_model_idx)

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

        if target_ch_idx != -1:
            # Update existing channel in-place
            try:
                tgt_ch = self._channel_model.channel(target_ch_idx)
                tgt_ch.mask_data = working_labels   # Use working labels
                tgt_ch.pos_lut = pos_lut    # Store states separately
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

    def _run_cell_identification(self):
        """Perform logical gating based on phenotyping definitions in a background thread."""
        if not self._image: return


        definitions = self._phenotyping_tab.get_phenotype_definitions()
        if not definitions:
            self._ops_panel.stop_loading()
            self.statusBar().showMessage("No cell types defined in the Phenotyping tab.", 5000)
            return

        cell_masks = {}
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_cell_mask:
                key = ch.source_marker if ch.source_marker else ch.name
                cell_masks[key] = ch.mask_data

        if not cell_masks:
            self._ops_panel.stop_loading()
            self.statusBar().showMessage("No cell positivity masks found. Detect marker positivity first.", 5000)
            return

        def _run():
            try:
                some_mask = next(iter(cell_masks.values()))
                h, w = some_mask.shape
                
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
                        if marker_name in cell_masks:
                            target_val = 2 if required_state == 1 else 1
                            valid_mask &= (cell_masks[marker_name] == target_val)
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
                        self.segmentationResultReady.emit(type_data, type_name, False, row_color, None, "", True, -1)
                        identified_count += 1
                    
                    self.operationProgress.emit(i+1, total_types)

                # Add Unknown phenotype for cells that matched nothing
                is_cell = (some_mask > 0)
                unknown_mask = is_cell & (~any_identified_mask)
                if np.any(unknown_mask):
                    unknown_data = np.zeros((h, w), dtype=np.uint8)
                    unknown_data[unknown_mask] = 1
                    self.segmentationResultReady.emit(unknown_data, "Unknown", False, QColor(128, 128, 128), None, "", True, -1)
                    identified_count += 1

                self.operationFinished.emit(identified_count)

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(f"Identification Error: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _on_operation_complete(self, count):
        self._ops_panel.stop_loading()
        # count might be total markers or total types
        self.statusBar().showMessage(f"Task complete. Processed {count} items.", 5000)

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
