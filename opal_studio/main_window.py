"""
Main application window — assembles the three-panel layout.
Handles image loading, coordinate mapping, and background operations.
"""

from __future__ import annotations

import sys
import os
import threading
from pathlib import Path
import random
import colorsys
import xml.etree.ElementTree as ET
import numpy as np
import tifffile
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QSize
from PySide6.QtGui import QAction, QColor, QPixmap, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QSplitter, QWidget,
    QHBoxLayout, QStatusBar, QMessageBox, QTabWidget
)

from opal_studio.channel_model import Channel, ChannelListModel
from opal_studio.image_loader import ImageData, open_image, get_tile, _get_yx
from opal_studio.widgets.channel_panel import ChannelPanel
from opal_studio.widgets.image_canvas import ImageCanvas
from opal_studio.widgets.operations_panel import OperationsPanel
from opal_studio.widgets.phenotyping_tab import PhenotypingTab

from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed, find_boundaries

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

    # 4. Apply watershed
    expanded_labels = watershed(
        elevation,
        markers=markers,
        mask=expansion_mask,
        watershed_line=True
    )
    
    return expanded_labels > 0


class MainWindow(QMainWindow):
    """Top-level window for Opal Studio."""

    # Signals for cross-thread communication
    segmentationResultReady = Signal(object, str, bool, object) # labels, name, is_cell_mask, color
    segmentationError = Signal(str)
    preprocessingResultReady = Signal(str, object)
    preprocessingError = Signal(str)

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
        spacer_pixmap = QPixmap(1, 20)
        spacer_pixmap.fill(Qt.GlobalColor.transparent)
        spacer_icon = QIcon(spacer_pixmap)
        
        self._center_tabs.setIconSize(QSize(1, 20))
        self._center_tabs.addTab(self._canvas, spacer_icon, "Image")
        self._center_tabs.addTab(self._phenotyping_tab, spacer_icon, "Phenotyping")
        self._splitter.addWidget(self._center_tabs)
        
        self._splitter.addWidget(self._ops_panel)
        
        self._splitter.setStretchFactor(0, 0) # channel
        self._splitter.setStretchFactor(1, 1) # canvas
        self._splitter.setStretchFactor(2, 0) # operations
        self._splitter.setSizes([300, 760, 340])
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
                        visible=(i == 0), data_min=float(dmin), data_max=float(dmax), index=i
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
                    
                    new_ch = Channel(
                        name=clean_name,
                        color=QColor(255, 255, 255),
                        visible=True,
                        is_mask=(not final_is_cell),
                        is_cell_mask=final_is_cell,
                        mask_data=mask_data,
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
                idx = params["channel_index"]
                ch = self._channel_model.channel(idx)
                data = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)
                
                if params.get("is_filter"):
                    import scipy.signal
                    import cv2
                    filter_type = params["filter_type"]
                    filter_value = params["filter_value"]
                    
                    if filter_type == 'median':
                        # Ensure odd size for medfilt2d
                        size = filter_value if filter_value % 2 != 0 else filter_value + 1
                        x = scipy.signal.medfilt2d(data, size)
                    elif filter_type == 'tophat':
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (filter_value, filter_value))
                        tophat_data = cv2.morphologyEx(data, cv2.MORPH_TOPHAT, kernel)
                        x = data - tophat_data
                    else:
                        x = data
                else:
                    from csbdeep.utils import normalize
                    from skimage import exposure
                    
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
                
                self.preprocessingResultReady.emit(f"Processed: {ch.name}", x)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.preprocessingError.emit(str(e))
        threading.Thread(target=_run, daemon=True).start()

    def _on_preprocessing_complete(self, name, data):
        self._ops_panel.stop_loading()
        new_ch = Channel(
            name=name, color=QColor(255, 255, 255), visible=True,
            data_min=float(np.min(data)), data_max=float(np.max(data)),
            is_processed=True, processed_data=data, index=-1
        )
        self._channel_model.add_channel(new_ch)
        self._status.showMessage(f"Generated: {name}", 3000)

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
                override_name = ""
                x = None
                for idx in indices:
                    ch = self._channel_model.channel(idx)
                    if ch.is_processed and ch.processed_data is not None:
                        data = ch.processed_data
                    else:
                        raw = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)
                        p = np.percentile(raw, (1, 99.8))
                        data = np.clip((raw - p[0]) / (p[1] - p[0] + 1e-6), 0, 1)
                    x = data if x is None else x + data
                
                if method == "stardist":
                    from stardist.models import StarDist2D
                    model_folder = params.get("model_folder", params["model_name"])
                    model_path = os.path.join(os.getcwd(), "models", model_folder)
                    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                        model = StarDist2D(None, name=model_folder, basedir='models')
                    else:
                        model = StarDist2D.from_pretrained(params["model_name"])

                    kwargs = {"nms_thresh": params["nms_thresh"]}
                    if not params["use_default_thresh"]: kwargs["prob_thresh"] = params["prob_thresh"]
                    labels, _ = model.predict_instances(x, **kwargs)
                elif method == "cellpose":
                    from cellpose import models
                    if params.get("model_path"):
                        model = models.CellposeModel(pretrained_model=params["model_path"])
                    else:
                        model = models.Cellpose(model_type=params["model_name"])
                    
                    channels = [[0, 0]]
                    result = model.eval([x], diameter=params.get("diameter"), channels=channels)
                    labels = result[0][0]
                elif method == "instanseg":
                    from instanseg import InstanSeg
                    import torch
                    model = InstanSeg(params["model_name"])
                    x_input = x
                    if "brightfield" in params["model_name"].lower() and x.ndim == 2:
                        x_input = np.stack([x]*3, axis=-1)
                    
                    out, _ = model.eval_small_image(x_input, params["pixel_size"])
                    
                    labels_tensor = out[0]
                    if hasattr(labels_tensor, 'cpu'):
                        labels_tensor = labels_tensor.cpu().numpy()
                    elif hasattr(labels_tensor, 'numpy'):
                        labels_tensor = labels_tensor.numpy()
                        
                    labels_tensor = np.squeeze(labels_tensor)
                    if labels_tensor.ndim == 3:
                        # Process extra channels like cells before proceeding with main labels (nuclei)
                        for c in range(1, labels_tensor.shape[0]):
                           self.segmentationResultReady.emit(labels_tensor[c], "InstanSeg Cells", True, None)
                        labels = labels_tensor[0]
                        override_name = "InstanSeg Nuclei"
                    else:
                        labels = labels_tensor
                        override_name = "InstanSeg Nuclei"
                elif method == "watershed":
                    from scipy import ndimage as ndi
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
                        # Blur and detect local maxima
                        blurred_spots = gaussian(nuclei_gauss, spot_sigma)
                        spot_centroids = local_maxima(blurred_spots)
                        # Blur and threshold
                        blurred_outline = gaussian(nuclei_gauss, outline_sigma)
                        # Normalise to [0, 255] for local thresholding
                        blurred_outline = (
                            (blurred_outline - blurred_outline.min())
                            / (blurred_outline.max() - blurred_outline.min() + 1e-8)
                            * 255
                        )
                        sk_thresh = sk_threshold_local(blurred_outline, 101, offset=0)
                        # threshold acts as an additive offset: >1 makes segmentation stricter (fewer cells),
                        # <1 makes it more permissive. Convert to 0-255 scale (range centre = 1.0).
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

                self.segmentationResultReady.emit(labels, override_name, False, None)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

        self._segmentation_thread = threading.Thread(target=_run, daemon=True)
        self._segmentation_thread.start()

    def _on_segmentation_complete(self, labels, name=None, is_cell_mask=False, color=None):
        self._ops_panel.stop_loading()
        mask_name = name if name else f"Mask {self._channel_model.rowCount() + 1}"
        row_color = color if color else QColor(255, 255, 255)
        
        print(f"[DEBUG-STAT] RECEIVED MASK: name={mask_name}, is_cell_mask={is_cell_mask}, min={np.min(labels)}, max={np.max(labels)}")
        
        new_ch = Channel(
            name=mask_name, color=row_color, visible=True,
            is_mask=(not is_cell_mask), is_cell_mask=is_cell_mask,
            mask_data=labels.astype(np.int32), index=-1
        )
        self._channel_model.add_channel(new_ch)
        self._status.showMessage(f"Task Complete: {mask_name}", 5000)

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
                    new_name = f"Sampled Mask ({merit})"
                    
                    self.segmentationResultReady.emit(mask_result, new_name, False, None)
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
                    mask_result = expanded.astype(np.int32)
                    new_name = f"{original_mask_ch.name} +Exp{expansion_pixels}"
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
                    
                    new_name = f"{original_mask_ch.name} Filtered"
                
                self.segmentationResultReady.emit(mask_result, new_name, original_mask_ch.is_cell_mask, original_mask_ch.color)
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
                import tensorflow as tf
                from scipy.ndimage import label as cc_label
                from opal_studio.image_loader import _get_yx
                
                print(f"[AI] Starting cell positivity detection for mask: {original_mask_ch.name}")
                
                model_path = os.path.join(os.getcwd(), "models", "cellpos", "marker_cnn_epoch_100.h5")
                if not os.path.exists(model_path):
                    self.segmentationError.emit(f"Model not found: {model_path}")
                    return
                
                print(f"[AI] Loading model: {model_path}")
                model = tf.keras.models.load_model(model_path)
                
                PATCH_SIZE = 64
                THRESHOLD = 0.5
                CHUNK_SIZE = 1024
                
                target_channels = []
                for i in range(self._channel_model.rowCount()):
                    ch = self._channel_model.channel(i)
                    if not ch.is_mask and not ch.is_cell_mask:
                        target_channels.append(ch)
                
                S = len(target_channels)
                h, w = _get_yx(self._image.base_shape, self._image.axes, self._image.is_rgb)
                markers = np.zeros((h, w, S), dtype=np.float32)
                for z, ch in enumerate(target_channels):
                    data = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)
                    dh, dw = data.shape[:2]
                    copy_h = min(h, dh)
                    copy_w = min(w, dw)
                    markers[:copy_h, :copy_w, z] = data[:copy_h, :copy_w]

                def get_neighbor_slices(img_data, curr_z):
                    Z = img_data.shape[2]
                    if curr_z == 0:
                        z_prev = curr_z
                        z_next = curr_z + 1 if Z > 1 else curr_z
                    elif curr_z == Z - 1:
                        z_prev = curr_z - 1
                        z_next = curr_z
                    else:
                        z_prev = curr_z - 1
                        z_next = curr_z + 1
                    return img_data[:, :, z_prev], img_data[:, :, curr_z], img_data[:, :, z_next]

                def center_crop_on_mask(img, mask, crop_size=64):
                    H, W = img.shape[:2]
                    ys, xs = np.where(mask > 0.5)
                    if ys.size == 0:
                        return None, None
                    cy = int(np.mean(ys))
                    cx = int(np.mean(xs))
                    half = crop_size // 2
                    y0 = max(0, cy - half)
                    x0 = max(0, cx - half)
                    y1 = min(H, y0 + crop_size)
                    x1 = min(W, x0 + crop_size)
                    if y1 - y0 < crop_size: y0 = max(0, y1 - crop_size)
                    if x1 - x0 < crop_size: x0 = max(0, x1 - crop_size)
                    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

                for z in range(S):
                    ch = target_channels[z]
                    print(f"\nProcessing slice/channel {z+1}/{S}")
                    img_prev, img_curr, img_next = get_neighbor_slices(markers, z)
                    
                    cell_region = labels > 0
                    labeled_slice, num_components = cc_label(cell_region)
                    
                    if num_components == 0:
                        continue
                        
                    slice_out = np.zeros_like(labels, dtype=np.int16)
                    patches = []
                    cell_ids = []
                    
                    def flush_chunk():
                        if not patches: return
                        X = np.stack(patches, axis=0).astype(np.float32)
                        probs = model.predict(X, batch_size=512, verbose=0).reshape(-1)
                        for cid, p in zip(cell_ids, probs):
                            label_val = 2 if p >= THRESHOLD else 1
                            slice_out[labeled_slice == cid] = label_val
                        patches.clear()
                        cell_ids.clear()
                        
                    for comp_id in range(1, num_components + 1):
                        cell_mask_full = (labeled_slice == comp_id).astype(np.float32)
                        img_curr_patch, cell_patch = center_crop_on_mask(img_curr, cell_mask_full, PATCH_SIZE)
                        if img_curr_patch is None: continue
                        
                        ys, xs = np.where(cell_mask_full > 0.5)
                        cy = int(np.mean(ys))
                        cx = int(np.mean(xs))
                        half = PATCH_SIZE // 2
                        Hc, Wc = img_curr.shape
                        y0 = max(0, cy - half)
                        x0 = max(0, cx - half)
                        y1 = min(Hc, y0 + PATCH_SIZE)
                        x1 = min(Wc, x0 + PATCH_SIZE)
                        if y1 - y0 < PATCH_SIZE: y0 = max(0, y1 - PATCH_SIZE)
                        if x1 - x0 < PATCH_SIZE: x0 = max(0, x1 - PATCH_SIZE)
                        
                        img_prev_patch = img_prev[y0:y1, x0:x1]
                        img_next_patch = img_next[y0:y1, x0:x1]
                        
                        stacked = np.stack([img_prev_patch, img_curr_patch, img_next_patch, cell_patch], axis=-1)
                        patches.append(stacked)
                        cell_ids.append(comp_id)
                        
                        if len(patches) >= CHUNK_SIZE:
                            flush_chunk()
                            
                    flush_chunk()
                    
                    self.segmentationResultReady.emit(slice_out, f"{ch.name}", True, ch.color)

                print("[AI] Cell positivity detection complete.")

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

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
