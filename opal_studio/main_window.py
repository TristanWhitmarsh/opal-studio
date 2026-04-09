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
import numpy as np
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
        self._splitter.setSizes([300, 800, 320])
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
                        visible=(i < 6), data_min=float(dmin), data_max=float(dmax), index=i
                    ))
                self._channel_model.set_channels(channels)

            self._canvas.set_image(img)
            self._status.showMessage(f"Loaded: {Path(path).name}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image: {e}")

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
                from skimage import exposure
                from csbdeep.utils import normalize
                idx = params["channel_index"]
                ch = self._channel_model.channel(idx)
                data = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)
                x = normalize(data, params["p_low"], params["p_high"], axis=(0, 1))
                x = np.clip(x, 0.0, 1.0)
                if params["apply_clahe"]:
                    x = exposure.equalize_adapthist(x, kernel_size=params["clahe_kernel"], clip_limit=params["clahe_clip"]).astype(np.float32)
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
                    expand = params.get("expand", 2)

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

                    if expand > 0:
                        labels = expand_labels(labels, distance=expand)

                self.segmentationResultReady.emit(labels, "", False, None)
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
        mask_idx = params["mask_index"]
        tool = params.get("tool", "expansion_watershed")
        
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
                
                print(f"[AI] Starting cell positivity detection for mask: {original_mask_ch.name}")
                
                model_path = os.path.join(os.getcwd(), "models", "cellpos", "marker_cnn_epoch_100.h5")
                if not os.path.exists(model_path):
                    print(f"[AI] ERROR: Model not found at {model_path}")
                    self.segmentationError.emit(f"Model not found: {model_path}")
                    return
                
                print(f"[AI] Loading model: {model_path}")
                model = tf.keras.models.load_model(model_path)
                
                PATCH_SIZE = 64
                THRESHOLD = 0.5
                CHUNK_SIZE = 512
                
                # Get all Standard channels (physical intensity channels)
                target_channels = []
                for i in range(self._channel_model.rowCount()):
                    ch = self._channel_model.channel(i)
                    if not ch.is_mask and not ch.is_cell_mask:
                        target_channels.append(ch)
                
                print(f"[AI] Found {len(target_channels)} channels to process.")
                
                # If mask is binary (0/1), label it to get unique cells
                unique_labels = np.unique(labels)
                if len(unique_labels) == 2 and 0 in unique_labels:
                    print("[AI] Input mask is binary, labeling components...")
                    labeled_mask, num_cells = cc_label(labels > 0)
                else:
                    print("[AI] Input mask is already labeled.")
                    labeled_mask = labels
                    num_cells = int(np.max(labeled_mask))
                
                print(f"[AI] Processing {num_cells} cells...")
                if num_cells == 0:
                    self.segmentationError.emit("No cells found in selected mask.")
                    return

                for ch_idx, ch in enumerate(target_channels):
                    print(f"[AI] Processing channel {ch_idx+1}/{len(target_channels)}: {ch.name}")
                    # Load current channel data
                    img_curr = self._image.get_full_channel_data(ch.index, level=0).astype(np.float32)
                    # For 2D Opal, simulate neighbor slices
                    img_prev = img_curr
                    img_next = img_curr
                    
                    inference_mask = np.zeros_like(labeled_mask, dtype=np.int32)
                    patches = []
                    cell_ids = []
                    
                    for cid in range(1, num_cells + 1):
                        cell_mask_full = (labeled_mask == cid).astype(np.float32)
                        ys, xs = np.where(cell_mask_full > 0.5)
                        if ys.size == 0: continue
                        
                        cy, cx = int(np.mean(ys)), int(np.mean(xs))
                        half = PATCH_SIZE // 2
                        H, W = img_curr.shape
                        
                        y0 = max(0, cy - half); x0 = max(0, cx - half)
                        y1 = min(H, y0 + PATCH_SIZE); x1 = min(W, x0 + PATCH_SIZE)
                        if y1 - y0 < PATCH_SIZE: y0 = max(0, y1 - PATCH_SIZE)
                        if x1 - x0 < PATCH_SIZE: x0 = max(0, x1 - PATCH_SIZE)
                        y1 = min(H, y0 + PATCH_SIZE); x1 = min(W, x0 + PATCH_SIZE)
                        
                        p_curr = img_curr[y0:y1, x0:x1]
                        p_prev = img_prev[y0:y1, x0:x1]
                        p_next = img_next[y0:y1, x0:x1]
                        p_mask = cell_mask_full[y0:y1, x0:x1]
                        
                        # Pad if necessary
                        if p_curr.shape != (64, 64):
                            p_curr = np.pad(p_curr, ((0, 64-p_curr.shape[0]), (0, 64-p_curr.shape[1])))
                            p_prev = np.pad(p_prev, ((0, 64-p_prev.shape[0]), (0, 64-p_prev.shape[1])))
                            p_next = np.pad(p_next, ((0, 64-p_next.shape[0]), (0, 64-p_next.shape[1])))
                            p_mask = np.pad(p_mask, ((0, 64-p_mask.shape[0]), (0, 64-p_mask.shape[1])))

                        stacked = np.stack([p_prev, p_curr, p_next, p_mask], axis=-1)
                        patches.append(stacked)
                        cell_ids.append(cid)
                        
                        if len(patches) >= CHUNK_SIZE:
                            X = np.stack(patches, axis=0)
                            probs = model.predict(X, batch_size=512, verbose=0).reshape(-1)
                            for cid_inner, p in zip(cell_ids, probs):
                                # label_val = 2 if p >= THRESHOLD else 1
                                label_val = 2 if p >= THRESHOLD else 1
                                inference_mask[labeled_mask == cid_inner] = label_val
                            patches = []; cell_ids = []
                            
                    if patches:
                        X = np.stack(patches, axis=0)
                        probs = model.predict(X, batch_size=len(patches), verbose=0).reshape(-1)
                        for cid_inner, p in zip(cell_ids, probs):
                            label_val = 2 if p >= THRESHOLD else 1
                            inference_mask[labeled_mask == cid_inner] = label_val
                    
                    print(f"[AI] Raw inference mask for {ch.name}: min={np.min(inference_mask)}, max={np.max(inference_mask)}")
                    
                    pos_count = np.count_nonzero(inference_mask)
                    print(f"[DEBUG-STAT] final binary mask for {ch.name}: min={np.min(inference_mask)}, max={np.max(inference_mask)}, positive_cells={pos_count}")
                    self.segmentationResultReady.emit(inference_mask, f"{ch.name} Pos Cells", True, ch.color)

                print("[AI] Cell positivity detection complete.")

            except Exception as e:
                import traceback; traceback.print_exc()
                self.segmentationError.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def _on_segmentation_error(self, message):
        self._ops_panel.stop_loading()
        QMessageBox.critical(self, "Segmentation Error", message)

    def closeEvent(self, event):
        super().closeEvent(event)
