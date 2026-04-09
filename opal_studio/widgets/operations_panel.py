"""
Right-side operations panel.
Includes expandable sections for image pre-processing and multi-engine segmentation.
"""

from __future__ import annotations

import os
import numpy as np
import threading
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy,
    QComboBox, QLineEdit, QPushButton, QProgressBar,
    QFormLayout, QMessageBox, QCheckBox,
    QHBoxLayout, QScrollArea, QTabWidget, QToolButton
)
from PySide6.QtGui import QDoubleValidator, QIntValidator, QPixmap, QIcon

from opal_studio.channel_model import ChannelListModel, Channel

class CollapsiblePanel(QWidget):
    """A simple collapsible section with a native QToolButton toggle."""
    def __init__(self, title, collapsed=True, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._btn = QToolButton()
        self._btn.setText(title)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._btn.setAutoRaise(True)
        self._btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self._is_expanded = not collapsed
        self._btn.setArrowType(Qt.ArrowType.DownArrow if self._is_expanded else Qt.ArrowType.RightArrow)
        self._btn.clicked.connect(self._toggle)
        self._layout.addWidget(self._btn)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 10, 10, 10)
        self._content_layout.setSpacing(10)
        self._layout.addWidget(self._content)
        
        self._content.setVisible(self._is_expanded)

    def _toggle(self):
        self._is_expanded = not self._is_expanded
        self._content.setVisible(self._is_expanded)
        self._btn.setArrowType(Qt.ArrowType.DownArrow if self._is_expanded else Qt.ArrowType.RightArrow)

    def addWidget(self, widget):
        self._content_layout.addWidget(widget)
    
    def addLayout(self, layout):
        self._content_layout.addLayout(layout)

class StarDistTab(QWidget):
    """Sub-widget for StarDist segmentation parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(10)

        # Model Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setMinimumWidth(80)
        self._channel_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        form.addRow("Channel:", self._channel_combo)

        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(80)
        self._model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        self._model_combo.addItems(["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"])
        self._scan_models()
        form.addRow("Model:", self._model_combo)

        self._default_thresh_cb = QCheckBox("Use Default Prob")
        self._default_thresh_cb.setChecked(True)
        form.addRow("", self._default_thresh_cb)

        self._prob_thresh = QLineEdit("0.5")
        self._prob_thresh.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self._prob_thresh.setFixedWidth(80)
        form.addRow("Prob Match:", self._prob_thresh)

        self._nms_thresh = QLineEdit("0.3")
        self._nms_thresh.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self._nms_thresh.setFixedWidth(80)
        form.addRow("NMS Thresh:", self._nms_thresh)
        
        self._prob_thresh.setEnabled(False)  # checkbox starts checked → field starts disabled
        self._default_thresh_cb.toggled.connect(lambda checked: self._prob_thresh.setEnabled(not checked))
        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Run StarDist")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        self._refresh_channels()
        self._channel_model.modelReset.connect(self._refresh_channels)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_channels())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_channels())

    def _refresh_channels(self):
        current = self._channel_combo.currentText()
        self._channel_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if not ch.is_mask:
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)
        

    def _scan_models(self):
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir): return
        for folder in os.listdir(models_dir):
            path = os.path.join(models_dir, folder)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                name = "Custom" if folder.lower() == "stardist" else folder
                if self._model_combo.findText(name) == -1:
                    self._model_combo.addItem(name, folder)

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        self.runRequested.emit({
            "method": "stardist",
            "channel_indices": [self._channel_combo.currentData()],
            "model_name": self._model_combo.currentText(),
            "model_folder": self._model_combo.currentData() or self._model_combo.currentText(),
            "use_default_thresh": self._default_thresh_cb.isChecked(),
            "prob_thresh": float(self._prob_thresh.text() or 0.5),
            "nms_thresh": float(self._nms_thresh.text() or 0.3),
        })

class CellposeTab(QWidget):
    """Sub-widget for Cellpose segmentation parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(10)

        # Model Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setMinimumWidth(80)
        self._channel_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        form.addRow("Channel:", self._channel_combo)

        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(80)
        self._model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        self._model_combo.addItems(["cyto", "nuclei", "cyto2"])
        self._scan_models()
        form.addRow("Model:", self._model_combo)

        self._diameter = QLineEdit()
        self._diameter.setPlaceholderText("Auto")
        self._diameter.setValidator(QDoubleValidator(0.1, 1000.0, 2))
        self._diameter.setFixedWidth(80)
        form.addRow("Diameter (px):", self._diameter)
        
        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Run Cellpose")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        self._refresh_channels()
        self._channel_model.modelReset.connect(self._refresh_channels)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_channels())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_channels())

    def _refresh_channels(self):
        current = self._channel_combo.currentText()
        self._channel_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if not ch.is_mask:
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)

    def _scan_models(self):
        models_dir = os.path.join(os.getcwd(), "models", "cellpose")
        if not os.path.exists(models_dir): return
        for file in os.listdir(models_dir):
            path = os.path.join(models_dir, file)
            if os.path.isfile(path) and not file.endswith('.txt') and not file.endswith('.ipynb'):
                self._model_combo.addItem(f"Custom: {file[:30]}...", path)

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        diam_text = self._diameter.text()
        diam = float(diam_text) if diam_text else None
        
        self.runRequested.emit({
            "method": "cellpose",
            "channel_indices": [self._channel_combo.currentData()],
            "model_name": self._model_combo.currentText(),
            "model_path": self._model_combo.currentData(),
            "diameter": diam,
        })

class WatershedTab(QWidget):
    """Sub-widget for Watershed segmentation parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(10)

        # Model Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setMinimumWidth(80)
        self._channel_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        form.addRow("Channel:", self._channel_combo)

        # Labeller mode
        self._labeller_combo = QComboBox()
        self._labeller_combo.setMinimumWidth(80)
        self._labeller_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        self._labeller_combo.addItems(["voronoi", "gauss"])
        self._labeller_combo.currentTextChanged.connect(self._on_labeller_changed)
        form.addRow("Labeller:", self._labeller_combo)

        # Spot sigma (only for voronoi)
        self._spot_sigma = QLineEdit("2")
        self._spot_sigma.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self._spot_sigma.setFixedWidth(80)
        self._spot_sigma_label = QLabel("Spot Sigma:")
        form.addRow(self._spot_sigma_label, self._spot_sigma)

        # Outline sigma
        self._outline_sigma = QLineEdit("2")
        self._outline_sigma.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self._outline_sigma.setFixedWidth(80)
        form.addRow("Outline Sigma:", self._outline_sigma)

        # Threshold
        self._threshold = QLineEdit("1")
        self._threshold.setValidator(QDoubleValidator(0.01, 100.0, 2))
        self._threshold.setFixedWidth(80)
        self._threshold.setToolTip(
            "Voronoi: offset from local threshold (>1 = stricter, <1 = more permissive).\n"
            "Gauss: multiplier on Otsu threshold (>1 = stricter, <1 = more permissive)."
        )
        form.addRow("Threshold:", self._threshold)

        # Min Mean Intensity
        self._min_mean_intensity = QLineEdit("0")
        self._min_mean_intensity.setValidator(QDoubleValidator(-1000000.0, 1000000.0, 4))
        self._min_mean_intensity.setFixedWidth(80)
        self._min_mean_intensity.setToolTip("Minimum mean intensity in a cell to keep it.")
        form.addRow("Min Intensity:", self._min_mean_intensity)

        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Run Watershed")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        self._refresh_channels()
        self._channel_model.modelReset.connect(self._refresh_channels)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_channels())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_channels())

    def _on_labeller_changed(self, text):
        is_voronoi = (text == "voronoi")
        self._spot_sigma.setVisible(is_voronoi)
        self._spot_sigma_label.setVisible(is_voronoi)

    def _refresh_channels(self):
        current = self._channel_combo.currentText()
        self._channel_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if not ch.is_mask:
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        self.runRequested.emit({
            "method": "watershed",
            "channel_indices": [self._channel_combo.currentData()],
            "labeller": self._labeller_combo.currentText(),
            "spot_sigma": float(self._spot_sigma.text() or 2),
            "outline_sigma": float(self._outline_sigma.text() or 2),
            "threshold": float(self._threshold.text() or 1),
            "min_mean_intensity": float(self._min_mean_intensity.text() or 0),
        })


class MaskFilterSizeTab(QWidget):
    """Sub-widget for filtering mask regions by size."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(10)

        # Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        # Mask Selector
        self._mask_combo = QComboBox()
        self._mask_combo.setMinimumWidth(80)
        self._mask_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        form.addRow("Input Mask:", self._mask_combo)
        
        self._min_size = QLineEdit("10")
        self._min_size.setValidator(QIntValidator(0, 1000000))
        self._min_size.setFixedWidth(80)
        form.addRow("Min Size (px):", self._min_size)

        self._max_size = QLineEdit("1000")
        self._max_size.setValidator(QIntValidator(0, 1000000))
        self._max_size.setFixedWidth(80)
        form.addRow("Max Size (px):", self._max_size)

        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Filter Sizes")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)
        
        self._refresh_masks()
        self._channel_model.modelReset.connect(self._refresh_masks)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_masks())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_masks())

    def _refresh_masks(self):
        current = self._mask_combo.currentText()
        self._mask_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_mask:
                self._mask_combo.addItem(ch.name, i)
        idx = self._mask_combo.findText(current)
        if idx >= 0: self._mask_combo.setCurrentIndex(idx)

    def _on_run(self):
        if self._mask_combo.currentIndex() < 0: return
        self.runRequested.emit({
            "mask_index": self._mask_combo.currentData(),
            "min_size": int(self._min_size.text() or 10),
            "max_size": int(self._max_size.text() or 1000),
            "tool": "filter_size"
        })

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._run_btn.setEnabled(enabled)


class MaskExpansionTab(QWidget):
    """Sub-widget for mask expansion parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(10)

        # Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        # Mask Selector
        self._mask_combo = QComboBox()
        self._mask_combo.setMinimumWidth(80)
        self._mask_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        form.addRow("Input Mask:", self._mask_combo)
        self._pixels = QLineEdit("6")
        self._pixels.setValidator(QIntValidator(1, 100))
        self._pixels.setFixedWidth(80)
        form.addRow("Expansion (pixels):", self._pixels)
        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Run Expansion")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)
        self._refresh_masks()
        self._channel_model.modelReset.connect(self._refresh_masks)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_masks())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_masks())

    def _refresh_masks(self):
        current = self._mask_combo.currentText()
        self._mask_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_mask:
                self._mask_combo.addItem(ch.name, i)
        idx = self._mask_combo.findText(current)
        if idx >= 0: self._mask_combo.setCurrentIndex(idx)

    def _on_run(self):
        if self._mask_combo.currentIndex() < 0: return
        self.runRequested.emit({
            "mask_index": self._mask_combo.currentData(),
            "expansion_pixels": int(self._pixels.text() or 6),
            "tool": "expansion_watershed"
        })

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._run_btn.setEnabled(enabled)

class OperationsPanel(QWidget):
    """Right-side panel with collapsible sections for Pre-processing and Segmentation."""

    WIDTH = 320
    runPreprocessingRequested = Signal(dict)
    runSegmentationRequested = Signal(dict)
    runMaskProcessingRequested = Signal(dict)
    runCellPositivityRequested = Signal(dict)
    segmentationFinished = Signal(object)

    def __init__(self, channel_model: ChannelListModel, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        self.setFixedWidth(self.WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        self._container_layout = QVBoxLayout(container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(0)
        
        # Inject an invisible native icon to force the tab bar to draw taller, preventing cut-off text
        spacer_pixmap = QPixmap(1, 20)
        spacer_pixmap.fill(Qt.GlobalColor.transparent)
        self._spacer_icon = QIcon(spacer_pixmap)

        # Sections — all collapsed by default
        self._setup_preprocessing_section()
        self._setup_segmentation_section()
        self._setup_mask_processing_section()
        self._setup_cell_positivity_section()
        self._setup_cell_identification_section()

        self._container_layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # Progress bar at the absolute bottom
        self._progress = QProgressBar()
        self._progress.setVisible(False); self._progress.setTextVisible(False)
        main_layout.addWidget(self._progress)

    def _setup_preprocessing_section(self):
        panel = CollapsiblePanel("Pre-processing", collapsed=True)
        self._container_layout.addWidget(panel)

        form = QFormLayout()
        form.setContentsMargins(0, 5, 0, 5)
        form.setSpacing(8)
        self._pre_target_combo = QComboBox()
        self._pre_target_combo.setMinimumWidth(80)
        self._pre_target_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        form.addRow("Channel:", self._pre_target_combo)

        # Percentiles
        self._p_low = QLineEdit("1.0"); self._p_low.setValidator(QDoubleValidator(0, 100, 2))
        self._p_high = QLineEdit("99.8"); self._p_high.setValidator(QDoubleValidator(0, 100, 2))
        self._p_low.setFixedWidth(80); self._p_high.setFixedWidth(80)
        
        p_lay = QHBoxLayout()
        p_lay.addWidget(self._p_low); p_lay.addWidget(QLabel("-")); p_lay.addWidget(self._p_high)
        p_lay.addStretch()
        form.addRow("Percentiles:", p_lay)

        # CLAHE
        self._clahe_cb = QCheckBox("Apply CLAHE")
        self._clahe_cb.setChecked(True)
        form.addRow("", self._clahe_cb)
        
        self._clahe_clip = QLineEdit("0.02")
        self._clahe_clip.setValidator(QDoubleValidator(0.001, 1.0, 3))
        self._clahe_clip.setFixedWidth(80)
        form.addRow("Clip Limit:", self._clahe_clip)
        
        self._clahe_kernel = QLineEdit("50")
        self._clahe_kernel.setValidator(QIntValidator(8, 256))
        self._clahe_kernel.setFixedWidth(80)
        form.addRow("Kernel Size:", self._clahe_kernel)

        self._clahe_cb.toggled.connect(lambda chk: [self._clahe_clip.setEnabled(chk), self._clahe_kernel.setEnabled(chk)])
        panel.addLayout(form)

        self._pre_run_btn = QPushButton("Run Pre-processing")
        self._pre_run_btn.clicked.connect(self._on_run_preprocessing)
        panel.addWidget(self._pre_run_btn)

        self._channel_model.modelReset.connect(self._refresh_pre_channels)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_pre_channels())
        self._refresh_pre_channels()

    def _setup_segmentation_section(self):
        panel = CollapsiblePanel("Segmentation", collapsed=True)
        self._container_layout.addWidget(panel)

        self._seg_tabs = QTabWidget()
        self._seg_tabs.setIconSize(QSize(1, 20))
        self._seg_tabs.setMinimumWidth(0)
        self._seg_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._stardist_tab = StarDistTab(self._channel_model)
        self._stardist_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._stardist_tab, self._spacer_icon, "StarDist")
        self._cellpose_tab = CellposeTab(self._channel_model)
        self._cellpose_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._cellpose_tab, self._spacer_icon, "Cellpose")
        self._watershed_tab = WatershedTab(self._channel_model)
        self._watershed_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._watershed_tab, self._spacer_icon, "Watershed")
        panel.addWidget(self._seg_tabs)

    def _setup_mask_processing_section(self):
        panel = CollapsiblePanel("Mask Processing", collapsed=True)
        self._container_layout.addWidget(panel)

        self._mask_tabs = QTabWidget()
        self._mask_tabs.setIconSize(QSize(1, 20))
        self._mask_tabs.setMinimumWidth(0)
        self._mask_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        
        self._filter_size_tab = MaskFilterSizeTab(self._channel_model)
        self._filter_size_tab.runRequested.connect(self._on_run_mask_processing)
        self._mask_tabs.addTab(self._filter_size_tab, self._spacer_icon, "Filter by Size")
        
        self._expansion_tab = MaskExpansionTab(self._channel_model)
        self._expansion_tab.runRequested.connect(self._on_run_mask_processing)
        self._mask_tabs.addTab(self._expansion_tab, self._spacer_icon, "Expansion")
        
        panel.addWidget(self._mask_tabs)

    def _setup_cell_positivity_section(self):
        panel = CollapsiblePanel("Cell positivity", collapsed=True)
        self._container_layout.addWidget(panel)

        form = QFormLayout()
        form.setContentsMargins(0, 5, 0, 5)
        form.setSpacing(8)
        self._pos_mask_combo = QComboBox()
        self._pos_mask_combo.setMinimumWidth(80)
        self._pos_mask_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        form.addRow("Input Mask:", self._pos_mask_combo)

        panel.addLayout(form)

        self._pos_run_btn = QPushButton("Detect Cell Positivity")
        self._pos_run_btn.clicked.connect(self._on_run_cell_positivity)
        panel.addWidget(self._pos_run_btn)

        self._channel_model.modelReset.connect(self._refresh_positivity_combos)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_positivity_combos())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_positivity_combos())
        self._refresh_positivity_combos()

    def _refresh_positivity_combos(self):
        # Refresh Mask Combo
        m_current = self._pos_mask_combo.currentText()
        self._pos_mask_combo.clear()

        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_mask:
                self._pos_mask_combo.addItem(ch.name, i)
        
        m_idx = self._pos_mask_combo.findText(m_current)
        if m_idx >= 0: self._pos_mask_combo.setCurrentIndex(m_idx)

    def _setup_cell_identification_section(self):
        panel = CollapsiblePanel("Cell identification", collapsed=True)
        self._container_layout.addWidget(panel)

        self._ident_run_btn = QPushButton("Identify Cells")
        panel.addWidget(self._ident_run_btn)

    def _refresh_pre_channels(self):
        current = self._pre_target_combo.currentText()
        self._pre_target_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if not ch.is_mask:
                self._pre_target_combo.addItem(ch.name, i)
        idx = self._pre_target_combo.findText(current)
        if idx >= 0: self._pre_target_combo.setCurrentIndex(idx)

    def _on_run_preprocessing(self):
        if self._pre_target_combo.currentIndex() < 0: return
        self._pre_run_btn.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runPreprocessingRequested.emit({
            "channel_index": self._pre_target_combo.currentData(),
            "p_low": float(self._p_low.text() or 1.0),
            "p_high": float(self._p_high.text() or 99.8),
            "apply_clahe": self._clahe_cb.isChecked(),
            "clahe_clip": float(self._clahe_clip.text() or 0.02),
            "clahe_kernel": (int(self._clahe_kernel.text() or 50), int(self._clahe_kernel.text() or 50))
        })

    def _on_run_segmentation(self, params):
        self._stardist_tab.setEnabled(False)
        self._cellpose_tab.setEnabled(False)
        self._watershed_tab.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runSegmentationRequested.emit(params)

    def _on_run_mask_processing(self, params):
        self._expansion_tab.setEnabled(False)
        self._filter_size_tab.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runMaskProcessingRequested.emit(params)

    def _on_run_cell_positivity(self):
        if self._pos_mask_combo.currentIndex() < 0:
            return
        self._pos_run_btn.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runCellPositivityRequested.emit({
            "mask_index": self._pos_mask_combo.currentData()
        })

    def stop_loading(self):
        self._pre_run_btn.setEnabled(True)
        self._stardist_tab.setEnabled(True)
        self._cellpose_tab.setEnabled(True)
        self._watershed_tab.setEnabled(True)
        self._filter_size_tab.setEnabled(True)
        self._expansion_tab.setEnabled(True)
        self._pos_run_btn.setEnabled(True)
        self._progress.setVisible(False)
