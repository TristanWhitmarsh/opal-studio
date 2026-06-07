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
    QFormLayout, QMessageBox, QCheckBox, QRadioButton, QButtonGroup,
    QHBoxLayout, QGridLayout, QScrollArea, QTabWidget, QToolButton,
    QListWidget, QListWidgetItem, QAbstractItemView, QPlainTextEdit
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
        self._content_layout.setContentsMargins(2, 5, 2, 5)
        self._content_layout.setSpacing(5)
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

class OperationsTabWidget(QTabWidget):
    """A QTabWidget that adjusts its height to match the current tab's contents."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setElideMode(Qt.TextElideMode.ElideNone)
        self.setUsesScrollButtons(True)
        self.setTabBarAutoHide(False)
        self.tabBar().setExpanding(False)
        # Enable wheel scrolling to cycle through tabs
        self.tabBar().wheelEvent = self._wheel_scroll_tabs
        self.currentChanged.connect(self._on_current_changed)

    def _wheel_scroll_tabs(self, event):
        if event.angleDelta().y() > 0:
            self.setCurrentIndex(max(0, self.currentIndex() - 1))
        else:
            self.setCurrentIndex(min(self.count() - 1, self.currentIndex() + 1))
        event.accept()

    def _on_current_changed(self, index):
        self.updateGeometry()
        # Force parent update if it's a collapsible panel content
        if self.parent() and self.parent().parent():
            self.parent().updateGeometry()

    def sizeHint(self):
        hint = super().sizeHint()
        if self.currentWidget():
            # Height = current tab height + tab bar height + margin
            h = self.currentWidget().sizeHint().height() + self.tabBar().sizeHint().height() + 8
            hint.setHeight(h)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setHeight(self.sizeHint().height())
        return hint

class FilterTab(QWidget):
    """Smoothing, normalization, and custom image filters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel_combo.setMinimumWidth(50)
        form.addRow("Channel:", self._channel_combo)

        self.filter_type = QComboBox()
        self.filter_type.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.filter_type.setMinimumWidth(50)
        self.filter_type.addItems([
            "Median",
            "Opening",
            "CLAHE",
            "Subtract Background",
            "Remove Hotpixels",
            "Intensity Rescale"
        ])
        self.filter_type.currentIndexChanged.connect(self._on_filter_type_changed)
        form.addRow("Type:", self.filter_type)
        
        layout.addLayout(form)

        # 1. Median / Opening container
        self.median_container = QWidget()
        med_lay = QFormLayout(self.median_container)
        med_lay.setContentsMargins(0, 0, 0, 0)
        med_lay.setSpacing(6)
        med_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        med_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        med_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.filter_value = QLineEdit("3")
        self.filter_value.setValidator(QIntValidator(1, 101))
        self.filter_value.setFixedWidth(60)
        med_lay.addRow("Size:", self.filter_value)
        layout.addWidget(self.median_container)

        # 2. Equalize container
        self.equalize_container = QWidget()
        eq_lay = QFormLayout(self.equalize_container)
        eq_lay.setContentsMargins(0, 0, 0, 0)
        eq_lay.setSpacing(6)
        eq_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        eq_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        eq_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        self.p_low = QLineEdit("1.0")
        self.p_low.setValidator(QDoubleValidator(0, 100, 2))
        self.p_high = QLineEdit("99.8")
        self.p_high.setValidator(QDoubleValidator(0, 100, 2))
        self.p_low.setFixedWidth(50)
        self.p_high.setFixedWidth(50)
        p_lay = QHBoxLayout()
        p_lay.addWidget(self.p_low)
        p_lay.addWidget(QLabel(" - "))
        p_lay.addWidget(self.p_high)
        p_lay.addStretch()
        eq_lay.addRow("Range:", p_lay)
        
        self.clahe_clip = QLineEdit("0.05")
        self.clahe_clip.setValidator(QDoubleValidator(0.001, 1.0, 3))
        self.clahe_clip.setFixedWidth(60)
        eq_lay.addRow("Clip:", self.clahe_clip)
        
        self.clahe_kernel = QLineEdit("20")
        self.clahe_kernel.setValidator(QIntValidator(8, 256))
        self.clahe_kernel.setFixedWidth(60)
        eq_lay.addRow("Kernel:", self.clahe_kernel)
        layout.addWidget(self.equalize_container)

        # 3. Subtract Background container
        self.subtract_bkg_container = QWidget()
        sub_lay = QFormLayout(self.subtract_bkg_container)
        sub_lay.setContentsMargins(0, 0, 0, 0)
        sub_lay.setSpacing(6)
        sub_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        sub_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        sub_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        self.bkg_sigma = QLineEdit("3")
        self.bkg_sigma.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self.bkg_sigma.setFixedWidth(60)
        sub_lay.addRow("Sigma:", self.bkg_sigma)
        
        self.bkg_radius = QLineEdit("15")
        self.bkg_radius.setValidator(QIntValidator(1, 1000))
        self.bkg_radius.setFixedWidth(60)
        sub_lay.addRow("Radius:", self.bkg_radius)
        layout.addWidget(self.subtract_bkg_container)

        # 4. Remove Hotpixels container
        self.remove_hotpixels_container = QWidget()
        hot_lay = QFormLayout(self.remove_hotpixels_container)
        hot_lay.setContentsMargins(0, 0, 0, 0)
        hot_lay.setSpacing(6)
        hot_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        hot_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        hot_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        self.hot_threshold = QLineEdit("10")
        self.hot_threshold.setValidator(QDoubleValidator(0.1, 1000.0, 2))
        self.hot_threshold.setFixedWidth(60)
        hot_lay.addRow("Threshold:", self.hot_threshold)
        
        self.hot_npass = QLineEdit("3")
        self.hot_npass.setValidator(QIntValidator(1, 20))
        self.hot_npass.setFixedWidth(60)
        hot_lay.addRow("Passes (npass):", self.hot_npass)
        
        self.hot_filter_size = QLineEdit("5")
        self.hot_filter_size.setValidator(QIntValidator(1, 101))
        self.hot_filter_size.setFixedWidth(60)
        hot_lay.addRow("Filter Size:", self.hot_filter_size)
        layout.addWidget(self.remove_hotpixels_container)

        # 5. Intensity Rescale container
        self.rescale_container = QWidget()
        rescale_lay = QFormLayout(self.rescale_container)
        rescale_lay.setContentsMargins(0, 0, 0, 0)
        rescale_lay.setSpacing(6)
        rescale_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        rescale_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        rescale_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        self.rescale_p1 = QLineEdit("2")
        self.rescale_p1.setValidator(QDoubleValidator(0, 100, 2))
        self.rescale_p1.setFixedWidth(60)
        rescale_lay.addRow("P1 percentile:", self.rescale_p1)
        
        self.rescale_p2 = QLineEdit("98")
        self.rescale_p2.setValidator(QDoubleValidator(0, 100, 2))
        self.rescale_p2.setFixedWidth(60)
        rescale_lay.addRow("P2 percentile:", self.rescale_p2)
        layout.addWidget(self.rescale_container)
        
        self._run_btn = QPushButton("Run Filter")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)
        layout.addStretch()

        self._refresh_channels()
        self._on_filter_type_changed() # Initialize visible parameters
        self._channel_model.modelReset.connect(self._refresh_channels)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_channels())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_channels())

    def _refresh_channels(self):
        current = self._channel_combo.currentText()
        self._channel_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if not ch.is_mask and not getattr(ch, "is_region", False):
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)

    def _on_filter_type_changed(self):
        text = self.filter_type.currentText().lower()
        
        # Hide all parameter containers
        self.median_container.setVisible(False)
        self.equalize_container.setVisible(False)
        self.subtract_bkg_container.setVisible(False)
        self.remove_hotpixels_container.setVisible(False)
        self.rescale_container.setVisible(False)
        
        if text in ["median", "opening"]:
            self.median_container.setVisible(True)
        elif text == "clahe":
            self.equalize_container.setVisible(True)
        elif text == "subtract background":
            self.subtract_bkg_container.setVisible(True)
        elif text == "remove hotpixels":
            self.remove_hotpixels_container.setVisible(True)
        elif text == "intensity rescale":
            self.rescale_container.setVisible(True)
            
        # Update widget and parent sizes
        self.updateGeometry()
        if self.parent():
            self.parent().updateGeometry()
            if self.parent().parent():
                self.parent().parent().updateGeometry()

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        
        filter_type = self.filter_type.currentText().lower()
        params = {
            "is_filter": True,
            "channel_index": self._channel_combo.currentData(),
            "filter_type": filter_type
        }
        
        if filter_type in ["median", "opening"]:
            params["filter_value"] = int(self.filter_value.text() or 3)
        elif filter_type == "clahe":
            params["p_low"] = float(self.p_low.text() or 1.0)
            params["p_high"] = float(self.p_high.text() or 99.8)
            params["apply_clahe"] = True
            params["clahe_clip"] = float(self.clahe_clip.text() or 0.05)
            k_val = int(self.clahe_kernel.text() or 20)
            params["clahe_kernel"] = (k_val, k_val)
        elif filter_type == "subtract background":
            params["sigma"] = float(self.bkg_sigma.text() or 3.0)
            params["radius"] = int(self.bkg_radius.text() or 15)
        elif filter_type == "remove hotpixels":
            params["threshold"] = float(self.hot_threshold.text() or 10.0)
            params["npass"] = int(self.hot_npass.text() or 3)
            params["filter_size"] = int(self.hot_filter_size.text() or 5)
        elif filter_type == "intensity rescale":
            params["p1"] = float(self.rescale_p1.text() or 2.0)
            params["p2"] = float(self.rescale_p2.text() or 98.0)
            
        self.runRequested.emit(params)

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._run_btn.setEnabled(enabled)

class MergeTab(QWidget):
    """Averaging two channels."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)
        
        # Channel 1 Selector
        self._channel1_combo = QComboBox()
        self._channel1_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel1_combo.setMinimumWidth(50)
        form.addRow("Channel 1:", self._channel1_combo)

        # Channel 2 Selector
        self._channel2_combo = QComboBox()
        self._channel2_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel2_combo.setMinimumWidth(50)
        form.addRow("Channel 2:", self._channel2_combo)

        layout.addLayout(form)
        
        self._run_btn = QPushButton("Run Merge")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)
        layout.addStretch()

        self._refresh_channels()
        self._channel_model.modelReset.connect(self._refresh_channels)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_channels())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_channels())

    def _refresh_channels(self):
        c1_current = self._channel1_combo.currentText()
        c2_current = self._channel2_combo.currentText()
        self._channel1_combo.clear()
        self._channel2_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if not ch.is_mask and not getattr(ch, "is_region", False):
                self._channel1_combo.addItem(ch.name, i)
                self._channel2_combo.addItem(ch.name, i)
        
        idx1 = self._channel1_combo.findText(c1_current)
        if idx1 >= 0: self._channel1_combo.setCurrentIndex(idx1)
        
        idx2 = self._channel2_combo.findText(c2_current)
        if idx2 >= 0: self._channel2_combo.setCurrentIndex(idx2)

    def _on_run(self):
        if self._channel1_combo.currentIndex() < 0: return
        if self._channel2_combo.currentIndex() < 0: return
        self.runRequested.emit({
            "is_merge": True,
            "channel1_index": self._channel1_combo.currentData(),
            "channel2_index": self._channel2_combo.currentData(),
        })

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._run_btn.setEnabled(enabled)

_BRIGHTFIELD_PRESETS = [
    "H&E", "IHC", "Aperio CS2", "Hamamatsu XR", "Hamamatsu S360",
    "Leica GT450", "3DHistech Pannoramic Scan II", "CyteFinder",
    "Axioscan 7", "Orion", "Masson Trichrome", "PAS",
    "Jones Silver", "Toluidine Blue",
]


class BrightfieldTab(QWidget):
    """Generate a virtual brightfield image from the multiplex image."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._preset_combo = QComboBox()
        self._preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for name in _BRIGHTFIELD_PRESETS:
            self._preset_combo.addItem(name)
        form.addRow("Preset:", self._preset_combo)
        layout.addLayout(form)

        self._run_btn = QPushButton("Generate Brightfield")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)
        layout.addStretch()

    def _on_run(self):
        self.runRequested.emit({"preset": self._preset_combo.currentText()})

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._run_btn.setEnabled(enabled)


class StarDistTab(QWidget):
    """Sub-widget for StarDist segmentation parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # Model Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel_combo.setMinimumWidth(50)
        form.addRow("Channel:", self._channel_combo)

        self._model_combo = QComboBox()
        self._model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._model_combo.setMinimumWidth(50)
        self._model_combo.addItems(["2D_versatile_fluo", "2D_paper_dsb2018"])
        self._scan_models()
        self._model_combo.setToolTip("Pre-trained StarDist models for different imaging modalities (e.g., fluorescence nuclei).")
        form.addRow("Model:", self._model_combo)

        self._prob_thresh = QLineEdit()
        self._prob_thresh.setPlaceholderText("Auto")
        self._prob_thresh.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self._prob_thresh.setFixedWidth(60)
        self._prob_thresh.setToolTip("Probability threshold for object detection. Higher values lead to fewer, more confident detections. Range [0.01, 1.0]. Leave empty for model defaults.")
        form.addRow("prob_thresh:", self._prob_thresh)

        self._nms_thresh = QLineEdit("0.3")
        self._nms_thresh.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self._nms_thresh.setFixedWidth(60)
        self._nms_thresh.setToolTip("Non-Maximum Suppression (NMS) threshold. Controls how much overlap is allowed between predicted shapes. Range [0.01, 1.0]. Higher values allow more overlap.")
        form.addRow("nms_thresh:", self._nms_thresh)
        
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
            if not ch.is_mask and not getattr(ch, "is_region", False):
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)
        

    def _scan_models(self):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "stardist")
        if not os.path.exists(models_dir): return
        for folder in os.listdir(models_dir):
            path = os.path.join(models_dir, folder)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                if self._model_combo.findText(folder) == -1:
                    self._model_combo.addItem(folder, folder)

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        
        prob_text = self._prob_thresh.text()
        use_default = not prob_text
        prob_val = float(prob_text) if prob_text else 0.5

        self.runRequested.emit({
            "method": "stardist",
            "channel_indices": [self._channel_combo.currentData()],
            "model_name": self._model_combo.currentText(),
            "model_folder": self._model_combo.currentData() or self._model_combo.currentText(),
            "use_default_thresh": use_default,
            "prob_thresh": prob_val,
            "nms_thresh": float(self._nms_thresh.text() or 0.3),
        })

class CellposeTab(QWidget):
    """Sub-widget for Cellpose segmentation parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Model Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel_combo.setMinimumWidth(50)
        form.addRow("Channel:", self._channel_combo)

        self._model_combo = QComboBox()
        self._model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._model_combo.setMinimumWidth(50)
        self._model_combo.addItems(["nuclei", "cyto", "cyto2"])
        self._scan_models()
        self._model_combo.setToolTip("Pre-trained Cellpose models. 'nuclei' for nuclear staining, 'cyto' or 'cyto2' for cytoplasmic/cellular staining.")
        form.addRow("Model:", self._model_combo)

        self._diameter = QLineEdit("6.5")
        self._diameter.setPlaceholderText("Auto")
        self._diameter.setValidator(QDoubleValidator(0.0, 1000.0, 2))
        self._diameter.setFixedWidth(60)
        self._diameter.setToolTip("Expected cell diameter in pixels. Set to 0 or 'Auto' for automated estimation based on image content.")
        form.addRow("Diameter:", self._diameter)

        self._cellprob_threshold = QLineEdit("0.0")
        self._cellprob_threshold.setValidator(QDoubleValidator(-6.0, 6.0, 2))
        self._cellprob_threshold.setFixedWidth(60)
        self._cellprob_threshold.setToolTip("Threshold for cell probability map. Higher values (stricter) reduce the number and size of detected cells. Range [-6.0, 6.0]. Default 0.0.")
        form.addRow("Cell Prob Threshold:", self._cellprob_threshold)

        self._flow_threshold = QLineEdit("0.4")
        self._flow_threshold.setValidator(QDoubleValidator(0.0, 10.0, 2))
        self._flow_threshold.setFixedWidth(60)
        self._flow_threshold.setToolTip("Maximum allowed error for the flow field. Higher values allow more masks to be created but may include low-quality results. Range [0.0, 10.0]. Default 0.4.")
        form.addRow("Flow Threshold:", self._flow_threshold)
        
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
            if not ch.is_mask and not getattr(ch, "is_region", False):
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)

    def _scan_models(self):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "cellpose")
        if not os.path.exists(models_dir): return
        for folder in os.listdir(models_dir):
            folder_path = os.path.join(models_dir, folder)
            if os.path.isdir(folder_path):
                # Find the model file inside the folder (largest non-metadata file)
                files = [f for f in os.listdir(folder_path) if not f.endswith(('.json', '.txt', '.ipynb', '.png', '.jpg', '.jpeg'))]
                if files:
                    model_file_name = max(files, key=lambda f: os.path.getsize(os.path.join(folder_path, f)))
                    model_file = os.path.join(folder_path, model_file_name)
                    if self._model_combo.findText(folder) == -1:
                        self._model_combo.addItem(folder, model_file)

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        diam_text = self._diameter.text()
        if diam_text.lower() == "auto" or not diam_text:
            diam = None
        else:
            try:
                diam = float(diam_text)
            except ValueError:
                diam = None
        
        cellprob_threshold = float(self._cellprob_threshold.text() or 0.0)
        flow_threshold = float(self._flow_threshold.text() or 0.4)
        
        self.runRequested.emit({
            "method": "cellpose",
            "channel_indices": [self._channel_combo.currentData()],
            "model_name": self._model_combo.currentText(),
            "model_path": self._model_combo.currentData(),
            "diameter": diam,
            "cellprob_threshold": cellprob_threshold,
            "flow_threshold": flow_threshold
        })
        
class OmniposeTab(QWidget):
    """Sub-widget for Omnipose segmentation parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Model Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel_combo.setMinimumWidth(50)
        form.addRow("Channel:", self._channel_combo)

        self._model_combo = QComboBox()
        self._model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._model_combo.setMinimumWidth(50)
        self._model_combo.addItems(["bact_phase_omni", "bact_fluor_omni", "worm_omni", "worm_bact_omni", "worm_high_res_omni", "cyto2_omni", "plant_omni"])
        self._scan_models()
        self._model_combo.setToolTip("Specialized Omnipose models for various cell shapes (bacteria, worms, plant cells, etc.).")
        form.addRow("Model:", self._model_combo)

        self._diameter = QLineEdit("Auto")
        self._diameter.setPlaceholderText("Auto")
        self._diameter.setValidator(QDoubleValidator(0.0, 1000.0, 2))
        self._diameter.setFixedWidth(60)
        self._diameter.setToolTip("Expected cell diameter in pixels. For Omnipose, 0 or 'Auto' uses internal scale estimation.")
        form.addRow("Diameter:", self._diameter)

        self._mask_threshold = QLineEdit("0.0")
        self._mask_threshold.setPlaceholderText("Default 0.0")
        self._mask_threshold.setValidator(QDoubleValidator(-10.0, 10.0, 2))
        self._mask_threshold.setFixedWidth(60)
        self._mask_threshold.setToolTip("Threshold for mask probability map. Higher values reduce the number and size of detected masks. Range [-10.0, 10.0]. Default 0.0.")
        form.addRow("Mask Threshold:", self._mask_threshold)

        self._flow_threshold = QLineEdit("0.4")
        self._flow_threshold.setValidator(QDoubleValidator(0.0, 10.0, 2))
        self._flow_threshold.setFixedWidth(60)
        self._flow_threshold.setToolTip("Maximum allowed error for the flow field. Higher values are less strict and may create more masks. Range [0.0, 10.0]. Default 0.4.")
        form.addRow("Flow Threshold:", self._flow_threshold)
        
        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Run Omnipose")
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
            if not ch.is_mask and not getattr(ch, "is_region", False):
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)

    def _scan_models(self):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "omnipose")
        if not os.path.exists(models_dir): return
        for file in os.listdir(models_dir):
            path = os.path.join(models_dir, file)
            if os.path.isfile(path) and not file.endswith('.txt') and not file.endswith('.ipynb'):
                self._model_combo.addItem(f"Custom: {file[:30]}...", path)

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        diam_text = self._diameter.text()
        if diam_text.lower() == "auto" or not diam_text:
            diam = 0.0
        else:
            try:
                diam = float(diam_text)
            except ValueError:
                diam = 0.0
        
        mask_threshold = float(self._mask_threshold.text() or 0.0)
        flow_threshold = float(self._flow_threshold.text() or 0.4)
        
        self.runRequested.emit({
            "method": "omnipose",
            "channel_indices": [self._channel_combo.currentData()],
            "model_name": self._model_combo.currentText(),
            "model_path": self._model_combo.currentData(),
            "diameter": diam,
            "mask_threshold": mask_threshold,
            "flow_threshold": flow_threshold
        })
        
class InstanSegTab(QWidget):
    """Sub-widget for InstanSeg segmentation parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Model Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel_combo.setMinimumWidth(50)
        form.addRow("Channel:", self._channel_combo)

        # Model Selector
        self._model_combo = QComboBox()
        self._model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._model_combo.setMinimumWidth(50)
        self._model_combo.setEditable(True)
        self._model_combo.addItems([
            "single_channel_nuclei",
            "fluorescence_nuclei_and_cells",
        ])
        self._scan_models()
        self._model_combo.setToolTip("InstanSeg models for single-channel nuclei or multi-channel fluorescence (nuclei and cells).")
        form.addRow("Model:", self._model_combo)

        # Pixel Size
        self._pixel_size = QLineEdit("1.0")
        self._pixel_size.setValidator(QDoubleValidator(0.001, 1000.0, 3))
        self._pixel_size.setFixedWidth(60)
        self._pixel_size.setToolTip("The resolution of the image in micrometers per pixel (\u03bcm/px). Used to scale the model's receptive field to the physical size of cells.")
        form.addRow("Pixel Size (\u03bcm):", self._pixel_size)

        self._fill_holes = QCheckBox("Fill holes")
        self._fill_holes.setChecked(False)
        self._fill_holes.setToolTip("Fill internal holes in detected cell masks.")
        form.addRow(self._fill_holes)

        self._keep_largest = QCheckBox("Keep largest component")
        self._keep_largest.setChecked(False)
        self._keep_largest.setToolTip("If a cell ID is fragmented, only keep the largest connected component.")
        form.addRow(self._keep_largest)
        
        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Run InstanSeg")
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
            if not ch.is_mask and not getattr(ch, "is_region", False):
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)

    def _scan_models(self):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "instanseg")
        if not os.path.exists(models_dir): return
        for folder in os.listdir(models_dir):
            folder_path = os.path.join(models_dir, folder)
            if os.path.isdir(folder_path):
                if os.path.exists(os.path.join(folder_path, "model_weights.pth")):
                    if self._model_combo.findText(folder) == -1:
                        self._model_combo.addItem(folder, folder_path)

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        self.runRequested.emit({
            "method": "instanseg",
            "channel_indices": [self._channel_combo.currentData()],
            "model_name": self._model_combo.currentText(),
            "model_path": self._model_combo.currentData(),
            "pixel_size": float(self._pixel_size.text() or 1.0),
            "fill_holes": self._fill_holes.isChecked(),
            "keep_largest": self._keep_largest.isChecked(),
        })

class MesmerTab(QWidget):
    """Sub-widget for Mesmer (DeepCell) segmentation parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)


        # ── Model / channel parameters ──────────────────────────────────────
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        # Nuclear Channel
        self._nuclear_combo = QComboBox()
        self._nuclear_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._nuclear_combo.setMinimumWidth(50)
        form.addRow("Nuclear Channel:", self._nuclear_combo)

        # Membrane Channel (hidden for local models)
        self._membrane_label = QLabel("Membrane Channel:")
        self._membrane_combo = QComboBox()
        self._membrane_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._membrane_combo.setMinimumWidth(50)
        form.addRow(self._membrane_label, self._membrane_combo)

        # Model Selector
        self._model_combo = QComboBox()
        self._model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._model_combo.setMinimumWidth(50)
        self._model_combo.addItem("Default (DeepCell)", None)
        self._model_combo.setToolTip("Select a local Mesmer model folder or use the default DeepCell cloud model.")
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._scan_models()
        form.addRow("Model:", self._model_combo)

        # Compartment
        self._compartment_combo = QComboBox()
        self._compartment_combo.addItems(["nuclear", "whole-cell"])
        self._compartment_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._compartment_combo.setToolTip("Segment either only the nuclei or the whole-cell area (including cytoplasm).")
        form.addRow("Compartment:", self._compartment_combo)

        # API Key
        self._api_key = QLineEdit("0eJIjCpR.fYenvVnMb4ZAKCjxYnjZ1R2V6kvLdq5V")
        self._api_key.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        form.addRow("API Key:", self._api_key)

        # Pixel Size
        self._pixel_size = QLineEdit("1.0")
        self._pixel_size.setValidator(QDoubleValidator(0.001, 1000.0, 3))
        self._pixel_size.setFixedWidth(60)
        self._pixel_size.setToolTip("The resolution of the image in micrometers per pixel (\u03bcm/px). Crucial for accurate deep learning inference on varied datasets.")
        form.addRow("Pixel Size (\u03bcm):", self._pixel_size)

        # ── Watershed postprocessing parameters ─────────────────────────────
        self._ws_radius = QLineEdit("3")
        self._ws_radius.setValidator(QIntValidator(1, 100))
        self._ws_radius.setFixedWidth(70)
        self._ws_radius.setToolTip("Radius for local maxima detection. Smaller values detect finer peaks; larger values merge nearby nuclei.")
        form.addRow("Radius:", self._ws_radius)

        self._ws_maxima_thresh = QLineEdit("0.0004")
        self._ws_maxima_thresh.setValidator(QDoubleValidator(0.0, 1.0, 6))
        self._ws_maxima_thresh.setFixedWidth(70)
        self._ws_maxima_thresh.setToolTip("Minimum height of a peak in the inner-distance map to be counted as a seed. Lower = more seeds (may over-segment).")
        form.addRow("Maxima threshold:", self._ws_maxima_thresh)

        self._ws_maxima_smooth = QLineEdit("0")
        self._ws_maxima_smooth.setValidator(QDoubleValidator(0.0, 10.0, 2))
        self._ws_maxima_smooth.setFixedWidth(70)
        self._ws_maxima_smooth.setToolTip("Gaussian smoothing sigma applied to the inner-distance map before peak detection. 0 = no smoothing.")
        form.addRow("Maxima smooth:", self._ws_maxima_smooth)

        self._ws_interior_thresh = QLineEdit("0.1")
        self._ws_interior_thresh.setValidator(QDoubleValidator(0.0, 1.0, 4))
        self._ws_interior_thresh.setFixedWidth(70)
        self._ws_interior_thresh.setToolTip("Pixels with outer-distance below this threshold are treated as background. Higher = smaller/fewer cells.")
        form.addRow("Interior threshold:", self._ws_interior_thresh)

        self._ws_interior_smooth = QLineEdit("2")
        self._ws_interior_smooth.setValidator(QDoubleValidator(0.0, 10.0, 2))
        self._ws_interior_smooth.setFixedWidth(70)
        self._ws_interior_smooth.setToolTip("Gaussian smoothing sigma applied to the outer-distance map before thresholding. 0 = no smoothing.")
        form.addRow("Interior smooth:", self._ws_interior_smooth)

        self._ws_small_objects = QLineEdit("0")
        self._ws_small_objects.setValidator(QIntValidator(0, 100000))
        self._ws_small_objects.setFixedWidth(70)
        self._ws_small_objects.setToolTip("Objects smaller than this area (px\u00b2) are removed after watershed.")
        form.addRow("Min object size:", self._ws_small_objects)

        self._ws_fill_holes = QLineEdit("15")
        self._ws_fill_holes.setValidator(QIntValidator(0, 100000))
        self._ws_fill_holes.setFixedWidth(70)
        self._ws_fill_holes.setToolTip("Holes smaller than this area (px\u00b2) inside segmented regions are filled.")
        form.addRow("Fill holes (px\u00b2):", self._ws_fill_holes)

        self._ws_exclude_border = QComboBox()
        self._ws_exclude_border.addItems(["False", "True"])
        self._ws_exclude_border.setFixedWidth(70)
        self._ws_exclude_border.setToolTip("If True, objects touching the image border are removed.")
        form.addRow("Exclude border:", self._ws_exclude_border)

        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Run Mesmer")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)


        self._refresh_channels()
        self._channel_model.modelReset.connect(self._refresh_channels)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_channels())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_channels())
        # Apply initial visibility
        self._on_model_changed(self._model_combo.currentIndex())


    def _scan_models(self):
        """Scan models/mesmer/ for subfolders containing a .keras model file."""
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "mesmer")
        if not os.path.exists(models_dir):
            return
        for folder in os.listdir(models_dir):
            folder_path = os.path.join(models_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            # Find the .keras model file in the folder
            keras_files = [f for f in os.listdir(folder_path) if f.endswith(".keras")]
            if keras_files:
                model_file = os.path.join(folder_path, keras_files[0])
                if self._model_combo.findText(folder) == -1:
                    self._model_combo.addItem(folder, model_file)

    def _on_model_changed(self, index):
        """Show/hide the membrane channel selector and update defaults."""
        model_path = self._model_combo.currentData()
        is_local = model_path is not None
        self._membrane_label.setVisible(not is_local)
        self._membrane_combo.setVisible(not is_local)
        
        # Reset to appropriate defaults based on model type
        if is_local:
            self._ws_radius.setText("2")
            self._ws_maxima_thresh.setText("0.1")
            self._ws_maxima_smooth.setText("0")
            self._ws_interior_thresh.setText("0.15")
            self._ws_interior_smooth.setText("1")
            self._ws_small_objects.setText("15")
            self._ws_fill_holes.setText("15")
            self._ws_exclude_border.setCurrentText("False")
        else:
            self._ws_radius.setText("2")
            self._ws_maxima_thresh.setText("0.1")
            self._ws_maxima_smooth.setText("0")
            self._ws_interior_thresh.setText("0.2")
            self._ws_interior_smooth.setText("2")
            self._ws_small_objects.setText("15")
            self._ws_fill_holes.setText("15")
            self._ws_exclude_border.setCurrentText("False")

    def _refresh_channels(self):
        n_current = self._nuclear_combo.currentText()
        m_current = self._membrane_combo.currentText()
        self._nuclear_combo.clear()
        self._membrane_combo.clear()
        self._membrane_combo.addItem("None", -1)
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if not ch.is_mask and not getattr(ch, "is_region", False):
                self._nuclear_combo.addItem(ch.name, i)
                self._membrane_combo.addItem(ch.name, i)
        
        ni = self._nuclear_combo.findText(n_current)
        if ni >= 0: self._nuclear_combo.setCurrentIndex(ni)
        mi = self._membrane_combo.findText(m_current)
        if mi >= 0: self._membrane_combo.setCurrentIndex(mi)

    def _on_run(self):
        if self._nuclear_combo.currentIndex() < 0: return
        
        model_path = self._model_combo.currentData()
        # For local models the membrane is always zero-filled (nucleus-only)
        indices = [self._nuclear_combo.currentData()]
        if model_path is None and self._membrane_combo.currentData() != -1:
            indices.append(self._membrane_combo.currentData())

        params = {
            "method": "mesmer",
            "channel_indices": indices,
            "model_path": model_path,
            "api_key": self._api_key.text(),
            "pixel_size": float(self._pixel_size.text() or 1.0),
            "compartment": self._compartment_combo.currentText(),
            "watershed_kwargs": {
                "radius": int(self._ws_radius.text() or 3),
                "maxima_threshold": float(self._ws_maxima_thresh.text() or 0.0004),
                "maxima_smooth": float(self._ws_maxima_smooth.text() or 0),
                "interior_threshold": float(self._ws_interior_thresh.text() or 0.1),
                "interior_smooth": float(self._ws_interior_smooth.text() or 2),
                "small_objects_threshold": int(self._ws_small_objects.text() or 0),
                "fill_holes_threshold": int(self._ws_fill_holes.text() or 15),
                "exclude_border": self._ws_exclude_border.currentText() == "True",
            },
        }

        self.runRequested.emit(params)

class WatershedTab(QWidget):
    """Sub-widget for Watershed segmentation parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # Model Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel_combo.setMinimumWidth(50)
        form.addRow("Channel:", self._channel_combo)

        # Labeller mode
        self._labeller_combo = QComboBox()
        self._labeller_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._labeller_combo.setMinimumWidth(50)
        self._labeller_combo.addItems(["voronoi", "gauss"])
        self._labeller_combo.setToolTip("The core logic for seeding: 'voronoi' uses local maxima; 'gauss' uses global Otsu thresholding.")
        self._labeller_combo.currentTextChanged.connect(self._on_labeller_changed)
        form.addRow("Labeller:", self._labeller_combo)

        # Spot sigma (only for voronoi)
        self._spot_sigma = QLineEdit("2")
        self._spot_sigma.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self._spot_sigma.setFixedWidth(80)
        self._spot_sigma.setToolTip("Smoothing factor for seed detection. Higher values merge nearby seeds, reducing over-segmentation. Range [0.1, 100.0].")
        self._spot_sigma_label = QLabel("Spot Sigma:")
        form.addRow(self._spot_sigma_label, self._spot_sigma)

        # Outline sigma
        self._outline_sigma = QLineEdit("2")
        self._outline_sigma.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self._outline_sigma.setFixedWidth(80)
        self._outline_sigma.setToolTip("Smoothing factor for boundary detection. Higher values create smoother, more generalized cell shapes. Range [0.1, 100.0].")
        form.addRow("Outline:", self._outline_sigma)

        # Threshold
        self._threshold = QLineEdit("1")
        self._threshold.setValidator(QDoubleValidator(0.01, 100.0, 2))
        self._threshold.setFixedWidth(80)
        self._threshold.setToolTip(
            "Sensitivity of detection. Range [0.01, 100.0].\n"
            "Voronoi: offset from local threshold (>1 = stricter, <1 = more permissive).\n"
            "Gauss: multiplier on Otsu threshold (>1 = stricter, <1 = more permissive)."
        )
        form.addRow("Threshold:", self._threshold)

        self._min_mean_intensity = QLineEdit("0.2")
        self._min_mean_intensity.setValidator(QDoubleValidator(-1000000.0, 1000000.0, 4))
        self._min_mean_intensity.setFixedWidth(80)
        self._min_mean_intensity.setToolTip("Filters out detected objects whose average pixel intensity is below this value. Useful for removing background noise or dim artifacts.")
        form.addRow("Intensity Threshold:", self._min_mean_intensity)

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
            if not ch.is_mask and not getattr(ch, "is_region", False):
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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        self._mask_combo = QComboBox()
        self._mask_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._mask_combo.setMinimumWidth(50)
        form.addRow("Mask:", self._mask_combo)
        
        self._min_size = QLineEdit("10")
        self._min_size.setValidator(QIntValidator(0, 1000000))
        self._min_size.setFixedWidth(60)
        form.addRow("Min size:", self._min_size)

        self._max_size = QLineEdit("1000")
        self._max_size.setValidator(QIntValidator(0, 1000000))
        self._max_size.setFixedWidth(60)
        form.addRow("Max size:", self._max_size)

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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        self._mask_combo = QComboBox()
        self._mask_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._mask_combo.setMinimumWidth(50)
        form.addRow("Mask:", self._mask_combo)

        self._pixels = QLineEdit("6")
        self._pixels.setValidator(QIntValidator(1, 100))
        self._pixels.setFixedWidth(60)
        form.addRow("Expand:", self._pixels)

        # Method selector
        self._method_combo = QComboBox()
        self._method_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._method_combo.setMinimumWidth(50)
        self._method_combo.addItem("Binary Mask", "binary_mask")
        self._method_combo.addItem("Label Map", "label_map")
        self._method_combo.setToolTip(
            "Binary Mask: expands each cell using watershed, creating separation lines between touching cells.\n"
            "Label Map: expands each cell with its integer label value, no watershed lines."
        )
        form.addRow("Method:", self._method_combo)

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
        method = self._method_combo.currentData()
        tool = "expansion_watershed" if method == "binary_mask" else "expansion_labelmap"
        self.runRequested.emit({
            "mask_index": self._mask_combo.currentData(),
            "expansion_pixels": int(self._pixels.text() or 6),
            "tool": tool
        })

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._run_btn.setEnabled(enabled)

class CellSamplerTab(QWidget):
    """Sub-widget for sampling and merging multiple masks."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # Parameters
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        # Mask Selector (List Widget for multiple selections)
        self._mask_list = QListWidget()
        self._mask_list.setFixedHeight(80)
        self._mask_list.setMinimumWidth(0)
        form.addRow("Masks:", self._mask_list)
        
        # Strategy Selector
        self._strategy_combo = QComboBox()
        self._strategy_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._strategy_combo.setMinimumWidth(50)
        self._strategy_combo.addItem("Largest cell count", "pop")
        self._strategy_combo.addItem("Highest Jaccard", "j1")
        self._strategy_combo.addItem("Min area variance", "cstd")
        form.addRow("Strategy:", self._strategy_combo)

        # Window Size Parameter
        self._nsize_edit = QLineEdit("40")
        self._nsize_edit.setValidator(QIntValidator(1, 10000))
        self._nsize_edit.setFixedWidth(80)
        self._nsize_edit.setToolTip("Local window size (in pixels) for comparing masks. Default is 40.")
        form.addRow("Window Size (px):", self._nsize_edit)

        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Run CellSampler")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)
        
        self._refresh_masks()
        self._channel_model.modelReset.connect(self._refresh_masks)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_masks())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_masks())

    def _refresh_masks(self):
        # Save selection
        checked_texts = []
        for i in range(self._mask_list.count()):
            item = self._mask_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked_texts.append(item.text())
        self._mask_list.clear()
        
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_mask:
                item = QListWidgetItem(ch.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if ch.name in checked_texts else Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, i)
                self._mask_list.addItem(item)

    def _on_run(self):
        selected_indices = []
        for i in range(self._mask_list.count()):
            item = self._mask_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_indices.append(item.data(Qt.ItemDataRole.UserRole))
                
        if len(selected_indices) < 2:
            return  # Need at least two to merge
            
        self.runRequested.emit({
            "mask_indices": selected_indices,
            "merit": self._strategy_combo.currentData(),
            "nsize": int(self._nsize_edit.text() or 40),
            "tool": "cell_sampler"
        })

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._run_btn.setEnabled(enabled)


class ThresholdPositivityTab(QWidget):
    """
    Threshold-based cell positivity tab.

    Workflow
    --------
    1. User picks a cell-mask and clicks "Get Thresholds".
       → Background thread computes per-cell mean intensity for every
         image channel using scipy.ndimage.mean (one vectorised call per
         channel, fast for 10 k cells).
    2. For each channel the Otsu threshold over the per-cell means is
       calculated and stored.  A channel dropdown + scrollbar let the
       user inspect and adjust any channel's threshold interactively.
    3. On every scrollbar move the positivity map is rebuilt with a
       single numpy LUT lookup (O(H×W), no Python cell loop) and the
       existing channel's mask_data is updated in-place, triggering a
       canvas repaint without creating a new channel object.
    """

    runThresholdComputeRequested = Signal(dict)   # {mask_index}
    applyThresholdRequested      = Signal(dict)   # {ch_index, threshold, mask_index, ch_name, ch_color}

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model

        # State shared with MainWindow after compute
        # keyed by image-channel model index → np.ndarray of per-cell means
        # index 0 = background (always 0)
        self._cell_means: dict[int, np.ndarray] = {}
        self._labels: np.ndarray | None = None          # the label map used for compute
        self._mask_model_index: int = -1                # which mask channel was used
        self._otsu_thresholds: dict[int, float] = {}    # ch_model_idx → otsu threshold
        self._generated_ch_indices: dict[int, int] = {} # ch_model_idx → result channel model idx

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        # Mask picker
        self._mask_combo = QComboBox()
        self._mask_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._mask_combo.setMinimumWidth(50)
        form.addRow("Mask:", self._mask_combo)
        layout.addLayout(form)

        self._get_btn = QPushButton("Get Thresholds")
        self._get_btn.clicked.connect(self._on_get)
        layout.addWidget(self._get_btn)

        # ---- interactive threshold controls (hidden until compute done) ----
        self._controls = QWidget()
        ctrl_lay = QVBoxLayout(self._controls)
        ctrl_lay.setContentsMargins(0, 6, 0, 0)
        ctrl_lay.setSpacing(6)

        ctrl_form = QFormLayout()
        ctrl_form.setContentsMargins(0, 0, 0, 0)
        ctrl_form.setSpacing(6)
        ctrl_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        ctrl_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        ctrl_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        ctrl_form.setHorizontalSpacing(4)

        self._channel_combo = QComboBox()
        self._channel_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        ctrl_form.addRow("Channel:", self._channel_combo)
        ctrl_lay.addLayout(ctrl_form)

        # Threshold value input
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold:"))
        self._thresh_input = QLineEdit()
        self._thresh_input.setFixedWidth(80)
        self._thresh_input.setValidator(QDoubleValidator())
        self._thresh_input.editingFinished.connect(self._on_input_changed)
        thresh_row.addWidget(self._thresh_input)
        thresh_row.addStretch()
        self._pos_count_label = QLabel("")
        thresh_row.addWidget(self._pos_count_label)
        ctrl_lay.addLayout(thresh_row)

        # Scrollbar for threshold
        from PySide6.QtWidgets import QScrollBar
        self._slider = QScrollBar(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(1000)   # mapped to [min_mean, max_mean]
        self._slider.setPageStep(10)
        self._slider.valueChanged.connect(self._on_slider_changed)
        ctrl_lay.addWidget(self._slider)

        layout.addWidget(self._controls)
        self._controls.setVisible(False)
        layout.addStretch()

        self._refresh_masks()
        self._channel_model.modelReset.connect(self._refresh_masks)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_masks())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_masks())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_masks(self):
        current = self._mask_combo.currentText()
        self._mask_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_mask:
                self._mask_combo.addItem(ch.name, i)
        idx = self._mask_combo.findText(current)
        if idx >= 0:
            self._mask_combo.setCurrentIndex(idx)

    def _on_get(self):
        if self._mask_combo.currentIndex() < 0:
            return
        self._get_btn.setEnabled(False)
        self._controls.setVisible(False)
        mask_idx = self._mask_combo.currentData()
        self.runThresholdComputeRequested.emit({"mask_index": mask_idx})

    def receive_means(self, labels, cell_means: dict, otsu_thresholds: dict, mask_model_index: int):
        """
        Called from MainWindow (main thread via signal) when per-cell
        means are ready.

        Parameters
        ----------
        labels            : (H,W) int32 label map
        cell_means        : {ch_model_idx: np.ndarray[max_label+1]}  (index 0 = bg)
        otsu_thresholds   : {ch_model_idx: float}
        mask_model_index  : which mask channel was used
        """
        self._labels = labels
        self._cell_means = cell_means
        self._otsu_thresholds = otsu_thresholds
        self._mask_model_index = mask_model_index
        self._generated_ch_indices = {}

        # Populate channel dropdown with image channels that have means
        prev = self._channel_combo.currentData()
        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for ch_idx in sorted(cell_means.keys()):
            ch = self._channel_model.channel(ch_idx)
            self._channel_combo.addItem(ch.name, ch_idx)
        self._channel_combo.blockSignals(False)

        # Restore previous selection or default to first
        restore = self._channel_combo.findData(prev)
        self._channel_combo.setCurrentIndex(restore if restore >= 0 else 0)

        # Apply Otsu thresholds for ALL channels immediately
        for ch_idx, otsu in self._otsu_thresholds.items():
            ch = self._channel_model.channel(ch_idx)
            self.applyThresholdRequested.emit({
                "ch_model_index":   ch_idx,
                "threshold":        otsu,
                "mask_index":       self._mask_model_index,
                "ch_name":          ch.name,
                "ch_color":         ch.color,
                "target_ch_index":  -1,
            })

        self._controls.setVisible(True)
        self._get_btn.setEnabled(True)
        self._on_channel_changed()  # Sync slider to Otsu threshold

    def _on_channel_changed(self):
        """Update slider range and position to the Otsu threshold for the selected channel."""
        ch_idx = self._channel_combo.currentData()
        if ch_idx is None or ch_idx not in self._cell_means:
            return
        means = self._cell_means[ch_idx]
        # means[0] = background; ignore it
        valid = means[1:]
        if valid.size == 0:
            return
        self._means_min = float(np.min(valid))
        self._means_max = float(np.max(valid))
        if self._means_max <= self._means_min:
            self._means_max = self._means_min + 1.0

        # Set slider position to Otsu threshold (safely)
        otsu = self._otsu_thresholds.get(ch_idx, (self._means_min + self._means_max) / 2)
        rng = self._means_max - self._means_min
        if rng > 0 and not np.isnan(otsu):
            slider_val = int((otsu - self._means_min) / rng * 1000)
            self._slider.blockSignals(True)
            self._slider.setValue(max(0, min(1000, slider_val)))
            self._slider.blockSignals(False)
        else:
            self._slider.setValue(0)

        self._update_threshold_display()
        # Apply immediately to show the Otsu result for this channel
        self._emit_apply()

    def _on_slider_changed(self):
        self._update_threshold_display()
        self._emit_apply()

    def _on_input_changed(self):
        try:
            val = float(self._thresh_input.text())
            if hasattr(self, '_means_min') and hasattr(self, '_means_max') and self._means_max > self._means_min:
                # Update slider position
                frac = (val - self._means_min) / (self._means_max - self._means_min)
                self._slider.blockSignals(True)
                self._slider.setValue(max(0, min(1000, int(frac * 1000))))
                self._slider.blockSignals(False)
                # Update pos count display
                ch_idx = self._channel_combo.currentData()
                means = self._cell_means.get(ch_idx)
                if means is not None:
                    n_pos = int(np.sum(means[1:] >= val))
                    n_total = int(np.sum(means[1:] > 0))
                    self._pos_count_label.setText(f"{n_pos}/{n_total}")
                self._emit_apply()
        except ValueError:
            pass

    def _update_threshold_display(self):
        thresh = self._current_threshold()
        if thresh is None:
            return
        # Count positive cells
        ch_idx = self._channel_combo.currentData()
        means = self._cell_means.get(ch_idx)
        if means is not None:
            n_pos = int(np.sum(means[1:] >= thresh))
            n_total = int(np.sum(means[1:] > 0))  # cells with any signal
            self._pos_count_label.setText(f"{n_pos}/{n_total}")
        self._thresh_input.blockSignals(True)
        self._thresh_input.setText(f"{thresh:.4g}")
        self._thresh_input.blockSignals(False)

    def _current_threshold(self) -> float | None:
        if not hasattr(self, '_means_min'):
            return None
        frac = self._slider.value() / 1000.0
        return self._means_min + frac * (self._means_max - self._means_min)

    def _emit_apply(self):
        thresh = self._current_threshold()
        if thresh is None or self._labels is None:
            return
        ch_idx = self._channel_combo.currentData()
        if ch_idx is None:
            return
        ch = self._channel_model.channel(ch_idx)
        target_idx = self._generated_ch_indices.get(ch_idx, -1)
        self.applyThresholdRequested.emit({
            "ch_model_index":   ch_idx,
            "threshold":        thresh,
            "mask_index":       self._mask_model_index,
            "ch_name":          ch.name,
            "ch_color":         ch.color,
            "target_ch_index":  target_idx,
        })

    def register_generated_channel(self, ch_model_index: int, result_ch_index: int):
        """Called by MainWindow after a new result channel is added for the first time."""
        self._generated_ch_indices[ch_model_index] = result_ch_index

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._get_btn.setEnabled(enabled)


class ClusteringTab(QWidget):
    """Sub-widget for cell clustering parameters."""
    runRequested = Signal(dict)

    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # ── Channel selector ───────────────────────────────────────────────
        ch_header = QHBoxLayout()
        ch_header.setSpacing(4)
        ch_header.addWidget(QLabel("Channels:"))
        ch_header.addStretch()
        self._ch_all_btn = QPushButton("All")
        self._ch_all_btn.setFixedWidth(34)
        self._ch_none_btn = QPushButton("None")
        self._ch_none_btn.setFixedWidth(40)
        ch_header.addWidget(self._ch_all_btn)
        ch_header.addWidget(self._ch_none_btn)
        layout.addLayout(ch_header)

        self._channel_list = QListWidget()
        self._channel_list.setFixedHeight(120)
        self._channel_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._channel_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self._channel_list)

        self._ch_all_btn.clicked.connect(lambda: self._set_all_channels(True))
        self._ch_none_btn.clicked.connect(lambda: self._set_all_channels(False))

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(4)

        # Mask selector
        self._mask_combo = QComboBox()
        self._mask_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._mask_combo.setMinimumWidth(50)
        self._mask_combo.setToolTip("Select a mask where each cell has a unique integer label (or cells are separated).")
        form.addRow("Mask:", self._mask_combo)

        # Method selector
        self._method_combo = QComboBox()
        self._method_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._method_combo.setMinimumWidth(50)
        self._method_combo.addItems(["Leiden", "Louvain", "PhenoGraph", "FlowSOM", "KMeans", "Hierarchical", "DBSCAN"])
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method:", self._method_combo)

        # Normalization selector
        self._norm_combo = QComboBox()
        self._norm_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._norm_combo.setMinimumWidth(50)
        self._norm_combo.addItem("Yeo-Johnson", "yeo-johnson")
        self._norm_combo.addItem("Arcsinh", "arcsinh")
        self._norm_combo.addItem("Log-Z", "log-z")
        self._norm_combo.addItem("Z-score", "zscore")
        self._norm_combo.addItem("Min-Max", "minmax")
        self._norm_combo.addItem("None", "none")
        self._norm_combo.setToolTip(
            "Normalization applied to per-cell mean intensities before clustering.\n"
            "Yeo-Johnson: power transform + Z-score (handles skewed data well).\n"
            "Arcsinh: arcsinh(x / cofactor) + Z-score (common for mass cytometry).\n"
            "Log-Z: log1p + Z-score for skewed channels, Z-score for others.\n"
            "Z-score: zero mean, unit variance per channel.\n"
            "Min-Max: scale each channel to [0, 1].\n"
            "None: use raw mean intensities."
        )
        self._norm_combo.currentIndexChanged.connect(self._on_norm_changed)
        form.addRow("Normalize:", self._norm_combo)

        layout.addLayout(form)

        # ── PCA pre-processing ─────────────────────────────────────────────
        pca_row = QHBoxLayout()
        pca_row.setSpacing(6)
        self._pca_check = QCheckBox("PCA")
        self._pca_check.setToolTip(
            "Reduce dimensionality with PCA before clustering.\n"
            "Number of components chosen automatically via Horn's parallel\n"
            "analysis (shuffle correlation matrix). Always applied for DBSCAN."
        )
        self._pca_n_override = QLineEdit("")
        self._pca_n_override.setPlaceholderText("# components")
        self._pca_n_override.setFixedWidth(90)
        self._pca_n_override.setValidator(QIntValidator(1, 200))
        self._pca_n_override.setToolTip(
            "Override number of PCA components.\n"
            "Leave blank to use Horn's parallel analysis."
        )
        self._pca_n_override.setVisible(False)
        pca_row.addWidget(self._pca_check)
        pca_row.addWidget(self._pca_n_override)
        pca_row.addStretch()
        self._pca_check.toggled.connect(self._pca_n_override.setVisible)
        layout.addLayout(pca_row)

        # ── Arcsinh normalization parameter ────────────────────────────────
        self._arcsinh_container = QWidget()
        arc_lay = QFormLayout(self._arcsinh_container)
        arc_lay.setContentsMargins(0, 0, 0, 0)
        arc_lay.setSpacing(6)
        arc_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        arc_lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        arc_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._arcsinh_cofactor = QLineEdit("5")
        self._arcsinh_cofactor.setValidator(QDoubleValidator(0.0001, 10000.0, 4))
        self._arcsinh_cofactor.setFixedWidth(60)
        self._arcsinh_cofactor.setToolTip("Cofactor divisor for arcsinh transform. Common values: 5 (CyTOF), 150 (fluorescence).")
        arc_lay.addRow("Cofactor:", self._arcsinh_cofactor)
        layout.addWidget(self._arcsinh_container)

        # ── Log-Z normalization parameter ─────────────────────────────────
        self._logz_container = QWidget()
        logz_lay = QFormLayout(self._logz_container)
        logz_lay.setContentsMargins(0, 0, 0, 0)
        logz_lay.setSpacing(6)
        logz_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        logz_lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        logz_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._logz_skewness = QLineEdit("1")
        self._logz_skewness.setValidator(QDoubleValidator(0.0, 100.0, 4))
        self._logz_skewness.setFixedWidth(60)
        self._logz_skewness.setToolTip("Channels with abs(skewness) above this threshold get log1p before Z-score.")
        logz_lay.addRow("Skew threshold:", self._logz_skewness)
        layout.addWidget(self._logz_container)

        # ── Leiden parameters ──────────────────────────────────────────────
        self._leiden_container = QWidget()
        lei_lay = QFormLayout(self._leiden_container)
        lei_lay.setContentsMargins(0, 0, 0, 0)
        lei_lay.setSpacing(6)
        lei_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        lei_lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        lei_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._leiden_resolution = QLineEdit("0.5")
        self._leiden_resolution.setValidator(QDoubleValidator(0.0001, 100.0, 4))
        self._leiden_resolution.setFixedWidth(60)
        self._leiden_resolution.setToolTip("Resolution parameter for Leiden. Higher values produce more clusters.")
        lei_lay.addRow("Resolution:", self._leiden_resolution)
        layout.addWidget(self._leiden_container)

        # ── Louvain parameters ─────────────────────────────────────────────
        self._louvain_container = QWidget()
        lou_lay = QFormLayout(self._louvain_container)
        lou_lay.setContentsMargins(0, 0, 0, 0)
        lou_lay.setSpacing(6)
        lou_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        lou_lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        lou_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._louvain_resolution = QLineEdit("0.5")
        self._louvain_resolution.setValidator(QDoubleValidator(0.0001, 100.0, 4))
        self._louvain_resolution.setFixedWidth(60)
        self._louvain_resolution.setToolTip("Resolution parameter for Louvain. Higher values produce more clusters.")
        lou_lay.addRow("Resolution:", self._louvain_resolution)
        layout.addWidget(self._louvain_container)

        # ── PhenoGraph parameters ──────────────────────────────────────────
        self._phenograph_container = QWidget()
        pg_lay = QFormLayout(self._phenograph_container)
        pg_lay.setContentsMargins(0, 0, 0, 0)
        pg_lay.setSpacing(6)
        pg_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        pg_lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        pg_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._phenograph_k = QLineEdit("30")
        self._phenograph_k.setValidator(QIntValidator(5, 500))
        self._phenograph_k.setFixedWidth(60)
        self._phenograph_k.setToolTip("Number of nearest neighbours for the k-NN graph.")
        pg_lay.addRow("k neighbours:", self._phenograph_k)
        layout.addWidget(self._phenograph_container)

        # ── FlowSOM parameters ─────────────────────────────────────────────
        self._flowsom_container = QWidget()
        fs_lay = QFormLayout(self._flowsom_container)
        fs_lay.setContentsMargins(0, 0, 0, 0)
        fs_lay.setSpacing(6)
        fs_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        fs_lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        fs_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._flowsom_xdim = QLineEdit("10")
        self._flowsom_xdim.setValidator(QIntValidator(2, 50))
        self._flowsom_xdim.setFixedWidth(60)
        self._flowsom_xdim.setToolTip("SOM grid X dimension.")
        fs_lay.addRow("Grid X:", self._flowsom_xdim)

        self._flowsom_ydim = QLineEdit("10")
        self._flowsom_ydim.setValidator(QIntValidator(2, 50))
        self._flowsom_ydim.setFixedWidth(60)
        self._flowsom_ydim.setToolTip("SOM grid Y dimension.")
        fs_lay.addRow("Grid Y:", self._flowsom_ydim)

        self._flowsom_n_clusters = QLineEdit("10")
        self._flowsom_n_clusters.setValidator(QIntValidator(2, 200))
        self._flowsom_n_clusters.setFixedWidth(60)
        self._flowsom_n_clusters.setToolTip("Number of metaclusters to produce from the SOM nodes.")
        fs_lay.addRow("Metaclusters:", self._flowsom_n_clusters)
        layout.addWidget(self._flowsom_container)

        # ── Hierarchical parameters ────────────────────────────────────────
        self._hierarchical_container = QWidget()
        hc_lay = QFormLayout(self._hierarchical_container)
        hc_lay.setContentsMargins(0, 0, 0, 0)
        hc_lay.setSpacing(6)
        hc_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        hc_lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        hc_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._hc_n_clusters = QLineEdit("5")
        self._hc_n_clusters.setValidator(QIntValidator(2, 1000))
        self._hc_n_clusters.setFixedWidth(60)
        self._hc_n_clusters.setToolTip("Number of clusters to produce.")
        hc_lay.addRow("Clusters:", self._hc_n_clusters)

        self._hc_linkage = QComboBox()
        self._hc_linkage.addItems(["ward", "complete", "average", "single"])
        self._hc_linkage.setToolTip("Linkage criterion. 'ward' minimises variance and requires Euclidean metric.")
        self._hc_linkage.currentIndexChanged.connect(self._on_hc_linkage_changed)
        hc_lay.addRow("Linkage:", self._hc_linkage)

        self._hc_metric = QComboBox()
        self._hc_metric.addItems(["euclidean", "cosine", "manhattan"])
        self._hc_metric.setToolTip("Distance metric. Ignored when linkage is 'ward' (always Euclidean).")
        hc_lay.addRow("Metric:", self._hc_metric)
        layout.addWidget(self._hierarchical_container)

        # ── DBSCAN parameters ─────────────────────────────────────────────
        self._dbscan_container = QWidget()
        db_lay = QFormLayout(self._dbscan_container)
        db_lay.setContentsMargins(0, 0, 0, 0)
        db_lay.setSpacing(6)
        db_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        db_lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        db_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._dbscan_eps = QLineEdit("")
        self._dbscan_eps.setPlaceholderText("Auto")
        self._dbscan_eps.setValidator(QDoubleValidator(0.0001, 1000.0, 4))
        self._dbscan_eps.setFixedWidth(60)
        self._dbscan_eps.setToolTip(
            "Neighbourhood radius (Euclidean distance in normalised feature space).\n"
            "Leave blank to auto-estimate from the k-NN distance distribution\n"
            "(90th percentile of each cell's distance to its min_samples-th\n"
            "nearest neighbour). Override if too many cells become noise."
        )
        db_lay.addRow("Epsilon:", self._dbscan_eps)

        self._dbscan_min_samples = QLineEdit("10")
        self._dbscan_min_samples.setValidator(QIntValidator(1, 100000))
        self._dbscan_min_samples.setFixedWidth(60)
        self._dbscan_min_samples.setToolTip(
            "Minimum number of neighbours within epsilon for a point to be a core point.\n"
            "Lower values produce more (smaller) clusters.\n"
            "Typical range: 5–50."
        )
        db_lay.addRow("Min samples:", self._dbscan_min_samples)
        layout.addWidget(self._dbscan_container)

        # ── KMeans parameters ─────────────────────────────────────────────
        self._kmeans_container = QWidget()
        km_lay = QFormLayout(self._kmeans_container)
        km_lay.setContentsMargins(0, 0, 0, 0)
        km_lay.setSpacing(6)
        km_lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        km_lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        km_lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._kmeans_n_clusters = QLineEdit("5")
        self._kmeans_n_clusters.setValidator(QIntValidator(2, 1000))
        self._kmeans_n_clusters.setFixedWidth(60)
        self._kmeans_n_clusters.setToolTip("Number of clusters to form.")
        km_lay.addRow("Clusters:", self._kmeans_n_clusters)
        layout.addWidget(self._kmeans_container)

        # Run button
        self._run_btn = QPushButton("Run Clustering")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        # Metrics output
        self._metrics_output = QPlainTextEdit()
        self._metrics_output.setReadOnly(True)
        self._metrics_output.setPlaceholderText("Clustering metrics will appear here after running.")
        self._metrics_output.setMinimumHeight(120)
        self._metrics_output.setMaximumHeight(300)
        self._metrics_output.setStyleSheet("""
            QPlainTextEdit {
                font-family: monospace;
                font-size: 11px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        layout.addWidget(self._metrics_output)
        layout.addStretch()

        self._refresh_channels()
        self._refresh_masks()
        self._on_method_changed()
        self._on_norm_changed()
        self._channel_model.modelReset.connect(self._refresh_channels)
        self._channel_model.modelReset.connect(self._refresh_masks)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_channels())
        self._channel_model.rowsInserted.connect(lambda: self._refresh_masks())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_channels())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_masks())
        self._channel_model.dataChanged.connect(lambda *_: self._refresh_channels())

    def _refresh_channels(self):
        checked_names = set()
        first_populate = self._channel_list.count() == 0
        if not first_populate:
            for i in range(self._channel_list.count()):
                item = self._channel_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    checked_names.add(item.text())
        self._channel_list.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_mask or ch.is_cell_mask or ch.is_type_mask or getattr(ch, 'is_region', False):
                continue
            item = QListWidgetItem(ch.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = True if first_populate else ch.name in checked_names
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._channel_list.addItem(item)

    def _set_all_channels(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self._channel_list.count()):
            self._channel_list.item(i).setCheckState(state)

    def _refresh_masks(self):
        current = self._mask_combo.currentText()
        self._mask_combo.clear()
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if ch.is_mask:
                self._mask_combo.addItem(ch.name, i)
        idx = self._mask_combo.findText(current)
        if idx >= 0: self._mask_combo.setCurrentIndex(idx)

    def _on_method_changed(self):
        method = self._method_combo.currentText().lower()
        self._leiden_container.setVisible(method == "leiden")
        self._louvain_container.setVisible(method == "louvain")
        self._phenograph_container.setVisible(method == "phenograph")
        self._flowsom_container.setVisible(method == "flowsom")
        self._dbscan_container.setVisible(method == "dbscan")
        self._kmeans_container.setVisible(method == "kmeans")
        self._hierarchical_container.setVisible(method == "hierarchical")
        # Default PCA on for DBSCAN (required), off for others
        self._pca_check.setChecked(method == "dbscan")

    def _on_hc_linkage_changed(self):
        """Disable the metric combo when ward linkage is selected (always Euclidean)."""
        is_ward = self._hc_linkage.currentText() == "ward"
        self._hc_metric.setEnabled(not is_ward)
        if is_ward:
            self._hc_metric.setCurrentIndex(0)  # euclidean

    def _on_norm_changed(self):
        norm = self._norm_combo.currentData()
        self._arcsinh_container.setVisible(norm == "arcsinh")
        self._logz_container.setVisible(norm == "log-z")

    def _on_run(self):
        if self._mask_combo.currentIndex() < 0:
            return
        method = self._method_combo.currentText().lower()
        norm = self._norm_combo.currentData()
        selected_channels = []
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_channels.append(item.data(Qt.ItemDataRole.UserRole))
        params = {
            "mask_index": self._mask_combo.currentData(),
            "method": method,
            "normalization": norm,
            "use_pca": self._pca_check.isChecked(),
            "pca_components": int(self._pca_n_override.text()) if self._pca_n_override.text().strip() else None,
            "selected_channels": selected_channels,
        }
        # Normalization-specific params
        if norm == "arcsinh":
            params["cofactor"] = float(self._arcsinh_cofactor.text() or 5)
        elif norm == "log-z":
            params["skewness_threshold"] = float(self._logz_skewness.text() or 1)
        # Method-specific params
        if method == "leiden":
            params["resolution"] = float(self._leiden_resolution.text() or 0.5)
        elif method == "louvain":
            params["resolution"] = float(self._louvain_resolution.text() or 0.5)
        elif method == "phenograph":
            params["k"] = int(self._phenograph_k.text() or 30)
        elif method == "flowsom":
            params["xdim"] = int(self._flowsom_xdim.text() or 10)
            params["ydim"] = int(self._flowsom_ydim.text() or 10)
            params["n_clusters"] = int(self._flowsom_n_clusters.text() or 10)
        elif method == "dbscan":
            eps_text = self._dbscan_eps.text().strip()
            params["eps"] = float(eps_text) if eps_text else None
            params["min_samples"] = int(self._dbscan_min_samples.text() or 10)
        elif method == "kmeans":
            params["n_clusters"] = int(self._kmeans_n_clusters.text() or 5)
        elif method == "hierarchical":
            params["n_clusters"] = int(self._hc_n_clusters.text() or 5)
            params["linkage"] = self._hc_linkage.currentText()
            params["metric"] = self._hc_metric.currentText()
        self.runRequested.emit(params)

    def set_metrics(self, text: str) -> None:
        self._metrics_output.setPlainText(text)

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._run_btn.setEnabled(enabled)


class OperationsPanel(QWidget):
    """Right-side panel with collapsible sections for Pre-processing and Segmentation."""

    DEFAULT_WIDTH = 300
    runPreprocessingRequested = Signal(dict)
    runBrightfieldRequested = Signal(dict)
    runSegmentationRequested = Signal(dict)
    runMaskProcessingRequested = Signal(dict)
    runCellPositivityRequested = Signal(dict)
    runCellIdentificationRequested = Signal()
    runClusteringRequested = Signal(dict)
    runThresholdComputeRequested = Signal(dict)
    applyThresholdRequested = Signal(dict)
    segmentationFinished = Signal(object)

    def __init__(self, channel_model: ChannelListModel, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        self.setMinimumWidth(325)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 6, 0)
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
        spacer_pixmap = QPixmap(1, 24)
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

    def reset(self):
        """Restore all operations panel settings to defaults."""
        self.stop_loading()
        self._radio_full.setChecked(True)
        self._radio_new_mask.setChecked(True)
        self._radio_selected_region.setChecked(False)
        
        # Optionally: Collapse all sections to start fresh
        # (Though some users might prefer them staying open, defaults are usually best)
        # for i in range(self._container_layout.count()):
        #     item = self._container_layout.itemAt(i)
        #     if item and item.widget() and isinstance(item.widget(), CollapsiblePanel):
        #         item.widget().set_expanded(False)

    def _setup_preprocessing_section(self):
        panel = CollapsiblePanel("Pre-processing", collapsed=True)
        self._container_layout.addWidget(panel)

        self._pre_tabs = OperationsTabWidget()
        self._pre_tabs.setIconSize(QSize(1, 24))
        self._filter_tab = FilterTab(self._channel_model)
        self._merge_tab = MergeTab(self._channel_model)
        self._brightfield_tab = BrightfieldTab(self._channel_model)

        self._filter_tab.runRequested.connect(self._on_run_preprocessing)
        self._merge_tab.runRequested.connect(self._on_run_preprocessing)
        self._brightfield_tab.runRequested.connect(self._on_run_brightfield)

        self._pre_tabs.addTab(self._merge_tab, self._spacer_icon, "Merge")
        self._pre_tabs.addTab(self._filter_tab, self._spacer_icon, "Filter")
        self._pre_tabs.addTab(self._brightfield_tab, self._spacer_icon, "Brightfield")
        panel.addWidget(self._pre_tabs)

    def _setup_segmentation_section(self):
        panel = CollapsiblePanel("Segmentation", collapsed=True)
        self._container_layout.addWidget(panel)

        # Region mode toggle
        region_lay = QHBoxLayout()
        self._radio_full = QRadioButton("Full image")
        self._radio_visible = QRadioButton("Visible region")
        self._radio_selected_region = QRadioButton("Selected region")
        self._radio_full.setChecked(True)
        self._radio_selected_region.setToolTip(
            "Run segmentation inside the bounding box of the currently selected region polygon.\n"
            "Only cells fully contained within the region polygon will be added to the mask."
        )

        self._region_group = QButtonGroup(self)
        self._region_group.addButton(self._radio_full)
        self._region_group.addButton(self._radio_visible)
        self._region_group.addButton(self._radio_selected_region)

        region_lay.addWidget(self._radio_full)
        region_lay.addWidget(self._radio_visible)
        region_lay.addWidget(self._radio_selected_region)
        region_lay.addStretch()
        panel.addLayout(region_lay)

        # Target mask toggle
        target_lay = QHBoxLayout()
        self._radio_new_mask = QRadioButton("New mask")
        self._radio_overwrite = QRadioButton("Overwrite selected mask")
        self._radio_new_mask.setChecked(True)

        self._target_group = QButtonGroup(self)
        self._target_group.addButton(self._radio_new_mask)
        self._target_group.addButton(self._radio_overwrite)

        target_lay.addWidget(self._radio_new_mask)
        target_lay.addWidget(self._radio_overwrite)
        target_lay.addStretch()
        panel.addLayout(target_lay)

        self._seg_tabs = OperationsTabWidget()
        self._seg_tabs.setIconSize(QSize(1, 24))
        
        self._watershed_tab = WatershedTab(self._channel_model)
        self._watershed_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._watershed_tab, self._spacer_icon, "Watershed")
        
        self._instanseg_tab = InstanSegTab(self._channel_model)
        self._instanseg_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._instanseg_tab, self._spacer_icon, "InstanSeg")
        
        self._mesmer_tab = MesmerTab(self._channel_model)
        self._mesmer_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._mesmer_tab, self._spacer_icon, "Mesmer")
        
        self._stardist_tab = StarDistTab(self._channel_model)
        self._stardist_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._stardist_tab, self._spacer_icon, "StarDist")
        
        self._cellpose_tab = CellposeTab(self._channel_model)
        self._cellpose_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._cellpose_tab, self._spacer_icon, "Cellpose")
        
        self._omnipose_tab = OmniposeTab(self._channel_model)
        self._omnipose_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._omnipose_tab, self._spacer_icon, "Omnipose")
        panel.addWidget(self._seg_tabs)

    def _setup_mask_processing_section(self):
        panel = CollapsiblePanel("Mask Processing", collapsed=True)
        self._container_layout.addWidget(panel)

        self._mask_tabs = OperationsTabWidget()
        self._mask_tabs.setIconSize(QSize(1, 24))
        
        self._filter_size_tab = MaskFilterSizeTab(self._channel_model)
        self._filter_size_tab.runRequested.connect(self._on_run_mask_processing)
        self._mask_tabs.addTab(self._filter_size_tab, self._spacer_icon, "Filter")
        
        self._cell_sampler_tab = CellSamplerTab(self._channel_model)
        self._cell_sampler_tab.runRequested.connect(self._on_run_mask_processing)
        self._mask_tabs.addTab(self._cell_sampler_tab, self._spacer_icon, "Sampler")
        
        self._expansion_tab = MaskExpansionTab(self._channel_model)
        self._expansion_tab.runRequested.connect(self._on_run_mask_processing)
        self._mask_tabs.addTab(self._expansion_tab, self._spacer_icon, "Expand")
        
        panel.addWidget(self._mask_tabs)

    def _setup_cell_positivity_section(self):
        panel = CollapsiblePanel("Cell positivity", collapsed=True)
        self._container_layout.addWidget(panel)

        self._pos_tabs = OperationsTabWidget()
        self._pos_tabs.setIconSize(QSize(1, 24))
        
        # AI Tab
        ai_tab = QWidget()
        ai_lay = QVBoxLayout(ai_tab)
        ai_lay.setContentsMargins(12, 12, 12, 12)
        ai_lay.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._pos_mask_combo = QComboBox()
        self._pos_mask_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._pos_mask_combo.setMinimumWidth(50)
        form.addRow("Mask:", self._pos_mask_combo)
        ai_lay.addLayout(form)

        self._pos_run_btn = QPushButton("Detect Cell Positivity")
        self._pos_run_btn.clicked.connect(self._on_run_cell_positivity)
        ai_lay.addWidget(self._pos_run_btn)
        ai_lay.addStretch()

        # Thresholds Tab
        self._thresh_tab = ThresholdPositivityTab(self._channel_model)
        self._thresh_tab.runThresholdComputeRequested.connect(self._on_run_threshold_compute)
        self._thresh_tab.applyThresholdRequested.connect(self._on_apply_threshold)

        self._pos_tabs.addTab(ai_tab, self._spacer_icon, "AI")
        self._pos_tabs.addTab(self._thresh_tab, self._spacer_icon, "Thresholds")
        panel.addWidget(self._pos_tabs)

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

        self._ident_tabs = OperationsTabWidget()
        self._ident_tabs.setIconSize(QSize(1, 24))
        
        # Gating Tab
        gating_tab = QWidget()
        gating_lay = QVBoxLayout(gating_tab)
        gating_lay.setContentsMargins(12, 12, 12, 12)
        gating_lay.setSpacing(8)

        self._ident_run_btn = QPushButton("Identify Cells")
        self._ident_run_btn.clicked.connect(self._on_run_cell_identification)
        gating_lay.addWidget(self._ident_run_btn)
        gating_lay.addStretch()

        # Clustering Tab
        self._clustering_tab = ClusteringTab(self._channel_model)
        self._clustering_tab.runRequested.connect(self._on_run_clustering)

        self._ident_tabs.addTab(gating_tab, self._spacer_icon, "Gating")
        self._ident_tabs.addTab(self._clustering_tab, self._spacer_icon, "Clustering")
        panel.addWidget(self._ident_tabs)


    def _on_run_preprocessing(self, params):
        self._filter_tab.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runPreprocessingRequested.emit(params)

    def _on_run_brightfield(self, params):
        self._brightfield_tab.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runBrightfieldRequested.emit(params)

    def _on_run_segmentation(self, params):
        target_mode = "new" if self._radio_new_mask.isChecked() else "overwrite"
        if self._radio_full.isChecked():
            params["region_mode"] = "full"
        elif self._radio_visible.isChecked():
            params["region_mode"] = "visible"
        else:
            params["region_mode"] = "selected_region"
        params["target_mode"] = target_mode

        # Validate selected region mode — find any selected region channel
        if params["region_mode"] == "selected_region":
            region_ch = None
            for i, ch in enumerate(self._channel_model._channels):
                if ch.selected and getattr(ch, 'is_region', False):
                    region_ch = ch
                    params["region_channel_index"] = i
                    break
            if region_ch is None:
                QMessageBox.warning(
                    self, "No region selected",
                    "Please select a region channel (drawn polygon) before running segmentation in 'Selected region' mode."
                )
                return

        # Validate overwrite mode
        if target_mode == "overwrite":
            # When in selected_region mode the region is the selected channel,
            # so look for any mask channel rather than requiring it to be "selected".
            if params["region_mode"] == "selected_region":
                mask_ch = None
                for i, ch in enumerate(self._channel_model._channels):
                    if ch.is_mask or ch.is_cell_mask or ch.is_type_mask:
                        mask_ch = ch
                        params["target_mask_index"] = i
                        break
                if mask_ch is None:
                    QMessageBox.warning(self, "No mask found",
                        "No mask channel found to overwrite. Please create a mask first.")
                    return
            else:
                # Normal case: require a mask to be the selected channel
                sel_ch = self._channel_model.selected_channel()
                if not sel_ch or not (sel_ch.is_mask or sel_ch.is_cell_mask or sel_ch.is_type_mask):
                    QMessageBox.warning(self, "No mask selected", "Please select a mask channel to overwrite.")
                    return
                params["target_mask_index"] = self._channel_model._channels.index(sel_ch)

        self._stardist_tab.setEnabled(False)
        self._cellpose_tab.setEnabled(False)
        self._omnipose_tab.setEnabled(False)
        self._instanseg_tab.setEnabled(False)
        self._watershed_tab.setEnabled(False)
        self._mesmer_tab.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runSegmentationRequested.emit(params)

    def _on_run_mask_processing(self, params):
        self._expansion_tab.setEnabled(False)
        self._filter_size_tab.setEnabled(False)
        self._cell_sampler_tab.setEnabled(False)
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

    def _on_run_threshold_compute(self, params):
        self._thresh_tab.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self.runThresholdComputeRequested.emit(params)

    def _on_apply_threshold(self, params):
        """Forwarded directly to MainWindow — intentionally not blocking the UI."""
        self.applyThresholdRequested.emit(params)

    def _on_run_cell_identification(self):
        self._ident_run_btn.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runCellIdentificationRequested.emit()

    def _on_run_clustering(self, params):
        self._clustering_tab.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runClusteringRequested.emit(params)

    def stop_loading(self):
        self._filter_tab.setEnabled(True)
        self._brightfield_tab.setEnabled(True)
        self._stardist_tab.setEnabled(True)
        self._cellpose_tab.setEnabled(True)
        self._omnipose_tab.setEnabled(True)
        self._instanseg_tab.setEnabled(True)
        self._mesmer_tab.setEnabled(True)
        self._watershed_tab.setEnabled(True)
        self._filter_size_tab.setEnabled(True)
        self._expansion_tab.setEnabled(True)
        self._cell_sampler_tab.setEnabled(True)
        self._pos_run_btn.setEnabled(True)
        self._thresh_tab.setEnabled(True)
        self._ident_run_btn.setEnabled(True)
        self._clustering_tab.setEnabled(True)
        self._progress.setVisible(False)

    def set_clustering_metrics(self, text: str) -> None:
        self._clustering_tab.set_metrics(text)

    @Slot(int, int)
    def set_progress_info(self, val, total):
        self._progress.setRange(0, total)
        self._progress.setValue(val)
