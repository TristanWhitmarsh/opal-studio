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
    QHBoxLayout, QScrollArea, QTabWidget, QToolButton,
    QListWidget, QListWidgetItem, QAbstractItemView
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

class EqualizeTab(QWidget):
    """Percentile normalization and CLAHE equalization."""
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
        
        # Channel Selector
        self._channel_combo = QComboBox()
        self._channel_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._channel_combo.setMinimumWidth(50)
        form.addRow("Channel:", self._channel_combo)

        # Percentiles
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
        form.addRow("Range:", p_lay)

        # CLAHE (Always on, just show parameters)
        self.clahe_clip = QLineEdit("0.05")
        self.clahe_clip.setValidator(QDoubleValidator(0.001, 1.0, 3))
        self.clahe_clip.setFixedWidth(60)
        form.addRow("Clip:", self.clahe_clip)
        
        self.clahe_kernel = QLineEdit("20")
        self.clahe_kernel.setValidator(QIntValidator(8, 256))
        self.clahe_kernel.setFixedWidth(60)
        form.addRow("Kernel:", self.clahe_kernel)

        layout.addLayout(form)
        
        self._run_btn = QPushButton("Run Equalize")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)
        layout.addStretch()

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

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        self.runRequested.emit({
            "is_filter": False,
            "channel_index": self._channel_combo.currentData(),
            "p_low": float(self.p_low.text() or 1.0),
            "p_high": float(self.p_high.text() or 99.8),
            "apply_clahe": True,
            "clahe_clip": float(self.clahe_clip.text() or 0.02),
            "clahe_kernel": (int(self.clahe_kernel.text() or 50), int(self.clahe_kernel.text() or 50))
        })

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self._run_btn.setEnabled(enabled)

class FilterTab(QWidget):
    """Smoothing and morphological filters."""
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
        self.filter_type.addItems(["Median", "Opening"])
        form.addRow("Type:", self.filter_type)
        
        self.filter_value = QLineEdit("3")
        self.filter_value.setValidator(QIntValidator(1, 101))
        self.filter_value.setFixedWidth(60)
        form.addRow("Size:", self.filter_value)
        
        layout.addLayout(form)
        
        self._run_btn = QPushButton("Run Filter")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)
        layout.addStretch()

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

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        self.runRequested.emit({
            "is_filter": True,
            "channel_index": self._channel_combo.currentData(),
            "filter_type": self.filter_type.currentText().lower(),
            "filter_value": int(self.filter_value.text() or 3)
        })

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
            if not ch.is_mask:
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

        self._prob_thresh = QLineEdit("0.1")
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
            if not ch.is_mask:
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)

    def _scan_models(self):
        models_dir = os.path.join(os.getcwd(), "models", "omnipose")
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
        self._model_combo.setToolTip("InstanSeg models for single-channel nuclei or multi-channel fluorescence (nuclei and cells).")
        form.addRow("Model:", self._model_combo)

        # Pixel Size
        self._pixel_size = QLineEdit("1.0")
        self._pixel_size.setValidator(QDoubleValidator(0.001, 1000.0, 3))
        self._pixel_size.setFixedWidth(60)
        self._pixel_size.setToolTip("The resolution of the image in micrometers per pixel (\u03bcm/px). Used to scale the model's receptive field to the physical size of cells.")
        form.addRow("Pixel Size (\u03bcm):", self._pixel_size)
        
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
            if not ch.is_mask:
                self._channel_combo.addItem(ch.name, i)
        idx = self._channel_combo.findText(current)
        if idx >= 0: self._channel_combo.setCurrentIndex(idx)

    def _on_run(self):
        if self._channel_combo.currentIndex() < 0: return
        self.runRequested.emit({
            "method": "instanseg",
            "channel_indices": [self._channel_combo.currentData()],
            "model_name": self._model_combo.currentText(),
            "pixel_size": float(self._pixel_size.text() or 1.0),
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

        # Model Parameters
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
        form.addRow("Nuclear:", self._nuclear_combo)

        # Membrane Channel
        self._membrane_combo = QComboBox()
        self._membrane_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._membrane_combo.setMinimumWidth(50)
        form.addRow("Membrane:", self._membrane_combo)

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
        
        layout.addLayout(form)

        # Run Button
        self._run_btn = QPushButton("Run Mesmer")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        self._refresh_channels()
        self._channel_model.modelReset.connect(self._refresh_channels)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_channels())
        self._channel_model.rowsRemoved.connect(lambda: self._refresh_channels())

    def _refresh_channels(self):
        n_current = self._nuclear_combo.currentText()
        m_current = self._membrane_combo.currentText()
        self._nuclear_combo.clear()
        self._membrane_combo.clear()
        self._membrane_combo.addItem("None", -1)
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if not ch.is_mask:
                self._nuclear_combo.addItem(ch.name, i)
                self._membrane_combo.addItem(ch.name, i)
        
        ni = self._nuclear_combo.findText(n_current)
        if ni >= 0: self._nuclear_combo.setCurrentIndex(ni)
        mi = self._membrane_combo.findText(m_current)
        if mi >= 0: self._membrane_combo.setCurrentIndex(mi)

    def _on_run(self):
        if self._nuclear_combo.currentIndex() < 0: return
        
        indices = [self._nuclear_combo.currentData()]
        if self._membrane_combo.currentData() != -1:
            indices.append(self._membrane_combo.currentData())
            
        self.runRequested.emit({
            "method": "mesmer",
            "channel_indices": indices,
            "api_key": self._api_key.text(),
            "pixel_size": float(self._pixel_size.text() or 1.0),
            "compartment": self._compartment_combo.currentText(),
        })

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


class OperationsPanel(QWidget):
    """Right-side panel with collapsible sections for Pre-processing and Segmentation."""

    DEFAULT_WIDTH = 300
    runPreprocessingRequested = Signal(dict)
    runSegmentationRequested = Signal(dict)
    runMaskProcessingRequested = Signal(dict)
    runCellPositivityRequested = Signal(dict)
    runCellIdentificationRequested = Signal()
    runThresholdComputeRequested = Signal(dict)
    applyThresholdRequested = Signal(dict)
    segmentationFinished = Signal(object)

    def __init__(self, channel_model: ChannelListModel, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        self.setMinimumWidth(300)
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

    def _setup_preprocessing_section(self):
        panel = CollapsiblePanel("Pre-processing", collapsed=True)
        self._container_layout.addWidget(panel)

        # Tabs for Equalize vs Filter vs Merge
        self._pre_tabs = OperationsTabWidget()
        self._pre_tabs.setIconSize(QSize(1, 24))
        self._equalize_tab = EqualizeTab(self._channel_model)
        self._filter_tab = FilterTab(self._channel_model)
        self._merge_tab = MergeTab(self._channel_model)
        
        self._equalize_tab.runRequested.connect(self._on_run_preprocessing)
        self._filter_tab.runRequested.connect(self._on_run_preprocessing)
        self._merge_tab.runRequested.connect(self._on_run_preprocessing)
        
        self._pre_tabs.addTab(self._merge_tab, self._spacer_icon, "Merge")
        self._pre_tabs.addTab(self._filter_tab, self._spacer_icon, "Filter")
        self._pre_tabs.addTab(self._equalize_tab, self._spacer_icon, "Equalize")
        panel.addWidget(self._pre_tabs)

    def _setup_segmentation_section(self):
        panel = CollapsiblePanel("Segmentation", collapsed=True)
        self._container_layout.addWidget(panel)

        # Region mode toggle
        region_lay = QHBoxLayout()
        region_lay.setContentsMargins(12, 5, 12, 5)
        self._radio_full = QRadioButton("Full image")
        self._radio_visible = QRadioButton("Visible region only")
        self._radio_full.setChecked(True)
        
        self._region_group = QButtonGroup(self)
        self._region_group.addButton(self._radio_full)
        self._region_group.addButton(self._radio_visible)
        
        region_lay.addWidget(self._radio_full)
        region_lay.addWidget(self._radio_visible)
        region_lay.addStretch()
        panel.addLayout(region_lay)

        # Target mask toggle
        target_lay = QHBoxLayout()
        target_lay.setContentsMargins(12, 0, 12, 5)
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
        self._stardist_tab = StarDistTab(self._channel_model)
        self._stardist_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._stardist_tab, self._spacer_icon, "StarDist")
        self._cellpose_tab = CellposeTab(self._channel_model)
        self._cellpose_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._cellpose_tab, self._spacer_icon, "Cellpose")
        self._omnipose_tab = OmniposeTab(self._channel_model)
        self._omnipose_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._omnipose_tab, self._spacer_icon, "Omnipose")
        self._instanseg_tab = InstanSegTab(self._channel_model)
        self._instanseg_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._instanseg_tab, self._spacer_icon, "InstanSeg")
        self._mesmer_tab = MesmerTab(self._channel_model)
        self._mesmer_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._mesmer_tab, self._spacer_icon, "Mesmer")
        self._watershed_tab = WatershedTab(self._channel_model)
        self._watershed_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._watershed_tab, self._spacer_icon, "Watershed")
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
        clustering_tab = QWidget()
        clustering_lay = QVBoxLayout(clustering_tab)
        clustering_lay.addWidget(QLabel("Clustering features coming soon..."))
        clustering_lay.addStretch()

        self._ident_tabs.addTab(gating_tab, self._spacer_icon, "Gating")
        self._ident_tabs.addTab(clustering_tab, self._spacer_icon, "Clustering")
        panel.addWidget(self._ident_tabs)


    def _on_run_preprocessing(self, params):
        self._equalize_tab.setEnabled(False)
        self._filter_tab.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runPreprocessingRequested.emit(params)

    def _on_run_segmentation(self, params):
        target_mode = "new" if self._radio_new_mask.isChecked() else "overwrite"
        params["region_mode"] = "full" if self._radio_full.isChecked() else "visible"
        params["target_mode"] = target_mode
        
        if target_mode == "overwrite":
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

    def stop_loading(self):
        self._equalize_tab.setEnabled(True)
        self._filter_tab.setEnabled(True)
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
        self._progress.setVisible(False)

    @Slot(int, int)
    def set_progress_info(self, val, total):
        self._progress.setRange(0, total)
        self._progress.setValue(val)
