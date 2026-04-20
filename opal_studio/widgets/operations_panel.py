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
        self.setElideMode(Qt.TextElideMode.ElideRight)
        self.setUsesScrollButtons(True)
        self.setTabBarAutoHide(False)
        self.tabBar().setExpanding(True)
        self.currentChanged.connect(self._on_current_changed)

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
        self.p_high = QLineEdit("99.0")
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
        
        self.clahe_kernel = QLineEdit("10")
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
        self._model_combo.addItems(["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"])
        self._scan_models()
        form.addRow("Model:", self._model_combo)

        self._prob_thresh = QLineEdit()
        self._prob_thresh.setPlaceholderText("Auto")
        self._prob_thresh.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self._prob_thresh.setFixedWidth(60)
        form.addRow("prob_thresh:", self._prob_thresh)

        self._nms_thresh = QLineEdit("0.3")
        self._nms_thresh.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self._nms_thresh.setFixedWidth(60)
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
        self._model_combo.addItems(["cyto", "nuclei", "cyto2"])
        self._scan_models()
        form.addRow("Model:", self._model_combo)

        self._diameter = QLineEdit()
        self._diameter.setPlaceholderText("Auto")
        self._diameter.setValidator(QDoubleValidator(0.1, 1000.0, 2))
        self._diameter.setFixedWidth(60)
        form.addRow("Diameter:", self._diameter)
        
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
            "brightfield_nuclei",
            "multi_channel_fluorescence"
        ])
        form.addRow("Model:", self._model_combo)

        # Pixel Size
        self._pixel_size = QLineEdit("1.0")
        self._pixel_size.setValidator(QDoubleValidator(0.001, 1000.0, 3))
        self._pixel_size.setFixedWidth(60)
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

        # API Key
        self._api_key = QLineEdit("0eJIjCpR.fYenvVnMb4ZAKCjxYnjZ1R2V6kvLdq5V")
        self._api_key.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        form.addRow("API Key:", self._api_key)

        # Pixel Size
        self._pixel_size = QLineEdit("1.0")
        self._pixel_size.setValidator(QDoubleValidator(0.001, 1000.0, 3))
        self._pixel_size.setFixedWidth(60)
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
        form.addRow("Outline:", self._outline_sigma)

        # Threshold
        self._threshold = QLineEdit("1")
        self._threshold.setValidator(QDoubleValidator(0.01, 100.0, 2))
        self._threshold.setFixedWidth(80)
        self._threshold.setToolTip(
            "Voronoi: offset from local threshold (>1 = stricter, <1 = more permissive).\n"
            "Gauss: multiplier on Otsu threshold (>1 = stricter, <1 = more permissive)."
        )
        form.addRow("Threshold:", self._threshold)

        self._min_mean_intensity = QLineEdit("0")
        self._min_mean_intensity.setValidator(QDoubleValidator(-1000000.0, 1000000.0, 4))
        self._min_mean_intensity.setFixedWidth(80)
        self._min_mean_intensity.setToolTip("Minimum mean intensity in a cell to keep it.")
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


class OperationsPanel(QWidget):
    """Right-side panel with collapsible sections for Pre-processing and Segmentation."""

    WIDTH = 420
    runPreprocessingRequested = Signal(dict)
    runSegmentationRequested = Signal(dict)
    runMaskProcessingRequested = Signal(dict)
    runCellPositivityRequested = Signal(dict)
    runCellIdentificationRequested = Signal()
    segmentationFinished = Signal(object)

    def __init__(self, channel_model: ChannelListModel, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        self.setFixedWidth(self.WIDTH)
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
        
        self._pre_tabs.addTab(self._equalize_tab, self._spacer_icon, "Equalize")
        self._pre_tabs.addTab(self._filter_tab, self._spacer_icon, "Filter")
        self._pre_tabs.addTab(self._merge_tab, self._spacer_icon, "Merge")
        panel.addWidget(self._pre_tabs)

    def _setup_segmentation_section(self):
        panel = CollapsiblePanel("Segmentation", collapsed=True)
        self._container_layout.addWidget(panel)

        self._seg_tabs = OperationsTabWidget()
        self._seg_tabs.setIconSize(QSize(1, 24))
        self._stardist_tab = StarDistTab(self._channel_model)
        self._stardist_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._stardist_tab, self._spacer_icon, "StarDist")
        self._cellpose_tab = CellposeTab(self._channel_model)
        self._cellpose_tab.runRequested.connect(self._on_run_segmentation)
        self._seg_tabs.addTab(self._cellpose_tab, self._spacer_icon, "Cellpose")
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
        thresh_tab = QWidget()
        thresh_lay = QVBoxLayout(thresh_tab)
        thresh_lay.addWidget(QLabel("Threshold parameters coming soon..."))
        thresh_lay.addStretch()

        self._pos_tabs.addTab(ai_tab, self._spacer_icon, "AI")
        self._pos_tabs.addTab(thresh_tab, self._spacer_icon, "Thresholds")
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
        self._stardist_tab.setEnabled(False)
        self._cellpose_tab.setEnabled(False)
        self._instanseg_tab.setEnabled(False)
        self._watershed_tab.setEnabled(False)
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

    def _on_run_cell_identification(self):
        self._ident_run_btn.setEnabled(False)
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self.runCellIdentificationRequested.emit()

    def stop_loading(self):
        self._equalize_tab.setEnabled(True)
        self._filter_tab.setEnabled(True)
        self._stardist_tab.setEnabled(True)
        self._cellpose_tab.setEnabled(True)
        self._instanseg_tab.setEnabled(True)
        self._watershed_tab.setEnabled(True)
        self._filter_size_tab.setEnabled(True)
        self._expansion_tab.setEnabled(True)
        self._cell_sampler_tab.setEnabled(True)
        self._pos_run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._watershed_tab.setEnabled(True)
        self._filter_size_tab.setEnabled(True)
        self._expansion_tab.setEnabled(True)
        self._cell_sampler_tab.setEnabled(True)
        self._pos_run_btn.setEnabled(True)
        self._ident_run_btn.setEnabled(True)
        self._progress.setVisible(False)

    @Slot(int, int)
    def set_progress_info(self, val, total):
        self._progress.setRange(0, total)
        self._progress.setValue(val)
