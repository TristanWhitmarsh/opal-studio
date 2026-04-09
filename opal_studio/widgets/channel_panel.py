"""
Left-side channel panel — lists all channels with eye toggle,
colour swatch, name, and dual-handle range slider.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QPalette
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QPushButton, QCheckBox, QSizePolicy, QColorDialog, QFrame, QSlider,
    QTabWidget
)

from opal_studio.channel_model import Channel, ChannelListModel
from opal_studio.widgets.range_slider import RangeSlider


class ChannelPanel(QWidget):
    """
    Scrollable panel showing one row per channel.

    Each row contains:
      [eye toggle] [colour swatch] [name label] [range slider]
    """

    def __init__(self, model: ChannelListModel, parent=None):
        super().__init__(parent)
        self._model = model
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setBackgroundRole(QPalette.ColorRole.Window)
        self.setAutoFillBackground(True)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Tabbed interface
        self._tabs = QTabWidget()
        self._tabs.setIconSize(QSize(1, 20))
        
        # Inject an invisible native icon to force the tab bar to draw taller, preventing cut-off text
        spacer_pixmap = QPixmap(1, 20)
        spacer_pixmap.fill(Qt.GlobalColor.transparent)
        self._spacer_icon = QIcon(spacer_pixmap)
        
        main_layout.addWidget(self._tabs)

        # Header & Bulk Controls (Persistent within the Channels tab)
        self._header_area = QWidget()
        self._header_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        header_layout = QVBoxLayout(self._header_area)
        header_layout.setContentsMargins(8, 8, 8, 8)
        header_layout.setSpacing(8)

        header_top = QHBoxLayout()
        header_label = QLabel("Channels")
        header_top.addWidget(header_label)
        header_top.addStretch()

        btn_all_on = QPushButton("Show all")
        btn_all_off = QPushButton("Hide all")
        btn_all_on.clicked.connect(lambda: self._model.set_all_visible(True, include_masks=False))
        btn_all_off.clicked.connect(lambda: self._model.set_all_visible(False, include_masks=False))
        header_top.addWidget(btn_all_on)
        header_top.addWidget(btn_all_off)
        header_layout.addLayout(header_top)

        # Global Brightness Slider
        bright_layout = QHBoxLayout()
        bright_label = QLabel("Brightness")
        bright_layout.addWidget(bright_label)

        self._bright_slider = QSlider(Qt.Orientation.Horizontal)
        self._bright_slider.setRange(1, 200) # 0.01 to 2.0
        self._bright_slider.setValue(100) # 1.0 initial
        self._bright_slider.valueChanged.connect(self._on_brightness_changed)
        bright_layout.addWidget(self._bright_slider)
        header_layout.addLayout(bright_layout)
        
        # Manually trigger initial brightness from slider
        self._on_brightness_changed(self._bright_slider.value())

        # Tab 1: Channels (Standard QWidget containing the scroll area)
        self._channel_tab = QWidget()
        self._channel_tab_layout = QVBoxLayout(self._channel_tab)
        self._channel_tab_layout.setContentsMargins(0, 0, 0, 0)
        
        self._channel_scroll = QScrollArea()
        self._channel_scroll.setWidgetResizable(True)
        self._channel_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._channel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self._channel_container = QWidget()
        self._channel_container.setBackgroundRole(QPalette.ColorRole.Base)
        self._channel_container.setAutoFillBackground(True)
        self._channel_layout = QVBoxLayout(self._channel_container)
        self._channel_layout.setContentsMargins(6, 6, 6, 6)
        self._channel_layout.setSpacing(6)
        self._channel_scroll.setWidget(self._channel_container)
        
        self._channel_tab_layout.addWidget(self._channel_scroll)
        self._tabs.addTab(self._channel_tab, self._spacer_icon, "Channels")
        
        self._channel_layout.addWidget(self._header_area)
        self._channel_layout.addStretch()

        # Tab 2: Masks (Standard QWidget containing the scroll area)
        self._mask_tab = QWidget()
        self._mask_tab_layout = QVBoxLayout(self._mask_tab)
        self._mask_tab_layout.setContentsMargins(0, 0, 0, 0)

        self._mask_scroll = QScrollArea()
        self._mask_scroll.setWidgetResizable(True)
        self._mask_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._mask_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self._mask_container = QWidget()
        self._mask_container.setBackgroundRole(QPalette.ColorRole.Base)
        self._mask_container.setAutoFillBackground(True)
        self._mask_layout = QVBoxLayout(self._mask_container)
        self._mask_layout.setContentsMargins(6, 6, 6, 6)
        self._mask_layout.setSpacing(6)
        self._mask_scroll.setWidget(self._mask_container)
        
        self._mask_tab_layout.addWidget(self._mask_scroll)
        self._tabs.addTab(self._mask_tab, self._spacer_icon, "Masks")
        self._mask_layout.addStretch()

        # Tab 3: Cells
        self._cell_tab = QWidget()
        self._cell_tab_layout = QVBoxLayout(self._cell_tab)
        self._cell_tab_layout.setContentsMargins(0, 0, 0, 0)

        self._cell_scroll = QScrollArea()
        self._cell_scroll.setWidgetResizable(True)
        self._cell_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._cell_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self._cell_container = QWidget()
        self._cell_container.setBackgroundRole(QPalette.ColorRole.Base)
        self._cell_container.setAutoFillBackground(True)
        self._cell_layout = QVBoxLayout(self._cell_container)
        self._cell_layout.setContentsMargins(6, 6, 6, 6)
        self._cell_layout.setSpacing(6)
        self._cell_scroll.setWidget(self._cell_container)
        
        self._cell_tab_layout.addWidget(self._cell_scroll)
        self._tabs.addTab(self._cell_tab, self._spacer_icon, "Cells")
        self._cell_layout.addStretch()

        self._row_widgets: list[QWidget] = []

        # React to model changes
        self._model.modelReset.connect(self._rebuild)
        self._model.rowsInserted.connect(lambda: self._rebuild())
        self._model.rowsRemoved.connect(lambda: self._rebuild())

    # ------------------------------------------------------------------

    def _rebuild(self):
        """Recreate all row widgets from the model."""
        # Clear existing rows
        for w in self._row_widgets:
            w.setParent(None)
            w.deleteLater()
        self._row_widgets.clear()

        # Clear layouts
        while self._channel_layout.count():
            item = self._channel_layout.takeAt(0)
            if item.widget():
                w = item.widget()
                if w == self._header_area:
                    continue # Keep it
                w.deleteLater()

        while self._mask_layout.count():
            item = self._mask_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        while self._cell_layout.count():
            item = self._cell_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        # Add header back to top of channel layout
        self._channel_layout.addWidget(self._header_area)

        for row in range(self._model.rowCount()):
            ch = self._model.channel(row)
            widget = self._make_row(row, ch)
            if ch.is_cell_mask:
                self._cell_layout.addWidget(widget)
            elif ch.is_mask:
                self._mask_layout.addWidget(widget)
            else:
                self._channel_layout.addWidget(widget)
            self._row_widgets.append(widget)

        self._channel_layout.addStretch()
        self._mask_layout.addStretch()
        self._cell_layout.addStretch()

    def _make_row(self, row: int, ch: Channel) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Sunken)
        
        # Use the standard 'Window' background role for a native look
        frame.setBackgroundRole(QPalette.ColorRole.Window)
        frame.setAutoFillBackground(True)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Top line: eye | colour | name
        top = QHBoxLayout()
        top.setSpacing(6)

        # Visibility toggle
        vis_cb = QCheckBox()
        vis_cb.setChecked(ch.visible)
        vis_cb.toggled.connect(lambda checked, r=row: self._toggle_vis(r, checked))
        top.addWidget(vis_cb)

        # Colour swatch
        swatch = QPushButton()
        swatch.setFixedSize(20, 20)
        swatch.setStyleSheet(
            f"background: {ch.color.name()}; border: 1px solid #555; border-radius: 3px;"
        )
        swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        swatch.clicked.connect(lambda _, r=row, s=swatch: self._pick_color(r, s))
        top.addWidget(swatch)

        # Name
        name = QLabel(ch.name)
        name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top.addWidget(name)

        # Delete button for mask / cell rows
        if ch.is_mask or ch.is_cell_mask:
            del_btn = QPushButton("🗑")
            del_btn.setFixedSize(24, 24)
            del_btn.setToolTip("Delete this mask")
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.clicked.connect(lambda _, r=row: self._model.remove_channel(r))
            top.addWidget(del_btn)

        layout.addLayout(top)

        # Control area (Window for intensity, Opacity for mask)
        if ch.is_mask or ch.is_cell_mask:
            trans_layout = QHBoxLayout()
            trans_label = QLabel("Opacity")
            trans_layout.addWidget(trans_label)

            opacity_slider = QSlider(Qt.Orientation.Horizontal)
            opacity_slider.setRange(0, 100)
            opacity_slider.setValue(int(ch.range_max * 100))
            opacity_slider.valueChanged.connect(lambda val, r=row: self._model.setData(self._model.index(r), val/100.0, ChannelListModel.RangeMaxRole))
            trans_layout.addWidget(opacity_slider)
            layout.addLayout(trans_layout)
        else:
            # Range slider for intensity window
            slider = RangeSlider(color=ch.color)
            slider.set_range(ch.range_min, ch.range_max)# Add this line to force a smaller minimum width!
            slider.setMinimumWidth(50)
            slider.rangeChanged.connect(lambda mn, mx, r=row: self._range_changed(r, mn, mx))
            layout.addWidget(slider)

        return frame

    # ---- callbacks ----------------------------------------------------

    def _toggle_vis(self, row: int, checked: bool):
        idx = self._model.index(row)
        self._model.setData(idx, checked, ChannelListModel.VisibleRole)

    def _range_changed(self, row: int, mn: float, mx: float):
        idx = self._model.index(row)
        self._model.setData(idx, mn, ChannelListModel.RangeMinRole)
        self._model.setData(idx, mx, ChannelListModel.RangeMaxRole)

    def _pick_color(self, row: int, swatch: QPushButton):
        ch = self._model.channel(row)
        color = QColorDialog.getColor(ch.color, self, "Channel Colour")
        if color.isValid():
            idx = self._model.index(row)
            self._model.setData(idx, color, ChannelListModel.ColorRole)
            swatch.setStyleSheet(
                f"background: {color.name()}; border: 1px solid #555; border-radius: 3px;"
            )

    def _on_brightness_changed(self, val: int):
        self._model.brightness = val / 100.0
