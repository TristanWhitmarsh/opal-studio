"""
Left-side channel panel — lists all channels with eye toggle,
colour swatch, name, and dual-handle range slider.
"""

from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, QSize, Signal, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QPalette

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QPushButton, QSizePolicy, QColorDialog, QFrame, QSlider,
    QTabWidget
)

class ClickableFrame(QFrame):
    clicked = Signal()
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

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
        self._tabs.setIconSize(QSize(1, 24))
        
        # Inject an invisible native icon to force the tab bar to draw taller, preventing cut-off text
        spacer_pixmap = QPixmap(1, 24)
        spacer_pixmap.fill(Qt.GlobalColor.transparent)
        self._spacer_icon = QIcon(spacer_pixmap)
        
        # Cache toggle icons
        icons_dir = Path(__file__).resolve().parent.parent / "icons"
        self._eye_open_icon = QIcon(str(icons_dir / "eye_open.png"))
        self._eye_closed_icon = QIcon(str(icons_dir / "eye_closed.png"))
        self._contour_open_icon = QIcon(str(icons_dir / "eye_open_contour.png"))
        self._contour_closed_icon = QIcon(str(icons_dir / "eye_closed_contour.png"))
        self._delete_icon = QIcon(str(icons_dir / "delete.png"))
        
        dash_pix = QPixmap(16, 16)
        dash_pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(dash_pix)
        pen = p.pen()
        pen.setColor(QColor(0, 0, 0))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(3, 8, 13, 8)
        p.end()
        self._dash_icon = QIcon(dash_pix)
        
        main_layout.addWidget(self._tabs)        # Header & Bulk Controls (Persistent within the Channels tab)
        self._header_area = QWidget()
        self._header_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        header_layout = QVBoxLayout(self._header_area)
        header_layout.setContentsMargins(8, 8, 8, 8)
        header_layout.setSpacing(6)

        # 1. Global Brightness (At very top)
        bright_layout = QHBoxLayout()
        bright_label = QLabel("Overall brightness")
        bright_layout.addWidget(bright_label)

        self._bright_slider = QSlider(Qt.Orientation.Horizontal)
        self._bright_slider.setRange(1, 200) # 0.01 to 2.0
        self._bright_slider.setValue(100) # 1.0 initial
        self._bright_slider.valueChanged.connect(self._on_brightness_changed)
        bright_layout.addWidget(self._bright_slider)
        header_layout.addLayout(bright_layout)

        # 2. Bulk Controls line
        header_top = QHBoxLayout()
        header_top.addWidget(QLabel("Channels"))
        header_top.addStretch()

        btn_all_on = QPushButton("Show all")
        btn_all_off = QPushButton("Hide all")
        btn_all_on.clicked.connect(lambda: self._model.set_all_visible(True, include_masks=False))
        btn_all_off.clicked.connect(lambda: self._model.set_all_visible(False, include_masks=False))
        header_top.addWidget(btn_all_on)
        header_top.addWidget(btn_all_off)
        header_layout.addLayout(header_top)
        
        # 3. Selected Channel Controls (Alpha & Limits) - Integrated into header
        self._sel_group = QWidget()
        self._sel_group.setEnabled(False) # Always visible but disabled if no selection
        sel_layout = QVBoxLayout(self._sel_group)
        sel_layout.setContentsMargins(0, 0, 0, 0)
        sel_layout.setSpacing(6)
        
        alpha_layout = QHBoxLayout()
        alpha_label = QLabel("Alpha")
        alpha_label.setFixedWidth(40)
        alpha_layout.addWidget(alpha_label)
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.valueChanged.connect(self._on_header_alpha_changed)
        alpha_layout.addWidget(self._alpha_slider)
        sel_layout.addLayout(alpha_layout)
        
        limits_layout = QHBoxLayout()
        limits_label = QLabel("Limits")
        limits_label.setFixedWidth(40)
        limits_layout.addWidget(limits_label)
        self._limits_slider = RangeSlider()
        self._limits_slider.rangeChanged.connect(self._on_header_limits_changed)
        limits_layout.addWidget(self._limits_slider)
        sel_layout.addLayout(limits_layout)
        
        header_layout.addWidget(self._sel_group)

        # Manually trigger initial brightness from slider
        self._on_brightness_changed(self._bright_slider.value())

        # Tab 1: Channels (Standard QWidget containing the scroll area)
        self._channel_tab = QWidget()
        self._channel_tab_layout = QVBoxLayout(self._channel_tab)
        self._channel_tab_layout.setContentsMargins(0, 0, 0, 0)
        
        self._channel_tab_layout.addWidget(self._header_area)

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

        # Tab 2: Masks (Standard QWidget containing the scroll area)
        self._mask_tab = QWidget()
        self._mask_tab_layout = QVBoxLayout(self._mask_tab)
        self._mask_tab_layout.setContentsMargins(0, 0, 0, 0)

        # Mask Header Area
        self._mask_header = QWidget()
        m_head_layout = QVBoxLayout(self._mask_header)
        m_head_layout.setContentsMargins(8, 8, 8, 8)
        m_head_layout.setSpacing(6)
        
        m_op_layout = QHBoxLayout()
        m_op_layout.addWidget(QLabel("Opacity"))
        self._mask_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._mask_opacity_slider.setRange(0, 100)
        self._mask_opacity_slider.setEnabled(False)
        self._mask_opacity_slider.valueChanged.connect(self._on_header_mask_opacity_changed)
        m_op_layout.addWidget(self._mask_opacity_slider)
        m_head_layout.addLayout(m_op_layout)
        
        self._mask_tab_layout.addWidget(self._mask_header)

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

        self._cell_header = QWidget()
        self._cell_header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        cell_header_layout = QHBoxLayout(self._cell_header)
        cell_header_layout.setContentsMargins(8, 8, 8, 8)
        cell_header_layout.addWidget(QLabel("Opacity"))
        self._cell_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._cell_opacity_slider.setRange(0, 100)
        self._cell_opacity_slider.setEnabled(False)
        self._cell_opacity_slider.valueChanged.connect(self._on_header_cell_opacity_changed)
        cell_header_layout.addWidget(self._cell_opacity_slider)
        self._cell_tab_layout.addWidget(self._cell_header)

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

        # Tab 4: Types
        self._type_tab = QWidget()
        self._type_tab_layout = QVBoxLayout(self._type_tab)
        self._type_tab_layout.setContentsMargins(0, 0, 0, 0)

        self._type_header = QWidget()
        self._type_header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        type_header_layout = QVBoxLayout(self._type_header)
        type_header_layout.setContentsMargins(8, 8, 8, 8)
        
        type_top = QHBoxLayout()
        type_top.addWidget(QLabel("Phenotypes"))
        type_top.addStretch()
        btn_type_on = QPushButton("Show all")
        btn_type_off = QPushButton("Hide all")
        btn_type_on.clicked.connect(lambda: self._model.set_category_visible("type", True))
        btn_type_off.clicked.connect(lambda: self._model.set_category_visible("type", False))
        type_top.addWidget(btn_type_on)
        type_top.addWidget(btn_type_off)
        type_header_layout.addLayout(type_top)

        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity"))
        self._type_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._type_opacity_slider.setRange(0, 100)
        self._type_opacity_slider.setEnabled(False)
        self._type_opacity_slider.valueChanged.connect(self._on_header_type_opacity_changed)
        opacity_layout.addWidget(self._type_opacity_slider)
        type_header_layout.addLayout(opacity_layout)
        
        self._type_tab_layout.addWidget(self._type_header)

        self._type_scroll = QScrollArea()
        self._type_scroll.setWidgetResizable(True)
        self._type_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._type_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self._type_container = QWidget()
        self._type_container.setBackgroundRole(QPalette.ColorRole.Base)
        self._type_container.setAutoFillBackground(True)
        self._type_layout = QVBoxLayout(self._type_container)
        self._type_layout.setContentsMargins(6, 6, 6, 6)
        self._type_layout.setSpacing(6)
        self._type_scroll.setWidget(self._type_container)
        
        self._type_tab_layout.addWidget(self._type_scroll)
        self._tabs.addTab(self._type_tab, self._spacer_icon, "Types")
        self._type_layout.addStretch()

        self._row_widgets: list[QWidget] = []

        # Debounced rebuild timer to prevent UI thrashing during batch updates
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(50)
        self._rebuild_timer.timeout.connect(self._rebuild)

        # React to model changes
        self._model.modelReset.connect(self._rebuild_timer.start)
        self._model.rowsInserted.connect(self._rebuild_timer.start)
        self._model.rowsRemoved.connect(self._rebuild_timer.start)
        self._model.dataChanged.connect(self._on_data_changed)

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
            if item.widget(): item.widget().deleteLater()

        while self._mask_layout.count():
            item = self._mask_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        while self._cell_layout.count():
            item = self._cell_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        while self._type_layout.count():
            item = self._type_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        for row in range(self._model.rowCount()):
            ch = self._model.channel(row)
            widget = self._make_row(row, ch)
            if ch.is_type_mask:
                self._type_layout.addWidget(widget)
            elif ch.is_cell_mask:
                self._cell_layout.addWidget(widget)
            elif ch.is_mask:
                self._mask_layout.addWidget(widget)
            else:
                self._channel_layout.addWidget(widget)
            self._row_widgets.append(widget)

        self._channel_layout.addStretch()
        self._mask_layout.addStretch()
        self._cell_layout.addStretch()
        self._type_layout.addStretch()

    def _make_row(self, row: int, ch: Channel) -> QWidget:
        frame = ClickableFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Sunken)
        
        # Use the standard 'Window' background role for a native look
        frame.setBackgroundRole(QPalette.ColorRole.Window)
        frame.setAutoFillBackground(True)
        if ch.selected:
            pal = frame.palette()
            color = frame.style().standardPalette().color(QPalette.ColorRole.Window).darker(115)
            pal.setColor(QPalette.ColorRole.Window, color)
            frame.setPalette(pal)
        
        frame.clicked.connect(lambda r=row: self._select_row(r))

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Top line: eye | colour | name
        top = QHBoxLayout()
        top.setSpacing(6)

        # Eye toggle
        eye_btn = QPushButton()
        eye_btn.setFixedSize(24, 24)
        eye_btn.setCheckable(True)
        eye_btn.setChecked(ch.visible)
        
        if ch.is_mask or ch.is_cell_mask:
            eye_btn.setIcon(self._eye_open_icon if ch.visible else self._eye_closed_icon)
        else:
            eye_btn.setIcon(self._eye_open_icon if ch.visible else self._dash_icon)
            
        eye_btn.setIconSize(QSize(16, 16))
        eye_btn.clicked.connect(lambda checked, r=row: self._toggle_vis(r, checked))
        top.addWidget(eye_btn)

        # Contour toggle (only for masks)
        if ch.is_mask or ch.is_cell_mask:
            contour_btn = QPushButton()
            contour_btn.setFixedSize(24, 24)
            contour_btn.setCheckable(True)
            contour_btn.setChecked(ch.contour_visible)
            contour_btn.setIcon(self._contour_open_icon if ch.contour_visible else self._contour_closed_icon)
            contour_btn.setIconSize(QSize(16, 16))
            contour_btn.clicked.connect(lambda checked, r=row: self._toggle_contour_vis(r, checked))
            top.addWidget(contour_btn)

        # Colour swatch
        if not ch.is_cell_mask and not ch.is_mask:
            swatch = QPushButton()
            swatch.setFixedSize(20, 20)
            pixmap = QPixmap(16, 16)
            pixmap.fill(ch.color)
            swatch.setIcon(QIcon(pixmap))
            swatch.setIconSize(QSize(16, 16))
            swatch.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch.clicked.connect(lambda _, r=row, s=swatch: self._pick_color(r, s))
            top.addWidget(swatch)

        # Name
        name = QLabel(ch.name)
        name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top.addWidget(name)

        # Delete button for mask or processed channel
        if ch.is_mask or ch.is_processed:
            del_btn = QPushButton()
            del_btn.setIcon(self._delete_icon)
            del_btn.setIconSize(QSize(16, 16))
            del_btn.setFixedSize(24, 24)
            del_btn.setToolTip("Delete this channel")
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.clicked.connect(lambda _, r=row: self._model.remove_channel(r))
            top.addWidget(del_btn)

        layout.addLayout(top)

        # Control area (Window for intensity, Opacity for mask)
        if ch.is_cell_mask:
            pass # Use the global cell opacity slider
        elif ch.is_mask:
            # Opacity slider moved to header
            pass
        else:
            # Sliders moved to header
            pass

        return frame

    # ---- callbacks ----------------------------------------------------

    def _select_row(self, row: int):
        for i, frame in enumerate(self._row_widgets):
            is_target = (i == row)
            idx = self._model.index(i)
            was_selected = self._model.data(idx, ChannelListModel.SelectedRole)
            if was_selected != is_target:
                self._model.setData(idx, is_target, ChannelListModel.SelectedRole)
                pal = frame.palette()
                color = frame.style().standardPalette().color(QPalette.ColorRole.Window)
                if is_target:
                    color = color.darker(115)
                pal.setColor(QPalette.ColorRole.Window, color)
                frame.setPalette(pal)

        # Update header controls for the newly selected row
        ch = self._model.channel(row)
        
        # Reset all header context-aware controls
        self._sel_group.setEnabled(False)
        self._mask_opacity_slider.setEnabled(False)
        self._type_opacity_slider.setEnabled(False)
        
        if not ch.is_mask and not ch.is_cell_mask and not ch.is_type_mask:
            # Traditional Channel tab controls
            self._sel_group.setEnabled(True)
            self._alpha_slider.blockSignals(True)
            self._alpha_slider.setValue(int(ch.alpha * 100))
            self._alpha_slider.blockSignals(False)
            self._limits_slider.blockSignals(True)
            self._limits_slider.set_range(ch.range_min, ch.range_max)
            self._limits_slider.blockSignals(False)
            
        elif ch.is_mask:
            # Mask tab controls
            self._mask_opacity_slider.setEnabled(True)
            self._mask_opacity_slider.blockSignals(True)
            self._mask_opacity_slider.setValue(int(ch.range_max * 100))
            self._mask_opacity_slider.blockSignals(False)

        elif ch.is_cell_mask:
            # Cell tab controls
            self._cell_opacity_slider.setEnabled(True)
            self._cell_opacity_slider.blockSignals(True)
            self._cell_opacity_slider.setValue(int(ch.range_max * 100))
            self._cell_opacity_slider.blockSignals(False)

        elif ch.is_type_mask:
            # Type tab controls
            self._type_opacity_slider.setEnabled(True)
            self._type_opacity_slider.blockSignals(True)
            self._type_opacity_slider.setValue(int(ch.range_max * 100))
            self._type_opacity_slider.blockSignals(False)

    def _on_data_changed(self, top_left, bottom_right, roles):
        row = top_left.row()
        if ChannelListModel.VisibleRole in roles or ChannelListModel.ContourVisibleRole in roles or ChannelListModel.SelectedRole in roles:
            if 0 <= row < len(self._row_widgets):
                ch = self._model.channel(row)
                frame = self._row_widgets[row]
                
                if ChannelListModel.SelectedRole in roles:
                    pal = frame.palette()
                    color = frame.style().standardPalette().color(QPalette.ColorRole.Window)
                    if ch.selected:
                        color = color.darker(115)
                    pal.setColor(QPalette.ColorRole.Window, color)
                    frame.setPalette(pal)
                    
                try:
                    top_layout = frame.layout().itemAt(0).layout()
                    
                    # 1. Update Eye Toggle
                    eye_btn = top_layout.itemAt(0).widget()
                    if isinstance(eye_btn, QPushButton):
                        eye_btn.setChecked(ch.visible)
                        if ch.is_mask or ch.is_cell_mask:
                            eye_btn.setIcon(self._eye_open_icon if ch.visible else self._eye_closed_icon)
                        else:
                            eye_btn.setIcon(self._eye_open_icon if ch.visible else self._dash_icon)
                    
                    # 2. Update Contour Toggle
                    if ch.is_mask or ch.is_cell_mask:
                        contour_btn = top_layout.itemAt(1).widget()
                        if isinstance(contour_btn, QPushButton):
                            contour_btn.setChecked(ch.contour_visible)
                            contour_btn.setIcon(self._contour_open_icon if ch.contour_visible else self._contour_closed_icon)
                except Exception:
                    pass

    def _toggle_vis(self, row: int, checked: bool):
        idx = self._model.index(row)
        self._model.setData(idx, checked, ChannelListModel.VisibleRole)

    def _toggle_contour_vis(self, row: int, checked: bool):
        idx = self._model.index(row)
        self._model.setData(idx, checked, ChannelListModel.ContourVisibleRole)

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
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            swatch.setIcon(QIcon(pixmap))

    def _on_brightness_changed(self, val: int):
        self._model.brightness = val / 100.0

    def _on_header_alpha_changed(self, val: int):
        ch = self._model.selected_channel()
        if ch:
            idx = self._model.index(self._model._channels.index(ch))
            self._model.setData(idx, val/100.0, ChannelListModel.AlphaRole)

    def _on_header_limits_changed(self, mn: float, mx: float):
        ch = self._model.selected_channel()
        if ch:
            idx = self._model.index(self._model._channels.index(ch))
            self._model.setData(idx, mn, ChannelListModel.RangeMinRole)
            self._model.setData(idx, mx, ChannelListModel.RangeMaxRole)

    def _on_header_mask_opacity_changed(self, val: int):
        ch = self._model.selected_channel()
        if ch:
            idx = self._model.index(self._model._channels.index(ch))
            self._model.setData(idx, val/100.0, ChannelListModel.RangeMaxRole)

    def _on_header_cell_opacity_changed(self, val: int):
        ch = self._model.selected_channel()
        if ch:
            idx = self._model.index(self._model._channels.index(ch))
            self._model.setData(idx, val/100.0, ChannelListModel.RangeMaxRole)

    def _on_header_type_opacity_changed(self, val: int):
        ch = self._model.selected_channel()
        if ch:
            idx = self._model.index(self._model._channels.index(ch))
            self._model.setData(idx, val/100.0, ChannelListModel.RangeMaxRole)
