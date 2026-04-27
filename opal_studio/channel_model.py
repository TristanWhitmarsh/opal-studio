"""
Channel data model — drives the left panel and the renderer.

Uses Qt's model/view architecture so the channel list widget can
bind directly to it and receive automatic updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor
import colorsys
import numpy as np


def generate_spaced_colors(target_count=100, min_dist=75):
    """
    Generates a set of visually distinct colors by finding the largest gaps
    in the hue space and subdividing them.
    
    New Constraint: The selected midpoint must be at least min_dist (hue space) 
    away from the previous color to prevent adjacent channels from appearing 
    too similar in sequence.
    """
    # Starting colors: Red, Green, Blue (RGB tuples)
    colors_rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    # We assume Saturation and Value are at 100% (1.0)
    saturation = 1.0
    value = 1.0
    
    while len(colors_rgb) < target_count:
        hues_list = []
        
        # 1. Convert all current RGB colors to Hue (0-255 scale)
        for r, g, b in colors_rgb:
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hues_list.append(h * 256.0)
            
        last_h = hues_list[-1]
        
        # 2. Sort the hues to find the gaps between adjacent colors
        sorted_hues = sorted(hues_list)
        
        # 3. Collect all gaps and their midpoints
        gaps = [] # list of (gap_size, midpoint)
        for i in range(len(sorted_hues)):
            h1 = sorted_hues[i]
            h2 = sorted_hues[(i + 1) % len(sorted_hues)]
            
            if i == len(sorted_hues) - 1:
                gap = (h2 + 256.0) - h1
            else:
                gap = h2 - h1
            
            midpoint = (h1 + (gap / 2.0)) % 256.0
            gaps.append((gap, midpoint))
            
        # 4. Sort gaps by size (largest first)
        gaps.sort(key=lambda x: x[0], reverse=True)
        
        # 5. Pick the largest gap that satisfies the distance constraint from the previous hue
        best_midpoint = gaps[0][1] # Fallback to the absolute largest gap
        for gap_size, midpoint in gaps:
            # Distance on a circle (0-256)
            diff = abs(midpoint - last_h)
            dist = min(diff, 256.0 - diff)
            
            if dist >= min_dist:
                best_midpoint = midpoint
                break
                
        # 6. Convert the chosen midpoint hue back to an RGB color
        r_float, g_float, b_float = colorsys.hsv_to_rgb(best_midpoint / 256.0, saturation, value)
        
        new_color = (
            int(round(r_float * 255)), 
            int(round(g_float * 255)), 
            int(round(b_float * 255))
        )
        colors_rgb.append(new_color)
        
    return colors_rgb


@dataclass
class Channel:
    """One image channel's display state."""
    name: str
    color: QColor
    visible: bool = True
    selected: bool = False
    range_min: float = 0.0   # normalised 0-1 — maps to alpha 0
    range_max: float = 1.0   # normalised 0-1 — maps to alpha 1
    data_min: float = 0.0    # actual intensity minimum in this channel
    data_max: float = 1.0    # actual intensity maximum in this channel
    index: int = 0           # channel index in the image
    is_mask: bool = False
    is_cell_mask: bool = False
    is_processed: bool = False
    processed_data: np.ndarray | None = None
    mask_data: np.ndarray | None = None
    contour_data: np.ndarray | None = None
    alpha: float = 1.0
    contour_visible: bool = False
    is_type_mask: bool = False
    source_marker: str = ""
    pos_lut: np.ndarray | None = None
    random_contour_colors: bool = True


class ChannelListModel(QAbstractListModel):
    """
    Qt list model wrapping a list of `Channel` objects.

    Emits ``channels_changed`` whenever any display property is altered
    so the renderer / canvas can update.
    """

    # Custom signal emitted on any channel-display change
    channels_changed = Signal()

    NameRole = Qt.UserRole + 1
    ColorRole = Qt.UserRole + 2
    VisibleRole = Qt.UserRole + 3
    RangeMinRole = Qt.UserRole + 4
    RangeMaxRole = Qt.UserRole + 5
    SelectedRole = Qt.UserRole + 6
    AlphaRole = Qt.UserRole + 7
    ContourVisibleRole = Qt.UserRole + 8
    TypeRole = Qt.UserRole + 9

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channels: List[Channel] = []
        self._brightness: float = 1.0
        self._cell_opacity: float = 1.0
        self._type_opacity: float = 1.0

    # ---- public helpers ------------------------------------------------

    def set_channels(self, channels: List[Channel]) -> None:
        self.beginResetModel()
        self._channels = list(channels)
        self.endResetModel()
        self.channels_changed.emit()

    @property
    def brightness(self) -> float:
        return self._brightness

    @brightness.setter
    def brightness(self, value: float):
        if self._brightness != value:
            self._brightness = value
            self.channels_changed.emit()

    @property
    def cell_opacity(self) -> float:
        return self._cell_opacity

    @cell_opacity.setter
    def cell_opacity(self, value: float):
        if self._cell_opacity != value:
            self._cell_opacity = value
            for ch in self._channels:
                if ch.is_cell_mask:
                    ch.range_max = value
            self.channels_changed.emit()

    @property
    def type_opacity(self) -> float:
        return self._type_opacity

    @type_opacity.setter
    def type_opacity(self, value: float):
        if self._type_opacity != value:
            self._type_opacity = value
            for ch in self._channels:
                if ch.is_type_mask:
                    ch.range_max = value
            self.channels_changed.emit()

    def set_all_visible(self, visible: bool, include_masks: bool = True):
        self.beginResetModel()
        for ch in self._channels:
            if not include_masks and (ch.is_mask or ch.is_cell_mask or ch.is_type_mask):
                continue
            ch.visible = visible
        self.endResetModel()
        self.channels_changed.emit()

    def set_category_visible(self, category: str, visible: bool):
        """category can be 'mask', 'cell', or 'type'."""
        self.beginResetModel()
        for ch in self._channels:
            if category == "mask" and ch.is_mask and not ch.is_cell_mask and not ch.is_type_mask:
                ch.visible = visible
            elif category == "cell" and ch.is_cell_mask:
                ch.visible = visible
            elif category == "type" and ch.is_type_mask:
                ch.visible = visible
        self.endResetModel()
        self.channels_changed.emit()

    def channel(self, row: int) -> Channel:
        return self._channels[row]

    def visible_channels(self) -> List[Channel]:
        return [c for c in self._channels if c.visible or c.contour_visible]

    def selected_channel(self) -> Channel | None:
        for c in self._channels:
            if c.selected:
                return c
        return None

    def add_channel(self, channel: Channel):
        if channel.is_cell_mask:
            channel.range_max = self._cell_opacity
            if channel.visible:
                for ch in self._channels:
                    if ch.is_cell_mask:
                        ch.visible = False
        
        if channel.is_type_mask:
            channel.range_max = self._type_opacity
                    
        row = len(self._channels)
        self.beginInsertRows(QModelIndex(), row, row)
        self._channels.append(channel)
        self.endInsertRows()
        self.channels_changed.emit()

    def get_unique_name(self, base_name: str, always_suffix: bool = False) -> str:
        """Find the next available name, optionally appending an incrementing number."""
        existing_names = {c.name for c in self._channels}
        
        if not always_suffix and base_name not in existing_names:
            return base_name
            
        i = 1
        while True:
            candidate = f"{base_name}{i}"
            if candidate not in existing_names:
                return candidate
            i += 1

    def remove_channel(self, row: int):
        """Remove a channel by row index and free its data from memory."""
        if row < 0 or row >= len(self._channels):
            return
        self.beginRemoveRows(QModelIndex(), row, row)
        ch = self._channels.pop(row)
        # Free heavy data
        ch.mask_data = None
        ch.contour_data = None
        ch.processed_data = None
        self.endRemoveRows()
        self.channels_changed.emit()

    # ---- Qt model interface -------------------------------------------

    def rowCount(self, parent=QModelIndex()):
        return len(self._channels)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        ch = self._channels[index.row()]
        if role == Qt.DisplayRole or role == self.NameRole:
            return ch.name
        if role == Qt.UserRole:
            return ch
        if role == self.ColorRole:
            return ch.color
        if role == self.VisibleRole:
            return ch.visible
        if role == self.RangeMinRole:
            return ch.range_min
        if role == self.RangeMaxRole:
            return ch.range_max
        if role == self.SelectedRole:
            return ch.selected
        if role == self.AlphaRole:
            return ch.alpha
        if role == self.ContourVisibleRole:
            return ch.contour_visible
        return None

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:
        if not index.isValid():
            return False
        ch = self._channels[index.row()]
        if role == self.VisibleRole:
            is_checked = bool(value)
            if ch.is_cell_mask and is_checked:
                for i, other in enumerate(self._channels):
                    if i != index.row() and other.is_cell_mask and other.visible:
                        other.visible = False
                        o_idx = self.index(i, 0)
                        self.dataChanged.emit(o_idx, o_idx, [self.VisibleRole])
            ch.visible = is_checked
        elif role == self.RangeMinRole:
            ch.range_min = float(value)
        elif role == self.RangeMaxRole:
            ch.range_max = float(value)
        elif role == self.SelectedRole:
            is_selected = bool(value)
            ch.selected = is_selected
            if is_selected:
                for i, other in enumerate(self._channels):
                    if i != index.row() and other.selected:
                        other.selected = False
                        o_idx = self.index(i, 0)
                        self.dataChanged.emit(o_idx, o_idx, [self.SelectedRole])
        elif role == self.ColorRole:
            ch.color = value
        elif role == self.AlphaRole:
            ch.alpha = float(value)
        elif role == self.ContourVisibleRole:
            ch.contour_visible = bool(value)
        else:
            return False
        self.dataChanged.emit(index, index, [role])
        self.channels_changed.emit()
        return True

    def flags(self, index):
        return super().flags(index) | Qt.ItemIsEditable

    def roleNames(self):
        return {
            self.NameRole: b"name",
            self.ColorRole: b"color",
            self.VisibleRole: b"visible",
            self.RangeMinRole: b"rangeMin",
            self.RangeMaxRole: b"rangeMax",
            self.SelectedRole: b"selected",
            self.AlphaRole: b"alpha",
            self.ContourVisibleRole: b"contourVisible",
        }
