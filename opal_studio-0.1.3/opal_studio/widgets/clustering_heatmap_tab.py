"""
Clustering heatmap tab — displays per-cluster mean channel intensities
as a colour-coded table with editable cluster names.

Rows   = clusters (editable first column for the user-chosen cell-type name)
Columns = image channels (header labels drawn vertically)
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, Signal, Slot, QRect, QSize
from PySide6.QtGui import QColor, QBrush, QPainter, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QStyledItemDelegate,
    QStyleOptionHeader, QStyle, QLabel,
)


# ---------------------------------------------------------------------------
# Custom header view that draws labels vertically
# ---------------------------------------------------------------------------
class VerticalHeaderView(QHeaderView):
    """Horizontal header that draws its section labels rotated 90° CCW."""

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self._min_section_width = 28

    def sectionSizeHint(self, logical_index):
        return QSize(self._min_section_width, self._label_height())

    def minimumSectionSize(self):
        return self._min_section_width

    def _label_height(self):
        """Estimate height needed for the tallest label drawn vertically."""
        fm = QFontMetrics(self.font())
        model = self.model()
        if model is None:
            return 80
        max_w = 80
        for i in range(model.columnCount()):
            text = model.headerData(i, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            if text:
                max_w = max(max_w, fm.horizontalAdvance(str(text)) + 16)
        return min(max_w, 200)

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self._label_height())

    def paintSection(self, painter: QPainter, rect: QRect, logical_index: int):
        painter.save()
        # Draw background using default style
        opt = QStyleOptionHeader()
        self.initStyleOption(opt)
        opt.section = logical_index
        opt.rect = rect
        opt.text = ""
        self.style().drawControl(QStyle.ControlElement.CE_Header, opt, painter, self)

        # Draw text rotated
        text = self.model().headerData(logical_index, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        if text:
            painter.setClipping(False)
            painter.setFont(self.font())
            painter.setPen(Qt.GlobalColor.black)
            # Translate to the bottom-left of the section rect
            painter.translate(rect.left(), rect.bottom())
            painter.rotate(-90)
            
            fm = QFontMetrics(self.font())
            # Center the text vertically in the column width
            y = int(rect.width() / 2 + fm.ascent() / 2 - 1)
            x = 8
            painter.drawText(x, y, str(text))
        painter.restore()


# ---------------------------------------------------------------------------
# Heatmap table widget
# ---------------------------------------------------------------------------
class ClusteringHeatmapTab(QWidget):
    """Centre-panel tab showing a heatmap of cluster × channel mean intensities.

    Rows = image channels
    Columns = clusters (editable via double-clicking the horizontal header)
    """

    # Emitted when the user renames a cluster: (cluster_id, new_name)
    clusterRenamed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cluster_ids: list[int] = []
        self._channel_names: list[str] = []
        self._heatmap_data: np.ndarray | None = None  # shape (n_clusters, n_channels)
        self._cluster_names: dict[int, str] = {}  # cluster_id -> user-assigned name

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Info label
        self._info_label = QLabel("Run clustering to generate a heatmap.")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._info_label)

        # Table
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        self._table.setShowGrid(True)

        self._table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(True)
        
        self._table.horizontalHeader().sectionDoubleClicked.connect(self._rename_cluster)

        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #f5f5f5;
                gridline-color: #d0d0d0;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                padding: 4px;
            }
            QTableCornerButton::section {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
            }
        """)

        layout.addWidget(self._table)
        self._table.setVisible(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_heatmap(self, cluster_ids: list[int], channel_names: list[str],
                    heatmap_data: np.ndarray):
        """Populate the heatmap table.

        Parameters
        ----------
        cluster_ids : list[int]
            Unique cluster IDs.
        channel_names : list[str]
            Channel names.
        heatmap_data : ndarray, shape (len(cluster_ids), len(channel_names))
            Per-cluster per-channel mean intensity.
        """
        self._cluster_ids = list(cluster_ids)
        self._channel_names = list(channel_names)
        self._heatmap_data = heatmap_data.copy()

        # Initialise cluster names (preserve any previous user edits)
        for cid in self._cluster_ids:
            if cid not in self._cluster_names:
                self._cluster_names[cid] = f"Cluster {cid}"

        self._info_label.setVisible(False)
        self._table.setVisible(True)
        self._rebuild_table()

    def clear(self):
        """Reset the heatmap to the empty state."""
        self._cluster_ids = []
        self._channel_names = []
        self._heatmap_data = None
        self._cluster_names = {}
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self._table.setVisible(False)
        self._info_label.setVisible(True)

    def get_cluster_names(self) -> dict[int, str]:
        """Return a mapping of cluster_id -> user-assigned name."""
        return dict(self._cluster_names)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild_table(self):
        self._table.blockSignals(True)

        n_clusters = len(self._cluster_ids)
        n_channels = len(self._channel_names)

        self._table.setRowCount(n_channels)
        self._table.setColumnCount(n_clusters)
        
        row_labels = self._channel_names
        col_labels = [self._cluster_names.get(cid, f"Cluster {cid}") for cid in self._cluster_ids]
        
        self._table.setVerticalHeaderLabels(row_labels)
        self._table.setHorizontalHeaderLabels(col_labels)

        if n_clusters == 0:
            self._table.blockSignals(False)
            return

        # Per-channel min/max for colour normalisation (across all clusters)
        # heatmap_data is (n_clusters, n_channels)
        ch_min = self._heatmap_data.min(axis=0)
        ch_max = self._heatmap_data.max(axis=0)
        ch_range = ch_max - ch_min
        ch_range[ch_range == 0] = 1.0  # avoid divide-by-zero

        for r in range(n_channels):
            for c, cid in enumerate(self._cluster_ids):
                val = self._heatmap_data[c, r]
                norm = (val - ch_min[r]) / ch_range[r]  # 0..1
                color = self._value_to_color(norm)

                item = QTableWidgetItem(f"{val:.2f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(QBrush(color))

                # Use white text on dark backgrounds for readability
                luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
                text_color = QColor("#ffffff") if luminance < 140 else QColor("#000000")
                item.setForeground(QBrush(text_color))

                font = item.font()
                font.setPointSize(max(font.pointSize() - 1, 7))
                item.setFont(font)

                self._table.setItem(r, c, item)

        # Sizing
        self._table.resizeColumnsToContents()
        self._table.verticalHeader().setDefaultSectionSize(26)
        self._table.blockSignals(False)

    @staticmethod
    def _value_to_color(norm: float) -> QColor:
        """Map a normalised value [0, 1] to a blue → white → red colour ramp."""
        norm = max(0.0, min(1.0, norm))
        if norm < 0.5:
            # Blue to white
            t = norm * 2.0  # 0..1
            r = int(65 + t * (255 - 65))
            g = int(105 + t * (255 - 105))
            b = int(225 + t * (255 - 225))
        else:
            # White to red
            t = (norm - 0.5) * 2.0  # 0..1
            r = 255
            g = int(255 - t * (255 - 70))
            b = int(255 - t * (255 - 70))
        return QColor(r, g, b)

    @Slot(int)
    def _rename_cluster(self, logical_index: int):
        from PySide6.QtWidgets import QLineEdit
        if logical_index < 0 or logical_index >= len(self._cluster_ids):
            return
        
        header = self._table.horizontalHeader()
        viewport = header.viewport()
        
        rect = QRect(
            header.sectionViewportPosition(logical_index),
            0,
            header.sectionSize(logical_index),
            header.height()
        )
        
        cid = self._cluster_ids[logical_index]
        old_name = self._cluster_names.get(cid, f"Cluster {cid}")
        
        edit = QLineEdit(viewport)
        edit.setStyleSheet("QLineEdit { border: 2px solid #0078d7; padding: 2px; background: white; color: black; }")
        edit.setGeometry(rect)
        edit.setText(old_name)
        edit.selectAll()
        edit.setFocus()
        
        def finish_edit():
            new_name = edit.text().strip()
            if new_name and new_name != old_name:
                self._cluster_names[cid] = new_name
                self._rebuild_table()
                self.clusterRenamed.emit(cid, new_name)
            edit.deleteLater()
            
        edit.editingFinished.connect(finish_edit)
        edit.show()
