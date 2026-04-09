from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QBrush, QPen
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QLineEdit, QHeaderView
)

class PhenotypingTab(QWidget):
    """Tab that holds a table mapping channels/markers to cell types."""
    def __init__(self, channel_model, parent=None):
        super().__init__(parent)
        self._channel_model = channel_model
        
        # Dictionary from (channel_name, cell_type) -> int (0: empty, 1: Pos, 2: Neg)
        self._cell_states = {}
        self._cell_types = []
        self._channel_names = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Top controls
        top_layout = QHBoxLayout()
        self._cell_type_input = QLineEdit()
        self._cell_type_input.setPlaceholderText("Enter Cell Type Name...")
        self._cell_type_input.returnPressed.connect(self._add_cell_type)
        
        self._add_col_btn = QPushButton("Add Cell Type")
        self._add_col_btn.clicked.connect(self._add_cell_type)
        
        top_layout.addWidget(self._cell_type_input)
        top_layout.addWidget(self._add_col_btn)
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # Table
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus) 
        self._table.cellPressed.connect(self._on_cell_clicked)
        self._table.cellDoubleClicked.connect(self._on_cell_clicked)
        self._table.setShowGrid(True)

        # --- NEW STYLING CODE ---
        self._table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.horizontalHeader().setSectionsMovable(True)

        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #f0f0f0;
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
        # -------------------------

        layout.addWidget(self._table)
        
        # Connect to model to populate rows automatically
        self._channel_model.modelReset.connect(self._refresh_rows)
        self._channel_model.rowsInserted.connect(lambda: self._refresh_rows())
        self._refresh_rows()

        #header = self._table.horizontalHeader()
        #   header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

    @Slot()
    def _add_cell_type(self):
        cell_type = self._cell_type_input.text().strip()
        if not cell_type or cell_type in self._cell_types:
            return # Ignore empty or duplicates
        
        self._cell_types.append(cell_type)
        self._cell_type_input.clear()
        self._refresh_table_ui()

    @Slot()
    def _refresh_rows(self):
        # We only want physical channels as markers, exclude segmentation masks
        self._channel_names = []
        for i in range(self._channel_model.rowCount()):
            ch = self._channel_model.channel(i)
            if not ch.is_mask and not ch.is_cell_mask:
                self._channel_names.append(ch.name)
        
        self._refresh_table_ui()
        
    def _refresh_table_ui(self):
        self._table.setRowCount(len(self._channel_names))
        self._table.setColumnCount(len(self._cell_types))
        
        self._table.setVerticalHeaderLabels(self._channel_names)
        self._table.setHorizontalHeaderLabels(self._cell_types)
        
        # Repopulate interactive items
        for r, ch_name in enumerate(self._channel_names):
            for c, c_type in enumerate(self._cell_types):
                state = self._cell_states.get((ch_name, c_type), 0)
                item = QTableWidgetItem()
                self._update_item_appearance(item, state)
                self._table.setItem(r, c, item)
                
        self._table.resizeColumnsToContents()

    def _on_cell_clicked(self, row, col):
        ch_name = self._channel_names[row]
        c_type = self._cell_types[col]
        
        current_state = self._cell_states.get((ch_name, c_type), 0)
        # Cycle: 0 (Empty) -> 1 (Pos) -> 2 (Neg) -> 0 (Empty)
        next_state = (current_state + 1) % 3
        self._cell_states[(ch_name, c_type)] = next_state
        
        item = self._table.item(row, col)
        if item:
            self._update_item_appearance(item, next_state)

    def _update_item_appearance(self, item: QTableWidgetItem, state: int):
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        
        # Colors that look nice and clean
        if state == 0:
            item.setText("")
            item.setBackground(QBrush(Qt.GlobalColor.white))
            item.setForeground(QBrush(Qt.GlobalColor.black))
        elif state == 1:
            item.setText("Pos")
            item.setBackground(QBrush(QColor("#c8e6c9"))) # Muted green
            item.setForeground(QBrush(Qt.GlobalColor.black))
        elif state == 2:
            item.setText("Neg")
            item.setBackground(QBrush(QColor("#ffcdd2"))) # Muted red
            item.setForeground(QBrush(Qt.GlobalColor.black))
