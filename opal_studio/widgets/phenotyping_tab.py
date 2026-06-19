from __future__ import annotations

from PySide6.QtCore import Qt, Slot, QPoint, QRect
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
        self._table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(True)
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
        self._table.horizontalHeader().sectionDoubleClicked.connect(self._rename_cell_type)
        self._table.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.horizontalHeader().customContextMenuRequested.connect(self._on_header_context_menu)
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

    @Slot(int)
    def _rename_cell_type(self, logical_index: int):
        from PySide6.QtWidgets import QLineEdit
        if logical_index < 0 or logical_index >= len(self._cell_types):
            return
            
        header = self._table.horizontalHeader()
        viewport = header.viewport()
        
        rect = QRect(
            header.sectionViewportPosition(logical_index),
            0,
            header.sectionSize(logical_index),
            header.height()
        )
        
        old_name = self._cell_types[logical_index]
        
        edit = QLineEdit(viewport)
        edit.setStyleSheet("QLineEdit { border: 2px solid #0078d7; padding: 2px; background: white; color: black; }")
        edit.setGeometry(rect)
        edit.setText(old_name)
        edit.selectAll()
        edit.setFocus()
        
        def finish_edit():
            new_name = edit.text().strip()
            if new_name and new_name != old_name and new_name not in self._cell_types:
                self._cell_types[logical_index] = new_name
                # Update state dictionary keys
                new_states = {}
                for (ch, ct), val in self._cell_states.items():
                    if ct == old_name:
                        new_states[(ch, new_name)] = val
                    else:
                        new_states[(ch, ct)] = val
                self._cell_states = new_states
                self._refresh_table_ui()
            edit.deleteLater()
            
        edit.editingFinished.connect(finish_edit)
        edit.show()

    @Slot(QPoint)
    def _on_header_context_menu(self, pos: QPoint):
        header = self._table.horizontalHeader()
        logical_index = header.logicalIndexAt(pos)
        if logical_index < 0 or logical_index >= len(self._cell_types):
            return
            
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction
        
        menu = QMenu(self)
        delete_action = menu.addAction("Delete Cell Type")
        
        action = menu.exec(header.mapToGlobal(pos))
        if action == delete_action:
            cell_type = self._cell_types[logical_index]
            self._cell_types.pop(logical_index)
            # Remove from cell states dictionary
            keys_to_remove = [k for k in self._cell_states.keys() if k[1] == cell_type]
            for k in keys_to_remove:
                del self._cell_states[k]
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

    def save_to_csv(self, path):
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header: Marker, Type1, Type2, ...
            header = ["Marker"] + self._cell_types
            writer.writerow(header)
            
            for ch_name in self._channel_names:
                row = [ch_name]
                for c_type in self._cell_types:
                    state = self._cell_states.get((ch_name, c_type), 0)
                    # Use symbols: + for Pos, - for Neg, empty for None
                    val = ""
                    if state == 1: val = "+"
                    elif state == 2: val = "-"
                    row.append(val)
                writer.writerow(row)

    def load_from_csv(self, path):
        import csv
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            if not header or header[0] != "Marker":
                raise ValueError("Invalid CSV format. First column must be 'Marker'.")
            
            new_types = header[1:]
            
            # Read all states from CSV into a temporary map
            csv_states = {}
            for row in reader:
                if not row: continue
                ch_name = row[0]
                for i, val in enumerate(row[1:]):
                    if i >= len(new_types): break
                    c_type = new_types[i]
                    state = 0
                    if val == "+": state = 1
                    elif val == "-": state = 2
                    csv_states[(ch_name, c_type)] = state
            
            # Update columns
            self._cell_types = new_types
            
            # Update states only for existing markers (already in self._channel_names)
            # and reset others to 0
            for ch_name in self._channel_names:
                for c_type in self._cell_types:
                    self._cell_states[(ch_name, c_type)] = csv_states.get((ch_name, c_type), 0)
            
            self._refresh_table_ui()

    def get_phenotype_definitions(self) -> dict:
        """
        Returns a mapping from cell_type -> dict(marker_name -> state).
        States are 1 for Pos, 2 for Neg. 0 is ignored.
        """
        definitions = {}
        for c_type in self._cell_types:
            criteria = {}
            for ch_name in self._channel_names:
                state = self._cell_states.get((ch_name, c_type), 0)
                if state != 0:
                    criteria[ch_name] = state
            definitions[c_type] = criteria
        return definitions

    def export_state(self) -> dict:
        """Serialise the phenotyping table to a JSON-friendly dict.

        Only non-empty (Pos/Neg) cells are stored; markers are kept by name so
        the table survives even if the current channel set differs on reload.
        """
        return {
            "cell_types": list(self._cell_types),
            "states": [[ch, ct, int(st)]
                       for (ch, ct), st in self._cell_states.items() if st],
        }

    def import_state(self, state: dict) -> None:
        """Restore the phenotyping table from :meth:`export_state` output."""
        self._cell_types = list(state.get("cell_types", []))
        self._cell_states = {}
        for entry in state.get("states", []):
            try:
                ch, ct, st = entry
            except (TypeError, ValueError):
                continue
            self._cell_states[(ch, ct)] = int(st)
        self._refresh_rows()

    def clear(self):
        """Reset all phenotyping definitions and states."""
        self._cell_states = {}
        self._cell_types = []
        self._refresh_rows()

