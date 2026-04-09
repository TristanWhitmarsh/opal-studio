"""
Entry point for the Opal Studio application.

Run with:  python -m opal_studio
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QIcon
from opal_studio.main_window import MainWindow

# Resolve icon path relative to project root
_ROOT = Path(__file__).resolve().parent.parent
_ICON = _ROOT / "icon.png"


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Application icon
    if _ICON.exists():
        app.setWindowIcon(QIcon(str(_ICON)))

    # Use a clean sans-serif font
    font = QFont("Segoe UI", 10)  # Falls back to system default on Linux/macOS
    app.setFont(font)

    window = MainWindow()
    if _ICON.exists():
        window.setWindowIcon(QIcon(str(_ICON)))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
