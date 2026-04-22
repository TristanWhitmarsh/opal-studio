"""
Entry point for the Opal Studio application.

Run with:  python -m opal_studio
"""

import sys
import os
import platform
from pathlib import Path

# Handle launcher creation BEFORE any GUI code
if "--create-launcher" in sys.argv:
    try:
        print("[opal-studio] Creating launcher...")
        
        # Try to find icon
        icon_path = ""
        try:
            import opal_studio
            icon_path = os.path.join(os.path.dirname(opal_studio.__file__), "icon.png")
            if not os.path.exists(icon_path):
                icon_path = ""
        except ImportError:
            pass

        if platform.system() == "Windows":
            import win32com.client
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            path = os.path.join(desktop, 'Opal Studio.lnk')
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(path)
            
            import shutil
            exe_path = shutil.which("opal-studio")
            if not exe_path:
                exe_path = sys.executable
                shortcut.Arguments = "-m opal_studio"
            
            shortcut.Targetpath = exe_path
            shortcut.WorkingDirectory = os.path.expanduser("~")
            if icon_path:
                shortcut.IconLocation = icon_path
            
            shortcut.save()
            print(f"[opal-studio] Windows Launcher created at: {path}")

        elif platform.system() == "Linux":
            home = os.path.expanduser("~")
            desktop_path = os.path.join(home, "Desktop", "OpalStudio.desktop")
            
            import shutil
            exe_path = shutil.which("opal-studio")
            if not exe_path:
                exe_path = f"{sys.executable} -m opal_studio"

            icon_str = icon_path if icon_path else "utilities-terminal"
            
            with open(desktop_path, "w") as f:
                f.write(f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Opal Studio
Comment=Launch Opal Studio Viewer
Exec={exe_path}
Icon={icon_str}
Terminal=false
""")
            os.chmod(desktop_path, 0o755)
            print(f"[opal-studio] Linux Launcher created at: {desktop_path}")

        else:
            print(f"[opal-studio] Launcher creation is not fully supported on {platform.system()} yet.")
        
        sys.exit(0)
    except Exception as e:
        print(f"[opal-studio] ❌ Failed to create launcher: {e}")
        sys.exit(1)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QIcon
from opal_studio.main_window import MainWindow

# Resolve icon path relative to this file
_ROOT = Path(__file__).resolve().parent
_ICON = _ROOT / "icon.png"


def main():
    import traceback
    def exception_hook(exctype, value, tb):
        print("CRITICAL ERROR: Unhandled Exception")
        traceback.print_exception(exctype, value, tb)
        sys.exit(1)
    
    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Application icon
    if _ICON.exists():
        app.setWindowIcon(QIcon(str(_ICON)))

    # Use a clean sans-serif font
    font = QFont("Segoe UI", 10)  # Falls back to system default on Linux/macOS
    app.setFont(font)

    try:
        window = MainWindow()
        if _ICON.exists():
            window.setWindowIcon(QIcon(str(_ICON)))
        window.show()
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
