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
        
        # Try to find icon and convert to .ico on Windows
        icon_path = ""
        try:
            import opal_studio
            pkg_dir = os.path.dirname(opal_studio.__file__)
            png_path = os.path.join(pkg_dir, "icon.png")
            ico_path = os.path.join(pkg_dir, "icon.ico")
            
            if os.path.exists(png_path):
                if platform.system() == "Windows" and not os.path.exists(ico_path):
                    try:
                        from PySide6.QtGui import QImage
                        img = QImage(png_path)
                        if not img.isNull():
                            img.save(ico_path, "ICO")
                    except Exception as e:
                        print(f"[opal-studio] Could not convert PNG to ICO: {e}")
                
                if platform.system() == "Windows" and os.path.exists(ico_path):
                    icon_path = ico_path
                else:
                    icon_path = png_path
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
                # Set working directory to project root so python can find the package if not installed
                pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                shortcut.WorkingDirectory = pkg_root
            else:
                shortcut.Arguments = ""
                shortcut.WorkingDirectory = os.path.expanduser("~")
            
            shortcut.TargetPath = exe_path
            if icon_path:
                shortcut.IconLocation = icon_path
            
            shortcut.save()
            print(f"[opal-studio] Windows Launcher created at: {path}")

        elif platform.system() == "Linux":
            home = os.path.expanduser("~")
            desktop_dir = os.path.join(home, "Desktop")
            os.makedirs(desktop_dir, exist_ok=True)
            desktop_path = os.path.join(desktop_dir, "OpalStudio.desktop")

            # Prepend CUDA lib64 to LD_LIBRARY_PATH if present on this machine so
            # TensorFlow (Mesmer) can find libcudart / libcublas / libcudnn etc.
            # os.path.isdir returns False when the path doesn't exist, making this
            # a safe no-op on desktops or clusters without a system CUDA install.
            cuda_lib = "/usr/local/cuda/lib64"
            if os.path.isdir(cuda_lib):
                cuda_export = f'export LD_LIBRARY_PATH="{cuda_lib}:$LD_LIBRARY_PATH"\n'
            else:
                cuda_export = ""

            # Always use the current Python interpreter directly rather than the
            # `opal-studio` console script, whose shebang may be hardcoded to the
            # original installer's home directory and inaccessible to other users.
            python = sys.executable

            # Run Opal Studio through a small launch script rather than inlining
            # the command in the .desktop Exec line.  The script keeps the
            # Terminal=true window open when the app exits with an error, so the
            # traceback stays on screen instead of the terminal closing instantly.
            script_dir = os.path.join(home, ".local", "share", "opal-studio")
            os.makedirs(script_dir, exist_ok=True)
            script_path = os.path.join(script_dir, "opal-studio-launch.sh")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(f"""#!/bin/bash
set -u

pause_on_error() {{
    status="$1"
    if [ "$status" -ne 0 ]; then
        echo
        echo "[opal-studio] Opal Studio exited with error code: $status"
        echo "[opal-studio] The terminal is kept open so you can read the error above."
        echo
        read -r -p "Press Enter to close this terminal..."
    fi
    exit "$status"
}}

{cuda_export}"{python}" -m opal_studio
pause_on_error "$?"
""")
            os.chmod(script_path, 0o755)

            icon_str = icon_path if icon_path else "utilities-terminal"

            with open(desktop_path, "w") as f:
                f.write(f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Opal Studio
Comment=Launch Opal Studio Viewer
Exec=bash -lc "{script_path}"
Icon={icon_str}
Terminal=true
""")
            os.chmod(desktop_path, 0o755)
            print(f"[opal-studio] Linux Launcher created at: {desktop_path}")
            print(f"[opal-studio] Launch script written to: {script_path}")

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
