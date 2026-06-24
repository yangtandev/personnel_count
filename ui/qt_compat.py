import os
import shutil
import subprocess
from pathlib import Path


def force_pyqt_plugin_path():
    from PyQt5.QtCore import QLibraryInfo

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
        QLibraryInfo.PluginsPath
    )


def configure_runtime_environment():
    project_dir = Path(__file__).resolve().parents[1]
    cache_dir = project_dir / ".cache"
    matplotlib_dir = cache_dir / "matplotlib"
    yolo_dir = cache_dir / "yolo"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_dir))
    force_pyqt_plugin_path()
    if "QT_QPA_PLATFORM" not in os.environ and not _display_available():
        os.environ["QT_QPA_PLATFORM"] = "offscreen"


def _display_available():
    display = os.environ.get("DISPLAY")
    if not display:
        return False
    if shutil.which("xdpyinfo"):
        try:
            subprocess.run(
                ["xdpyinfo", "-display", display],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=True,
            )
            return True
        except Exception:
            return False
    if not display.startswith(":"):
        return True
    display_num = display[1:].split(".", 1)[0]
    if not display_num:
        return False
    return Path(f"/tmp/.X11-unix/X{display_num}").exists()
