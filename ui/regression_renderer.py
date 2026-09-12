import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5 import QtGui, QtWidgets

from ui.status import user_status_text
from ui.window import PersonnelCountWindow


class RegressionWindowRenderer:
    """Render regression frames through the same widget used on site."""

    def __init__(self, config, camera_name, size=(1280, 820)):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        ui = config.get("ui", {})
        camera_names = dict(ui.get("camera_names") or {})
        if camera_name != "top":
            camera_names["top"] = camera_names.get(camera_name, camera_name)
        self.window = PersonnelCountWindow(
            title=ui.get("window_title", "人員停留數"),
            camera_names=camera_names,
            single_camera=True,
        )
        self.camera_name = "top"
        self.window.resize(*size)
        self.window.show()
        self.app.processEvents()

    def render(self, frame, count, status):
        self.window.set_count(count)
        self.window.set_camera_status(self.camera_name, user_status_text(status))
        self.window.set_frame(self.camera_name, frame)
        self.app.processEvents()
        image = self.window.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB888)
        pixels = image.bits()
        pixels.setsize(image.byteCount())
        rgb = np.frombuffer(pixels, np.uint8).reshape(
            image.height(), image.bytesPerLine() // 3, 3
        )[:, : image.width()]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self):
        self.window.close()
