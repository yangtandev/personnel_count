import cv2
from ui.qt_compat import configure_runtime_environment

configure_runtime_environment()
from PyQt5 import QtCore, QtGui, QtWidgets


class PersonnelCountWindow(QtWidgets.QWidget):
    def __init__(self, title="人員停留數", camera_names=None):
        super().__init__()
        camera_names = camera_names or {"top": "井上", "bottom": "井底"}
        self.setWindowTitle(title)
        self.camera_titles = {
            "top": camera_names.get("top", "井上"),
            "bottom": camera_names.get("bottom", "井底"),
        }
        self.camera_statuses = {"top": "啟動中", "bottom": "啟動中"}
        self.camera_frames = {}
        self.top_view = QtWidgets.QLabel("")
        self.bottom_view = QtWidgets.QLabel("")
        self.count_label = QtWidgets.QLabel("人員停留數：0")

        for view in (self.top_view, self.bottom_view):
            view.setAlignment(QtCore.Qt.AlignCenter)
            view.setMinimumSize(640, 360)
            view.setStyleSheet("background:#111;color:white;")

        self.count_label.setAlignment(QtCore.Qt.AlignCenter)
        self.count_label.setStyleSheet("font-weight:700;color:#111;")

        views = QtWidgets.QHBoxLayout()
        views.addWidget(self.top_view)
        views.addWidget(self.bottom_view)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(views, 1)
        layout.addWidget(self.count_label)
        self._update_font_sizes()

    def set_count(self, count):
        self.count_label.setText(f"人員停留數：{count}")

    def set_camera_status(self, camera_name, text):
        self.camera_statuses[camera_name] = text
        if camera_name in self.camera_frames:
            self._render_camera(camera_name)

    def set_frame(self, camera_name, frame):
        self.camera_frames[camera_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._render_camera(camera_name)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_font_sizes()
        for camera_name in self.camera_frames:
            self._render_camera(camera_name)

    def _update_font_sizes(self):
        h = max(1, self.height())
        self.bar_font = max(26, min(72, h // 18))
        count_font = max(42, min(96, h // 13))
        self.bar_height = max(58, min(116, h // 12))
        self.count_label.setStyleSheet(
            f"font-size:{count_font}px;font-weight:700;color:#111;"
        )

    def _render_camera(self, camera_name):
        label = self.top_view if camera_name == "top" else self.bottom_view
        frame = self.camera_frames[camera_name]
        canvas_size = label.size()
        canvas_w = max(1, canvas_size.width())
        canvas_h = max(1, canvas_size.height())
        top_bar = min(self.bar_height, canvas_h // 4)
        bottom_bar = min(self.bar_height, canvas_h // 4)
        image_area_h = max(1, canvas_h - top_bar - bottom_bar)

        h, w, ch = frame.shape
        image = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        source = QtGui.QPixmap.fromImage(image)
        scaled = source.scaled(canvas_w, image_area_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        canvas = QtGui.QPixmap(canvas_w, canvas_h)
        canvas.fill(QtGui.QColor("#050505"))
        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setPen(QtGui.QColor("white"))
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(self.bar_font)
        painter.setFont(font)

        img_x = (canvas_w - scaled.width()) // 2
        img_y = top_bar + (image_area_h - scaled.height()) // 2

        self._draw_centered_text(
            painter,
            QtCore.QRect(0, 0, canvas_w, img_y),
            self.camera_titles[camera_name],
        )

        painter.drawPixmap(img_x, img_y, scaled)

        bottom_y = img_y + scaled.height()
        self._draw_centered_text(
            painter,
            QtCore.QRect(0, bottom_y, canvas_w, canvas_h - bottom_y),
            self.camera_statuses[camera_name],
        )
        painter.end()
        label.setPixmap(canvas)

    def _draw_centered_text(self, painter, rect, text):
        metrics = QtGui.QFontMetrics(painter.font())
        text_width = metrics.horizontalAdvance(text)
        x = rect.x() + max(0, (rect.width() - text_width) // 2)
        y = rect.y() + (rect.height() - metrics.height()) // 2 + metrics.ascent()
        painter.drawText(x, y, text)
