import sys
import cv2
import time
import pyvirtualcam

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
)

from effects.base_effect import BaseEffect
from effects.effect1 import Effect1
from effects.effect2 import Effect2
from effects.effect3 import Effect3
from effects.effect4 import Effect4


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.capture = cv2.VideoCapture(0)
        self.timer.start(1)
        self.cam = pyvirtualcam.Camera(width=640, height=480, fps=30)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SuPeR DeEpFaKe")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        cap_l = QHBoxLayout()
        self.video_label = QLabel("video", self)
        self.video_label.setScaledContents(True)
        cap_l.addWidget(self.video_label)

        effects_l = QVBoxLayout()
        effects_l.setAlignment(Qt.AlignmentFlag.AlignTop)

        effect_label = QLabel("Effect")
        effect_label.setFixedWidth(200)
        effects_l.addWidget(effect_label)

        self.combo_box = QComboBox(self)
        self.combo_box.setFixedWidth(200)
        self.combo_box.currentIndexChanged.connect(self.change_effect)
        effects_l.addWidget(self.combo_box)

        self.info_label = QLabel("Info:")
        self.info_label.setFixedWidth(200)
        effects_l.addWidget(self.info_label)

        self.settings = QGridLayout()
        self.settings_list: list[tuple[QLabel, QLineEdit]] = []
        labels = ["1:", "2:", "3:", "4:", "5:"]

        for row, label in enumerate(labels):
            self.settings_list.append((QLabel(label), QLineEdit()))
            self.settings.addWidget(self.settings_list[row][0], row, 0)
            self.settings.addWidget(self.settings_list[row][1], row, 1)
            self.settings_list[row][0].setVisible(False)
            self.settings_list[row][1].setVisible(False)
            self.settings_list[row][0].setFixedWidth(100)
            self.settings_list[row][1].setFixedWidth(100)

        effects_l.addLayout(self.settings)

        self.button_start = QPushButton("Start", self)
        self.button_start.setFixedWidth(200)
        self.button_start.clicked.connect(self.start_effect)
        effects_l.addWidget(self.button_start)

        self.effect: BaseEffect | None = None

        self.effect_dict: dict[str, BaseEffect] = {
            "DVD": Effect1,
            "Rayn Gosling": Effect2,
            "Deepfake": Effect3,
            "Enot": Effect4,
        }
        for led in self.effect_dict.keys():
            self.combo_box.addItem(led)

        cap_l.addLayout(effects_l)
        layout.addLayout(cap_l)
        self.show()

    def start_effect(self):
        settings_dict = {}
        for i, (k, v) in enumerate(self.effect._settings_dict.items()):
            settings_dict[k] = self.settings_list[i][1].text()

        self.effect.settings(settings_dict)

    def change_effect(self):
        for i in range(len(self.settings_list)):
            self.settings_list[i][0].setVisible(False)
            self.settings_list[i][1].setVisible(False)
        effiect_name = self.combo_box.currentText()
        self.effect = self.effect_dict[effiect_name]()
        for i, (k, v) in enumerate(self.effect._settings_dict.items()):
            self.settings_list[i][0].setText(k)
            self.settings_list[i][1].setText(v)
            self.settings_list[i][0].setVisible(True)
            self.settings_list[i][1].setVisible(True)

    def update_frame(self):
        ret, frame = self.capture.read()
        old_time = time.time()
        if ret:
            if self.effect is not None:
                frame = self.effect.set_prikol_on_img(frame)
            self.show_img(frame)
            self.send_to_cam(frame)
        fps = 1 / (time.time() - old_time)
        self.info_label.setText(f"FPS:{int(fps)}")

    def show_img(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def send_to_cam(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.cam.send(image)
        self.cam.sleep_until_next_frame()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())
