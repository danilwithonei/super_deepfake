import numpy as np
import cv2
import random
import mediapipe as mp
from effects.base_effect import BaseEffect


class Effect1(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.square_size = 128
        self.speed_x = 8
        self.speed_y = 8

        self._settings_dict = {
            "square_size": f"{self.square_size}",
            "speed_x": f"{self.speed_x}",
            "speed_y": f"{self.speed_y}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.square_size = int(settings_dict["square_size"])
        self.speed_x = int(settings_dict["speed_x"])
        self.speed_y = int(settings_dict["speed_y"])

        self.screen_width = 640
        self.screen_height = 480

        self.screen = np.zeros(
            (self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.x = 0
        self.y = 0
        self.ch = 0
        self._filter = np.zeros(
            shape=(self.square_size, self.square_size, 3), dtype=np.uint8
        )
        self.face_img = np.zeros(
            (self.square_size, self.square_size, 3), dtype=np.uint8
        )
        self.mp_face_det = mp.solutions.face_detection
        self.model = self.mp_face_det.FaceDetection()
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        self.screen.fill(0)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x_f, y_f, w, h = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
            face_img = img[y_f : y_f + h, x_f : x_f + w]
        else:
            face_img = np.zeros((self.square_size, self.square_size, 3), dtype=np.uint8)
        face_img = cv2.resize(face_img, (self.square_size, self.square_size))
        r = int(self.x / self.screen_width * 100)
        self._filter[:, :, self.ch] = np.uint8(int(r))
        face_img = cv2.addWeighted(face_img, 0.5, self._filter, 0.5, 0)

        self.screen[
            self.y : self.y + self.square_size, self.x : self.x + self.square_size
        ] = face_img
        self.screen = cv2.cvtColor(self.screen, cv2.COLOR_BGR2RGB)

        self.x += self.speed_x
        self.y += self.speed_y

        if self.x <= 0 or self.x + self.square_size >= self.screen_width:
            self.speed_x = -self.speed_x
            self.ch = random.randint(0, 2)
        if self.y <= 0 or self.y + self.square_size >= self.screen_height:
            self.speed_y = -self.speed_y
            self.ch = random.randint(0, 2)

        return self.screen
