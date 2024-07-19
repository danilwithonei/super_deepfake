import numpy as np
import cv2
import random
import mediapipe as mp
from utils.utils import paste_piece_of_img
from effects.base_effect import BaseEffect
import mss


# 330 120    508 105
# 336 230    517 247


class Effect7(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.back_img_path = "./backs/image.png"
        self.face_size = 130
        self.pad = 18

        self._settings_dict = {
            "back_path": f"{self.back_img_path}",
        }
        self.scr = mss.mss()
        self.monitor = self.scr.monitors[1]
        self.is_ready = False
        self.mon_w = self.monitor["width"]
        self.mon_h = self.monitor["height"]

    def settings(self, settings_dict: dict):
        self.back_img_path = settings_dict["back_path"]
        self.back_img = cv2.imread(self.back_img_path)

        self.mp_face_det = mp.solutions.face_detection
        self.model = self.mp_face_det.FaceDetection()
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
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
            face_img = np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8)

        face_img = cv2.resize(face_img, (self.face_size, self.face_size))

        out = self.back_img.copy()
        out[
            120 : 120 + self.face_size - self.pad * 2,
            70 : 70 + self.face_size - self.pad * 2,
        ] = face_img[self.pad : -self.pad, self.pad : -self.pad]

        sceen = np.array(self.scr.grab(self.monitor))[:, :, :3]

        out = paste_piece_of_img(
            out.copy(),
            sceen.copy(),
            [
                [330, 120],
                [336, 230],
                [517, 247],
                [508, 105],
            ],
            [
                [0, 0],
                [0, self.mon_h],
                [self.mon_w, self.mon_h],
                [self.mon_w, 0],
            ],
        )

        out = cv2.resize(out, (640, 480))
        return out
