import numpy as np
import cv2
import mediapipe as mp
from effects.base_effect import BaseEffect


class Effect6(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.cloud_4ch = cv2.imread("./images/pngwing.com.png", cv2.IMREAD_UNCHANGED)
        self.cloud_4ch = cv2.resize(self.cloud_4ch, (200, 200))
        self.cloud_mask = self.cloud_4ch[:, :, 3]
        self.cloud_img = self.cloud_4ch[:, :, :3]
        self.ones = np.ones_like(self.cloud_img).astype(np.uint8) * 255
        self.text = "Hello"

        self._settings_dict = {
            "text": f"{self.text}",
        }
        self.prev_x, self.prev_y = 0, 0
        self.alpha = 0.9
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.text = settings_dict["text"]
        self.mp_face_det = mp.solutions.face_detection
        self.model = self.mp_face_det.FaceDetection()
        self.ones.fill(255)
        self.ones = cv2.putText(
            self.ones, self.text, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0)
        )
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
                smooth_x = int(self.alpha * self.prev_x + (1 - self.alpha) * x_f)
                smooth_y = int(self.alpha * self.prev_y + (1 - self.alpha) * y_f)
                self.prev_x, self.prev_y = smooth_x, smooth_y
                try:
                    img_ = cv2.bitwise_and(
                        img[smooth_y - 200 : smooth_y, smooth_x - 200 : smooth_x],
                        self.ones.copy(),
                        dst=self.ones.copy(),
                        mask=cv2.bitwise_not(self.cloud_mask),
                    )
                    img[smooth_y - 200 : smooth_y, smooth_x - 200 : smooth_x] = img_
                except:
                    pass
        return img
