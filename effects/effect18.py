import numpy as np
import cv2
import mediapipe as mp

from effects.base_effect import BaseEffect


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = 255
THICKNESS = 1

ASCII_CHARS = "@MW#$%*+E~-:. "


class Effect18(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.grid_width = 64
        self.grid_height = 64
        self.output_size = 640
        self.block_size = self.output_size // self.grid_width

        self._settings_dict = {
            "grid_width": f"{self.grid_width}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.grid_width = int(settings_dict["grid_width"])
        self.mp_face_det = mp.solutions.face_detection
        self.model = self.mp_face_det.FaceDetection()
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        ascii_frame = np.zeros((self.output_size, self.output_size), dtype=np.uint8)
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
            face_img = img.copy()
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.grid_width, self.grid_height), interpolation=cv2.INTER_AREA)

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                brightness = resized[y, x]
                char_index = int(brightness / 255 * (len(ASCII_CHARS) - 1))
                char = ASCII_CHARS[char_index]

                px = x * self.block_size
                py = y * self.block_size + 10
                cv2.putText(ascii_frame, char, (px, py), FONT, FONT_SCALE, (int(brightness)), THICKNESS, cv2.LINE_AA)
        return ascii_frame
