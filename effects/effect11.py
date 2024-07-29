import numpy as np
import cv2
import mediapipe as mp
from effects.base_effect import BaseEffect
from utils.utils import paste_piece_of_img



class Effect11(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.num_colors = 16
        self.h = 64

        self._settings_dict = {
            "h": f"{self.h}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.h = int(settings_dict["h"])

        self.mp_face_mesh = mp.solutions.face_mesh
        self.model = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        w = img.shape[1]
        h = img.shape[0]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                my_face_pts = [
                    [
                        int(face_landmarks.landmark[54].x * img.shape[1]),
                        int(face_landmarks.landmark[54].y * img.shape[0]),
                    ],
                    [
                        int(face_landmarks.landmark[284].x * img.shape[1]),
                        int(face_landmarks.landmark[284].y * img.shape[0]),
                    ],
                    [
                        int(face_landmarks.landmark[172].x * img.shape[1]),
                        int(face_landmarks.landmark[136].y * img.shape[0]),
                    ],
                    [
                        int(face_landmarks.landmark[288].x * img.shape[1]),
                        int(face_landmarks.landmark[365].y * img.shape[0]),
                    ],
                ]

                # apply mask
                a = np.array(
                    [
                        my_face_pts[0],
                        my_face_pts[2],
                        my_face_pts[3],
                        my_face_pts[1],
                    ]
                )
                canvas = np.zeros_like(img, dtype=np.uint8)
                mask_ = cv2.fillPoly(canvas, [a], (255, 255, 255))
                mask = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                only_face_img = cv2.bitwise_and(img, img, mask=mask)
                new_img = cv2.resize(
                    only_face_img, (self.h, self.h), interpolation=cv2.INTER_LINEAR
                )
                new_img = cv2.resize(new_img, (w, h), interpolation=cv2.INTER_NEAREST)
                img = paste_piece_of_img(img, new_img, my_face_pts, my_face_pts)

        return img
