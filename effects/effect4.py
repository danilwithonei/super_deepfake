from numpy import ndarray
from effects.base_effect import BaseEffect
import cv2
import time
import numpy as np
import math
import mediapipe as mp


class Effect4(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.angle_speed = 45
        self.motion_speed = 10
        self.size_speed = 10
        self._settings_dict = {
            "angle_speed": f"{self.angle_speed}",
            "motion_speed": f"{self.motion_speed}",
            "size_speed": f"{self.size_speed}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.angle_speed = int(settings_dict["angle_speed"])
        self.motion_speed = int(settings_dict["motion_speed"])
        self.size_speed = int(settings_dict["size_speed"])
        self.position_limits = [-100, -60]
        self.face = ((210, 100), (400, 350))
        self.enot_sizes = ((450, 450), (530, 530))
        self.face_size = (
            self.face[1][0] - self.face[0][0],
            self.face[1][1] - self.face[0][1],
        )

        self.amplitude = abs(self.position_limits[1] - self.position_limits[0]) / 2
        self.k = (self.position_limits[0] + self.position_limits[1]) / 2
        self.size_amplitudes = (
            abs(self.enot_sizes[0][0] - self.enot_sizes[1][0]) / 2,
            abs(self.enot_sizes[0][1] - self.enot_sizes[1][1]) / 2,
        )
        self.size_ks = (
            (self.enot_sizes[0][0] + self.enot_sizes[1][0]) / 2,
            (self.enot_sizes[0][1] + self.enot_sizes[1][1]) / 2,
        )

        self.enot_orig = cv2.imread("images/enot.png", -1)

        self.image_template = np.zeros((480, 640, 3), dtype=np.uint8)
        self.side_length = 50
        self.radius = 200

        self.h, self.w = self.image_template.shape[:2]
        self.center = (self.w // 2, self.h // 2)

        cv2.circle(self.image_template, self.center, self.radius, (255, 255, 255), -1)
        self.image_template_blured = cv2.blur(self.image_template, (15, 15))
        self.image_template_blured = cv2.blur(self.image_template_blured, (15, 15))

        self.angle = 0
        self.position = 0
        self.old_time = time.time()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_img = None
        self.is_ready = True

    def set_prikol_on_img(self, img: ndarray) -> ndarray:
        if not self.is_ready:
            return img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        rotated_image = np.copy(self.image_template_blured)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_img = img[
                    int(face_landmarks.landmark[54].y * img.shape[0]) : int(
                        face_landmarks.landmark[365].y * img.shape[0]
                    ),
                    int(face_landmarks.landmark[54].x * img.shape[1]) : int(
                        face_landmarks.landmark[288].x * img.shape[1]
                    ),
                ]
        else:
            return rotated_image

        if not (face_img is None):
            position = int(
                math.sin(time.time() * self.motion_speed) * self.amplitude + self.k
            )
            enot = self.enot_orig.copy()
            try:
                enot[
                    self.face[0][1] : self.face[1][1],
                    self.face[0][0] : self.face[1][0],
                    :-1,
                ] = cv2.resize(face_img, self.face_size)
            except:
                return rotated_image

            enot = cv2.resize(
                enot,
                (
                    int(
                        math.sin(time.time() * self.size_speed)
                        * self.size_amplitudes[0]
                        + self.size_ks[0]
                    ),
                    int(
                        math.sin(time.time() * self.size_speed)
                        * self.size_amplitudes[1]
                        + self.size_ks[1]
                    ),
                ),
            )

            part = rotated_image[
                self.center[1] + position : self.center[1] + enot.shape[0] + position,
                self.center[0]
                - enot.shape[1] // 2 : self.center[0]
                + enot.shape[1] // 2,
            ]

            mask_inv = cv2.bitwise_not(enot[: part.shape[0], : part.shape[1], -1])

            temp1 = cv2.bitwise_and(part, part, mask=mask_inv)
            temp2 = cv2.bitwise_and(
                enot[: part.shape[0], : part.shape[1], :-1],
                enot[: part.shape[0], : part.shape[1], :-1],
                mask=enot[: part.shape[0], : part.shape[1], -1],
            )

            rotated_image[
                self.center[1] + position : self.center[1] + enot.shape[0] + position,
                self.center[0]
                - enot.shape[1] // 2 : self.center[0]
                + enot.shape[1] // 2,
            ] = cv2.add(temp1, temp2)

            # Rotate the image
            rotation_matrix = cv2.getRotationMatrix2D(self.center, self.angle, 1)
            rotated_image = cv2.warpAffine(
                rotated_image, rotation_matrix, (self.w, self.h)
            )

            rotated_image = np.where(self.image_template == 0, 0, rotated_image)

            self.angle += (time.time() - self.old_time) * self.angle_speed
            self.old_time = time.time()
        return rotated_image
