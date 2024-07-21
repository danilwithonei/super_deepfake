import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from effects.base_effect import BaseEffect


# FIXME: pls
def draw_lines(points, image, size):
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]
        dist = np.linalg.norm(pt2 - pt1)

        if dist > 50:
            continue

        image = cv2.line(image, tuple(pt1), tuple(pt2), (255, 0, 255), size)

    return image


# FIXME : rewrite to numpy
def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5


class Effect10(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.size = 5
        

        self._settings_dict = {
            "size": f"{self.size}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.size = int(settings_dict["size"])

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.model = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.points = deque(maxlen=500)
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                thumb_tip = hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.THUMB_TIP
                ]
                index_tip = hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]

                index_x = int(index_tip.x * img.shape[1])
                index_y = int(index_tip.y * img.shape[0])

                distance = calculate_distance(thumb_tip, index_tip)

                if distance < 0.05:
                    self.points.append([index_x, index_y])
        if len(self.points) > 2:
            np_points = np.array(self.points)  # FIXME
            img = draw_lines(np_points, img, self.size)
        return img
