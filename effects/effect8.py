import numpy as np
import cv2
from effects.base_effect import BaseEffect
from collections import deque
import mediapipe as mp


class Effect8(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.maxlen = 10

        self._settings_dict = {
            "Jesuses": f"{self.maxlen}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.maxlen = int(settings_dict["Jesuses"])
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.model = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.images = deque(maxlen=self.maxlen)
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        bg_image = np.zeros(img.shape, dtype=np.uint8)
        bg_image[:] = (0, 0, 0)
        output_image = np.where(condition, img, bg_image)
        self.images.append(output_image)
        for img_ in self.images:
            img = cv2.addWeighted(img, 1, img_, 0.2, 1)

        return img
