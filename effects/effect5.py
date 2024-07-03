import numpy as np
import cv2
import random
from effects.base_effect import BaseEffect


class Effect5(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.pad = 10
        self.set_pad = 10

        self._settings_dict = {
            "set_pad": f"{self.set_pad}",
        }
        self.is_ready = False
        self.i = 0

    def settings(self, settings_dict: dict):
        self.set_pad = int(settings_dict["set_pad"])
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        if self.i == 0:
            self.pad = random.randint(1, self.set_pad)
            self.i = 2
        self.i -= 1
        img_1 = img[:, :, 0]
        img_2 = img[:, :, 1]
        img_3 = img[:, :, 2]

        img_1[:, self.pad :] = img_1[:, : -self.pad]
        img_2[:, : -self.pad] = img_2[:, self.pad :]
        img_1 = img[:, :, 0]
        img_2 = img[:, :, 1]
        img_3 = img[:, :, 2]

        img_1[:, self.pad :] = img_1[:, : -self.pad]
        img_2[:, : -self.pad] = img_2[:, self.pad :]

        return img
