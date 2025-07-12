import numpy as np
import cv2
from effects.base_effect import BaseEffect
import mediapipe as mp
import imageio


class Effect21(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.speed = 10

        self._settings_dict = {
            "speed": f"{self.speed}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict = None):
        self.speed = int(settings_dict["speed"])
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.model = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.is_ready = True
        gif = imageio.mimread("v.gif")
        self.back_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
        self.i = 0
        self.b_len = len(self.back_imgs)

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img

        if self.i // self.speed == self.b_len:
            self.i = 0

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        bg_image = np.zeros(img.shape, dtype=np.uint8)
        # bg_image[:] = (0, 0, 0)
        output_image = np.where(condition, img, bg_image).astype(np.uint8)

        output_image = cv2.resize(output_image, (303, 385))

        img = cv2.addWeighted(self.back_imgs[self.i // self.speed], 0.9, output_image, 2.5, 1)
        img = cv2.resize(img, (640, 480))
        self.i += 1
        return img
