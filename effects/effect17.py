import numpy as np
import cv2

from effects.base_effect import BaseEffect


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = 255
THICKNESS = 1

ASCII_CHARS = "@MW#$%*+E~-:. "


class Effect17(BaseEffect):
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
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        ascii_frame = np.zeros((self.output_size, self.output_size), dtype=np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
