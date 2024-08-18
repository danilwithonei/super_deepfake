import numpy as np
import cv2

from effects.base_effect import BaseEffect

def paste_piece_of_img(
    img: np.ndarray,
    piece_of_img: np.ndarray,
    in_points: tuple[int, int, int, int],
    from_points: tuple[int, int, int, int],
    mask=None,
):
    pts1 = np.array(in_points)
    pts2 = np.array(from_points)

    # perspective transform moment
    h, _ = cv2.findHomography(pts2, pts1)
    res = cv2.warpPerspective(piece_of_img, h, (img.shape[1], img.shape[0]))

    # paste piece of image in another by mask
    if mask is None:
        res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(res_gray, 1, 255, cv2.THRESH_BINARY)
    else:
        mask = cv2.warpPerspective(mask, h, (img.shape[1], img.shape[0]))
    result_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    result_img = cv2.add(result_img, res)
    return result_img



class Effect12(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.ph = 48
        self.w = 320
        self.h = 240

        self.lt = 300, 104
        self.rt = 392, 160
        self.ld = 261, 175
        self.rd = 349, 232

        self._settings_dict = {
            "ph": f"{self.ph}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.h = int(settings_dict["ph"])
        self.back_img = cv2.imread("./backs/noki601.jpg")
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        res_img = cv2.resize(img, (self.ph, self.ph))
        res_img = cv2.resize(res_img, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gray_image = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)

        green_image = np.ones(
            (gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8
        )*10
        green_image[:, :, 1] = gray_image
        green_image[:, :, 0] = gray_image // 2
        
        from_points = [[0, 0], [0, self.h], [self.w, 0], [self.w, self.h]]
        to_points = [self.lt, self.ld, self.rt, self.rd]
        result = paste_piece_of_img(
            self.back_img, green_image, in_points=to_points, from_points=from_points
        )
        return result
