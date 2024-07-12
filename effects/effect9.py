import numpy as np
import cv2
import random
import mediapipe as mp
from effects.base_effect import BaseEffect


class Effect9(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.path_to_back = "./back_videos/rock/"

        self._settings_dict = {
            "video dir": f"{self.path_to_back}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.path_to_back = settings_dict["video dir"]
        self.back_cap = cv2.VideoCapture(self.path_to_back + "/1.mp4")
        with open(self.path_to_back + "/1.txt", "r") as f:
            self.anns = f.readlines()

        self.back_index = 0
        self.mp_face_det = mp.solutions.face_detection
        self.model = self.mp_face_det.FaceDetection()
        self.is_ready = True

    def get_frame_by_index(self, i):
        self.back_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = self.back_cap.read()
        if ret:
            return frame

    def get_ann_by_index(self, i):
        for ann in self.anns:
            ann_i, ann_x, ann_y, ann_w, ann_h = ann.split(" ")
            if int(ann_i) == i:
                return int(ann_i), int(ann_x), int(ann_y), int(ann_w), int(ann_h)
        return -1, 0, 0, 0, 0

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        back_img = self.get_frame_by_index(self.back_index)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x_f, y_f, w, h = (
                    max(1,int(bboxC.xmin * iw)),
                    max(1,int(bboxC.ymin * ih)),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
            face_img = img[y_f : y_f + h, x_f : x_f + w]
            ann_i, ann_x, ann_y, ann_w, ann_h = self.get_ann_by_index(self.back_index)
            if not ann_i == -1 :
                face_img = cv2.resize(face_img, (ann_w, ann_h))
                back_img[ann_y : ann_y + ann_h, ann_x : ann_x + ann_w] = face_img
        if self.back_index == len(self.anns)-1:
            self.back_index = 0
        else:
            self.back_index += 1
        return back_img
