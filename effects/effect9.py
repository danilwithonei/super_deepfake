import numpy as np
import cv2
import mediapipe as mp
from effects.base_effect import BaseEffect
from utils.utils import paste_piece_of_img


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
        with open(self.path_to_back + "/2.txt", "r") as f:
            self.anns = f.readlines()

        self.back_index = 0
        self.mp_face_mesh = mp.solutions.face_mesh
        self.model = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.is_ready = True

    def get_frame_by_index(self, i):
        self.back_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = self.back_cap.read()
        if ret:
            return frame

    def get_ann_by_index(self, i):
        for ann in self.anns:  # FIXME : added dict not list!!!
            ann_i, x1, y1, x2, y2, x3, y3, x4, y4 = ann.split(" ")
            if int(ann_i) == i:
                return (
                    [int(ann_i)],
                    [
                        [int(x1), int(y1)],
                        [int(x2), int(y2)],
                        [int(x3), int(y3)],
                        [int(x4), int(y4)],
                    ],
                )
        return ([-1], [0, 0, 0, 0, 0, 0, 0, 0])

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        back_img = self.get_frame_by_index(self.back_index)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                _ann_i, pts1 = self.get_ann_by_index(self.back_index)
                if _ann_i != [-1]:
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
                    canvas = np.zeros_like(img,dtype=np.uint8)
                    mask_ = cv2.fillPoly(canvas, [a], (255, 255, 255))
                    mask = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                    only_face_img = cv2.bitwise_and(
                        img, img, mask=mask
                    )

                    back_img = paste_piece_of_img(back_img, only_face_img, pts1, my_face_pts,mask)

        if self.back_index == len(self.anns) - 1:
            self.back_index = 0
        else:
            self.back_index += 1
        return back_img
