import mediapipe as mp
import numpy as np
import cv2
import csv
import scipy

from utils.utils import calc_landmark_list, pre_process_landmark
from model import KeyPointClassifier
from numpy import ndarray
from effects.base_effect import BaseEffect


class Effect3(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.face_path = "faces/sasha/"
        self._settings_dict = {
            "face_path": f"{self.face_path}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.face_path = settings_dict["face_path"]
        self.mp_face_mesh = mp.solutions.face_mesh
        self.faces = {
            "Neutral": cv2.imread(self.face_path + "normal.png"),
            "Happy": cv2.imread(self.face_path + "happy.png"),
            "Sad": cv2.imread(self.face_path + "sad.png"),
            "Angry": cv2.imread(self.face_path + "sad.png"),
            "Surprise": cv2.imread(self.face_path + "surprise.png"),
        }
        with open(
            "model/keypoint_classifier/keypoint_classifier_label.csv",
            encoding="utf-8-sig",
        ) as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        self.keypoint_classifier = KeyPointClassifier()

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.is_ready = True

    def set_prikol_on_img(self, img: ndarray) -> ndarray:
        if not self.is_ready:
            return img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark_list = calc_landmark_list(img_rgb, face_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                facial_emotion_id = self.keypoint_classifier(
                    pre_processed_landmark_list
                )
                em = self.keypoint_classifier_labels[facial_emotion_id]

                face_img = self.faces[em].copy()
                pts1 = np.float32(
                    [
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
                )
                pts2 = np.float32(
                    [
                        [0, 0],
                        [face_img.shape[1], 0],
                        [0, face_img.shape[0]],
                        [face_img.shape[1], face_img.shape[0]],
                    ]
                )
                h, _ = cv2.findHomography(pts2, pts1)
                r = cv2.warpPerspective(face_img, h, (img.shape[1], img.shape[0]))

                mask = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                mask: np.ndarray = (
                    scipy.ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
                )

                background_with_mask = cv2.bitwise_and(
                    img, img, mask=cv2.bitwise_not(mask)
                )
                overlay_with_mask = cv2.bitwise_and(r, r, mask=mask)
                img = cv2.add(background_with_mask, overlay_with_mask)
        return img
