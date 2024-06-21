import cv2
import numpy as np
import mediapipe as mp
from utils.utils import mediapipe_detection, get_poly_by_line, paste_piece_of_img
from numpy import ndarray
from effects.base_effect import BaseEffect


class Effect2(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.length = 20
        self.donar_img_path = "./donar_imgs/14.png"

        self._settings_dict = {
            "lenght": f"{self.length}",
            "donar_path": f"{self.donar_img_path}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):

        self.donar_img_path = settings_dict["donar_path"]
        self.length = int(settings_dict["lenght"])

        self.donar_img = cv2.imread(self.donar_img_path)
        mp_pose = mp.solutions.pose
        self.model = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.landmarks = [
            (
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            ),
            (
                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                mp_pose.PoseLandmark.RIGHT_WRIST.value,
            ),
            (
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
            ),
            (
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value,
            ),
            (
                mp_pose.PoseLandmark.RIGHT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_KNEE.value,
            ),
            (
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value,
            ),
            (
                mp_pose.PoseLandmark.RIGHT_KNEE.value,
                mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            ),
            (
                mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.LEFT_ANKLE.value,
            ),
        ]
        self.pieces_of_donar_img = []
        self.donar_points_list = []
        self.donar_body_points_list = []
        self.body_lanmarks = [
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
        ]
        self.donar_body_img = None

        _, results = mediapipe_detection(img=self.donar_img, model=self.model)
        for landmark1, landmark2 in self.landmarks:
            donar_points = self.get_pts(
                results,
                landmark1,
                landmark2,
                *self.donar_img.shape[:-1],
            )
            pts_for_poly = np.array(
                [
                    donar_points[0],
                    donar_points[2],
                    donar_points[3],
                    donar_points[1],
                ]
            )
            mask = np.zeros(self.donar_img.shape[:-1], dtype=np.uint8)
            mask = cv2.fillPoly(mask, [pts_for_poly], 255)
            piece_of_donar = cv2.bitwise_and(self.donar_img, self.donar_img, mask=mask)
            self.pieces_of_donar_img.append(piece_of_donar)
            self.donar_points_list.append(donar_points)

        for b_l in self.body_lanmarks:
            x = int(results.pose_landmarks.landmark[b_l].x * self.donar_img.shape[1])
            y = int(results.pose_landmarks.landmark[b_l].y * self.donar_img.shape[0])
            self.donar_body_points_list.append([x, y])

        pts_for_poly = np.array(
            [
                self.donar_body_points_list[0],
                self.donar_body_points_list[2],
                self.donar_body_points_list[3],
                self.donar_body_points_list[1],
            ]
        )
        mask = np.zeros(self.donar_img.shape[:-1], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [pts_for_poly], 255)
        self.donar_body_img = cv2.bitwise_and(self.donar_img, self.donar_img, mask=mask)

        self.is_ready = True

    def get_pts(
        self,
        results: dict,
        landmark1: int,
        landmark2: int,
        img_height: int,
        img_weight: int,
    ) -> list[tuple[int, int]]:
        first_landmark = [
            int(results.pose_landmarks.landmark[landmark1].x * img_weight),
            int(results.pose_landmarks.landmark[landmark1].y * img_height),
        ]
        second_landmark = [
            int(results.pose_landmarks.landmark[landmark2].x * img_weight),
            int(results.pose_landmarks.landmark[landmark2].y * img_height),
        ]
        points = get_poly_by_line(*first_landmark, *second_landmark, self.length)
        return points

    def set_prikol_on_img(self, img: ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        _, results = mediapipe_detection(img=img, model=self.model)
        if results.pose_landmarks is None:
            return img
        person_body_pts = []
        for b_l in self.body_lanmarks:
            x = int(results.pose_landmarks.landmark[b_l].x * img.shape[1])
            y = int(results.pose_landmarks.landmark[b_l].y * img.shape[0])
            person_body_pts.append([x, y])

        img = paste_piece_of_img(
            img=img,
            piece_of_img=self.donar_body_img,
            in_points=person_body_pts,
            from_points=self.donar_body_points_list,
        )
        for i, (landmark1, landmark2) in enumerate(self.landmarks):
            person_points = self.get_pts(
                results,
                landmark1,
                landmark2,
                *img.shape[:-1],
            )
            img = paste_piece_of_img(
                img=img,
                piece_of_img=self.pieces_of_donar_img[i],
                in_points=person_points,
                from_points=self.donar_points_list[i],
            )
        return img
