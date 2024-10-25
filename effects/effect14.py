import numpy as np
import cv2
import mediapipe as mp
from effects.base_effect import BaseEffect
from skimage.transform import PiecewiseAffineTransform, warp


class FastPiecewiseAffineTransform(PiecewiseAffineTransform):
    def __call__(self, coords):
        coords = np.asarray(coords)
        simplex = self._tesselation.find_simplex(coords)
        affines = np.array(
            [self.affines[i].params for i in range(len(self._tesselation.simplices))]
        )[simplex]
        pts = np.c_[coords, np.ones((coords.shape[0], 1))]
        result = np.einsum("ij,ikj->ik", pts, affines)
        result[simplex == -1, :] = -1
        return result


def trans(src, dst, img, tform, shape):
    tform.estimate(src, dst)
    out = warp(img, tform, output_shape=shape)
    return out


class Effect14(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.face_path = "images/sasha.png"

        self._settings_dict = {
            "face_path": f"{self.face_path}",
        }
        self.is_ready = False

    def detection(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.process(image_rgb)
        if results.multi_face_landmarks:
            height, width, _ = image.shape
            points = np.array(
                [
                    [int(landmark.x * width), int(landmark.y * height)]
                    for i, landmark in enumerate(
                        results.multi_face_landmarks[0].landmark
                    )
                    if i in self.indices
                ]
            )
            return points
        else:
            return None

    def crop_face(self, image, landmarks):
        if landmarks is not None:
            x_min = int(np.min(landmarks[:, 0]))
            x_max = int(np.max(landmarks[:, 0]))
            y_min = int(np.min(landmarks[:, 1]))
            y_max = int(np.max(landmarks[:, 1]))
            padding = 0
            x_min = max(0, x_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(image.shape[0], y_max + padding)
            face_crop = image[y_min:y_max, x_min:x_max]
            relative_landmarks = (landmarks - np.array([x_min, y_min])).astype(int)
            return face_crop, relative_landmarks, y_min, y_max, x_min, x_max
        return None

    def settings(self, settings_dict: dict):
        self.face_path = settings_dict["face_path"]
        self.tform = FastPiecewiseAffineTransform()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.model = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.indices = [50, 280, 4, 168, 330, 111]
        for i, ii in self.mp_face_mesh.FACEMESH_CONTOURS:
            self.indices.append(i)
            self.indices.append(ii)

        self.target_img = cv2.imread(self.face_path)
        self.target_pts = self.detection(self.target_img)
        self.tar_face, self.target_face_pts, _, _, _, _ = self.crop_face(
            self.target_img,
            self.target_pts,
        )
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        new_face = None
        try:
            res = self.detection(img)

            my_face, my_face_pts, y_min, y_max, x_min, x_max = self.crop_face(img, res)
            new_face = (
                trans(
                    my_face_pts,
                    self.target_face_pts,
                    self.tar_face,
                    self.tform,
                    my_face.shape,
                )
                * 255
            )
            my_face_h, my_face_w, _ = my_face.shape
        except:
            pass
        if new_face is not None:
            face = np.where(new_face==[0,0,0],my_face,new_face)
            img[y_min:y_max, x_min:x_max] = face

        return img
