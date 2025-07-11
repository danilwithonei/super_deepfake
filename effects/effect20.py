import mediapipe as mp
import numpy as np
import cv2
import random
from numpy import ndarray
from effects.base_effect import BaseEffect


def is_face_visible_2d(face):
    v1 = face[1] - face[0]
    v2 = face[2] - face[0]
    normal = np.cross(v1, v2)
    return normal > 0


class Cube:
    def __init__(self, center: tuple[int, int, int], width: int, img: np.ndarray):
        self.center = center
        self.width = width
        half_width = width / 2
        self.img = img

        self.vertices = np.array(
            [
                [center[0] + half_width, center[1] + half_width, center[2] + half_width],
                [center[0] + half_width, center[1] + half_width, center[2] - half_width],
                [center[0] + half_width, center[1] - half_width, center[2] + half_width],
                [center[0] + half_width, center[1] - half_width, center[2] - half_width],
                [center[0] - half_width, center[1] + half_width, center[2] + half_width],
                [center[0] - half_width, center[1] + half_width, center[2] - half_width],
                [center[0] - half_width, center[1] - half_width, center[2] + half_width],
                [center[0] - half_width, center[1] - half_width, center[2] - half_width],
            ]
        )

        self.current_rotation = np.array([0.0, 0.0, 0.0])
        self.target_rotation = np.array([0.0, 0.0, 0.0])
        self.rotation_speed = 1
        self.change_direction = True

    @property
    def faces_proj(self):
        vertices = self.project_to_2d()
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[2], vertices[3], vertices[6], vertices[7]],
            [vertices[5], vertices[7], vertices[1], vertices[3]],
            [vertices[4], vertices[6], vertices[5], vertices[7]],
            [vertices[4], vertices[5], vertices[0], vertices[1]],
            [vertices[4], vertices[0], vertices[6], vertices[2]],
        ]
        return faces

    def rotate_cube(self, angle, axis):
        angle_rad = np.radians(angle)
        if axis == "x":
            rotation_matrix = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                    [0, np.sin(angle_rad), np.cos(angle_rad)],
                ]
            )
        elif axis == "y":
            rotation_matrix = np.array(
                [
                    [np.cos(angle_rad), 0, np.sin(angle_rad)],
                    [0, 1, 0],
                    [-np.sin(angle_rad), 0, np.cos(angle_rad)],
                ]
            )
        elif axis == "z":
            rotation_matrix = np.array(
                [
                    [np.cos(angle_rad), -np.sin(angle_rad), 0],
                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                    [0, 0, 1],
                ]
            )
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        self.vertices = self.vertices @ rotation_matrix.T

    def project_to_2d(self):
        return self.vertices[:, :2]

    def fill_face(self, canvas: np.ndarray, pts2, img=None):
        h_face, _, _ = self.img.shape
        canvas_h, _, _ = canvas.shape
        pts1 = np.float32(
            [
                [0, 0],
                [h_face, 0],
                [0, h_face],
                [h_face, h_face],
            ]
        )
        h, _ = cv2.findHomography(pts1, pts2)
        if img is None:
            r = cv2.warpPerspective(self.img.copy(), h, (canvas_h, canvas_h))
        else:
            r = cv2.warpPerspective(img.copy(), h, (canvas_h, canvas_h))
        points_for_mask = np.asarray([pts2[0], pts2[1], pts2[3], pts2[2]]).astype(np.int64)
        mask = cv2.fillPoly(np.zeros((canvas_h, canvas_h, 3)), [points_for_mask], (255, 255, 255))
        canvas = np.where(mask == [255, 255, 255], r, canvas)
        canvas = cv2.polylines(canvas, [points_for_mask], isClosed=True, color=(255, 255, 255), thickness=2)
        return canvas

    def draw(self, canvas: np.ndarray, img: np.ndarray = None):
        for face in self.faces_proj:
            points = np.asarray(face) + camera_x
            if is_face_visible_2d(points):
                canvas = self.fill_face(canvas, points.astype(np.float32), img)
        return canvas

    def update_rotation_random(self):
        if self.change_direction:
            axis = random.choice(["x", "y", "z"])
            direction = random.choice([-1, 1])

            if axis == "x":
                self.target_rotation[0] += self.rotation_speed * direction
            elif axis == "y":
                self.target_rotation[1] += self.rotation_speed * direction
            elif axis == "z":
                self.target_rotation[2] += self.rotation_speed * direction

            self.change_direction = False

        self.current_rotation += (self.target_rotation - self.current_rotation) * 0.1  # Коэффициент 0.1 для инерции

        self.rotate_cube(self.current_rotation[0], "x")
        self.rotate_cube(self.current_rotation[1], "y")
        self.rotate_cube(self.current_rotation[2], "z")

        if random.random() < 0.01:
            self.change_direction = True


camera_x, camera_y = 400, 400
canvas_h = 800
h = 200


class Effect20(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self.face_path = "faces/sasha/happy.png"
        self._settings_dict = {
            "face_path": f"{self.face_path}",
        }
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.face_path = settings_dict["face_path"]
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # self.canvas = np.zeros((canvas_h, canvas_h, 3), dtype=np.uint8)
        self.canvas = (np.random.random((canvas_h, canvas_h, 3))*255).astype(np.uint8)
        self._img = cv2.imread(self.face_path)
        self._img = cv2.resize(self._img, (h, h))

        self.cubes = [
            Cube((10, 10, 0), h, self._img.copy()),
            Cube((250, 0, 0), h, self._img.copy()),
            Cube((-250, 0, 0), h, self._img.copy()),
            
        ]
        self.is_ready = True

    def set_prikol_on_img(self, img: ndarray) -> ndarray:
        if not self.is_ready:
            return img
        face_img = None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                tlx = int(face_landmarks.landmark[54].x * img.shape[1])
                tly = int(face_landmarks.landmark[54].y * img.shape[0])

                drx = int(face_landmarks.landmark[288].x * img.shape[1])
                dry = int(face_landmarks.landmark[288].y * img.shape[0])

                face_img = img[tly:tly+(dry-tly),tlx:tlx+(drx-tlx)]
                face_img = cv2.resize(face_img,(h,h))
        # self.canvas.fill(0)
        self.canvas = (np.random.random((canvas_h, canvas_h, 3))*100).astype(np.uint8)

        for cube in self.cubes:
            cube.update_rotation_random()
            try:
                self.canvas = cube.draw(self.canvas, face_img)
            except:
                print("error!!")
                pass

        return self.canvas
