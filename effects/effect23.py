import numpy as np
import cv2
import mediapipe as mp
from effects.base_effect import BaseEffect

class Effect23(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self._settings_dict = {
            "amplitude": "0.3",
            "frequency": "0.1",
            "radius_factor": "2.0",
        }
        self.is_ready = False
        self.left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

    def settings(self, settings_dict: dict):
        self.frame_count = 0
        self.mp_face_mesh = mp.solutions.face_mesh
        self.model = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.amplitude = float(settings_dict["amplitude"])
        self.frequency = float(settings_dict["frequency"])
        self.radius_factor = float(settings_dict["radius_factor"])
        self.is_ready = True

    def get_eye_center(self, landmarks, eye_indices, img_shape):
        h, w = img_shape[:2]
        points = []
        for idx in eye_indices:
            lm = landmarks.landmark[idx]
            points.append((lm.x * w, lm.y * h))
        return np.mean(points, axis=0)

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img

        self.frame_count += 1
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)

        if not results.multi_face_landmarks:
            return img

        try:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = img.shape[:2]
            
            # Создаем координатные сетки
            y_coords, x_coords = np.indices((h, w))
            # Инициализируем карты преобразования
            map_x = x_coords.astype(np.float32).copy()
            map_y = y_coords.astype(np.float32).copy()
            
            # Рассчитываем текущий масштаб пульсации
            scale_val = 1.0 + self.amplitude * np.sin(self.frame_count * self.frequency)
            
            # Обрабатываем оба глаза
            for eye_indices in [self.left_eye_indices, self.right_eye_indices]:
                # Получаем центр глаза
                center = self.get_eye_center(face_landmarks, eye_indices, img.shape)
                cx, cy = center
                
                # Рассчитываем радиус глаза
                points = []
                for idx in eye_indices:
                    lm = face_landmarks.landmark[idx]
                    points.append((lm.x * w, lm.y * h))
                points = np.array(points)
                eye_radius = np.max(np.sqrt((points[:,0]-cx)**2 + (points[:,1]-cy)**2))
                radius = eye_radius * self.radius_factor
                
                # Вычисляем расстояния до центра глаза (относительно исходных координат)
                dx = x_coords - cx
                dy = y_coords - cy
                distances = np.sqrt(dx**2 + dy**2)
                
                # Создаем маску для области глаза
                mask = distances < radius
                
                # Рассчитываем коэффициенты масштабирования
                scale_factors = scale_val * (1 - distances / radius) + 1 * (distances / radius)
                
                # Вычисляем новые координаты для этой области
                new_x = cx + dx * scale_factors
                new_y = cy + dy * scale_factors
                
                # Обновляем только пиксели в маске
                map_x = np.where(mask, new_x, map_x).astype(np.float32)
                map_y = np.where(mask, new_y, map_y).astype(np.float32)

            # Применяем преобразование
            # print(map_x.dtype,map_y.dtype)
            warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
            return warped.astype(np.uint8)

        except Exception as e:
            print(f"Effect error: {e}")
            return img