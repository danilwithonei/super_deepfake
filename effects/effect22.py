import numpy as np
import cv2
import mediapipe as mp
from effects.base_effect import BaseEffect


class Effect22(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self._settings_dict = {
            "amplitude": "0.3",  # Амплитуда пульсации (0.1-0.5)
            "frequency": "0.1",  # Скорость пульсации (0.05-0.2)
            "radius_factor": "1.5",  # Размер области эффекта
        }
        self.is_ready = False

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

        self.face_indices = [4, 197, 195, 5, 6]  # Нос и центральные точки лица
        self.amplitude = float(settings_dict["amplitude"])
        self.frequency = float(settings_dict["frequency"])
        self.radius_factor = float(settings_dict["radius_factor"])

        self.is_ready = True

    def get_facial_center(self, landmarks, img_shape):
        h, w = img_shape[:2]
        points = []
        for idx in self.face_indices:
            lm = landmarks.landmark[idx]
            points.append((int(lm.x * w), int(lm.y * h)))
        return np.mean(points, axis=0, dtype=np.int32)

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img

        self.frame_count += 1
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)

        if not results.multi_face_landmarks:
            return img

        try:
            # Получаем центральную точку лица
            face_landmarks = results.multi_face_landmarks[0]
            center = self.get_facial_center(face_landmarks, img.shape)
            cx, cy = center

            # Рассчитываем текущий масштаб пульсации
            amplitude = self.amplitude
            scale = 1.0 + amplitude * np.sin(self.frame_count * self.frequency)

            # Рассчитываем радиус области эффекта
            h, w = img.shape[:2]
            radius = int(min(h, w) * self.radius_factor / 5)

            # Создаем координатную сетку вместо циклов
            y_coords, x_coords = np.indices((h, w))

            # Вычисляем расстояния от центра для всех точек сразу
            dx = x_coords - cx
            dy = y_coords - cy
            distances = np.sqrt(dx**2 + dy**2)

            # Создаем маску области внутри радиуса
            mask = (distances < radius).astype(np.float32)

            # Рассчитываем масштабный коэффициент для всех точек
            scale_factors = scale * (1 - distances / radius) + 1 * (distances / radius)

            # Вычисляем новые координаты
            new_x = np.where(mask, cx + dx * scale_factors, x_coords)
            new_y = np.where(mask, cy + dy * scale_factors, y_coords)

            # Преобразуем в формат для remap
            map_x = new_x.astype(np.float32)
            map_y = new_y.astype(np.float32)

            # Применяем трансформацию
            warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
            return warped.astype(np.uint8)

        except Exception as e:
            print(f"Effect error: {e}")
            return img
