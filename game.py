import cv2
import mediapipe as mp
import numpy as np

# Инициализация Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)  # 0 - номер камеры (обычно встроенная камера)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при чтении видеопотока")
        break

    h, w, _ = frame.shape

    # Обнаружение ключевых точек лица
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Создание черного фона изображения
        black_bg = np.zeros((h, w, 3), dtype=np.uint8)

        # Получение координат контура лица
        face_contour = []
        for i, landmark in enumerate(face_landmarks.landmark):
            # if i in [21, 54, 67, 103, 109, 10, 338, 297, 332, 264, 251]:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            face_contour.append([x, y])
        face_contour = sorted(face_contour, key=lambda x: x[1])

        # Нарисовать контур лица на черном фоне
        # cv2.fillConvexPoly(black_bg, np.array(face_contour), (255, 255, 255), 1)
        cv2.drawContours(black_bg, [np.array(face_contour)], -1, (255, 255, 255), 5)

        # Вырезать лицо из исходного изображения
        face_crop = cv2.bitwise_and(frame, black_bg)

        # Отображение результата
        cv2.imshow("Face Cropped", face_crop)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Очистка
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
