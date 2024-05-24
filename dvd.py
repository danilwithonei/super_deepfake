import numpy as np
import cv2
import mediapipe as mp
import pyvirtualcam
import random


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_face_det = mp.solutions.face_detection

screen_width = 640
screen_height = 480
screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

square_size = 150
x = 0
y = 0
speed_x = 5
speed_y = 5

_filter = np.zeros(shape=(square_size, square_size, 3), dtype=np.uint8)
ch = 0
face_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)
with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    with mp_face_det.FaceDetection() as face_det:
        while cap.isOpened():
            screen.fill(0)
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_det.process(image_rgb)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x_f, y_f, w, h = (
                        int(bboxC.xmin * iw),
                        int(bboxC.ymin * ih),
                        int(bboxC.width * iw),
                        int(bboxC.height * ih),
                    )
                face_img = image[y_f : y_f + h, x_f : x_f + w]
                if not len(face_img):
                    face_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)

            face_img = cv2.resize(face_img, (square_size, square_size))
            r = int(x / screen_width * 100)
            _filter[:, :, ch] = np.uint8(int(r))
            face_img = cv2.addWeighted(face_img, 0.5, _filter, 0.5, 0)

            screen[y : y + square_size, x : x + square_size] = face_img
            cv2.imshow("DVD", screen)

            x += speed_x
            y += speed_y

            if x <= 0 or x + square_size >= screen_width:
                speed_x = -speed_x
                ch = random.randint(0, 2)
            if y <= 0 or y + square_size >= screen_height:
                speed_y = -speed_y
                ch = random.randint(0, 2)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
