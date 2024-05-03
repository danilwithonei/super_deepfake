import mediapipe as mp
import pyvirtualcam
import cv2
import numpy as np
import csv
import scipy
from model import KeyPointClassifier
from utils import calc_landmark_list, pre_process_landmark

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_path = "faces/sasha/"

faces = {
    "Neutral": cv2.imread(face_path + "normal.png"),
    "Happy": cv2.imread(face_path + "happy.png"),
    "Sad": cv2.imread(face_path + "sad.png"),
    "Angry": cv2.imread(face_path + "sad.png"),
    "Surprise": cv2.imread(face_path + "surprise.png"),
}

with open(
    "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
) as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

keypoint_classifier = KeyPointClassifier()

with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    landmark_list = calc_landmark_list(image_rgb, face_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
                    em = keypoint_classifier_labels[facial_emotion_id]

                    face_img = faces[em].copy()
                    pts1 = np.float32(
                        [
                            [
                                int(face_landmarks.landmark[54].x * image.shape[1]),
                                int(face_landmarks.landmark[54].y * image.shape[0]),
                            ],
                            [
                                int(face_landmarks.landmark[284].x * image.shape[1]),
                                int(face_landmarks.landmark[284].y * image.shape[0]),
                            ],
                            [
                                int(face_landmarks.landmark[172].x * image.shape[1]),
                                int(face_landmarks.landmark[136].y * image.shape[0]),
                            ],
                            [
                                int(face_landmarks.landmark[288].x * image.shape[1]),
                                int(face_landmarks.landmark[365].y * image.shape[0]),
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
                    r = cv2.warpPerspective(
                        face_img, h, (image.shape[1], image.shape[0])
                    )

                    mask = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                    mask: np.ndarray = (
                        scipy.ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
                    )

                    background_with_mask = cv2.bitwise_and(
                        image, image, mask=cv2.bitwise_not(mask)
                    )
                    overlay_with_mask = cv2.bitwise_and(r, r, mask=mask)
                    image = cv2.add(background_with_mask, overlay_with_mask)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cam.send(image)
            cam.sleep_until_next_frame()
            # cv2.imshow(":)", image)

            # if cv2.waitKey(5) == ord("q"):
            #     break

        cap.release()
        cv2.destroyAllWindows()
