import mediapipe as mp
import pyvirtualcam
import cv2
import numpy as np
import scipy

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
face_close = cv2.imread("imgs/image_eye.png")
face_open = cv2.imread("imgs/image_eye_open.png")


with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mouth_open = (
                        face_landmarks.landmark[14].y - face_landmarks.landmark[13].y
                    )
                    if mouth_open > 0.005:
                        face_img = face_open.copy()
                    else:
                        face_img = face_close.copy()
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
                    # mask: np.ndarray = (
                    #     scipy.ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
                    # )

                    background_with_mask = cv2.bitwise_and(
                        image, image, mask=cv2.bitwise_not(mask)
                    )
                    overlay_with_mask = cv2.bitwise_and(r, r, mask=mask)
                    result = cv2.add(background_with_mask, overlay_with_mask)

            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            cam.send(result)
            cam.sleep_until_next_frame()
            # cv2.imshow(":)", result)

            # if cv2.waitKey(5) == ord("q"):
            #     break

        cap.release()
        cv2.destroyAllWindows()
