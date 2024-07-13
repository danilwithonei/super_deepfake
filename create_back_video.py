import mediapipe as mp
import cv2


path_to_video = "./back_videos/rock/1.mp4"
path_to_ann = "./back_videos/rock/2.txt"

cap = cv2.VideoCapture(path_to_video)

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as model:
    with open(path_to_ann, "w") as f:
        i = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model.process(img_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    x1 = int(face_landmarks.landmark[54].x * img.shape[1])
                    y1 = int(face_landmarks.landmark[54].y * img.shape[0])

                    x2 = int(face_landmarks.landmark[284].x * img.shape[1])
                    y2 = int(face_landmarks.landmark[284].y * img.shape[0])

                    x3 = int(face_landmarks.landmark[172].x * img.shape[1])
                    y3 = int(face_landmarks.landmark[136].y * img.shape[0])

                    x4 = int(face_landmarks.landmark[288].x * img.shape[1])
                    y4 = int(face_landmarks.landmark[365].y * img.shape[0])
            i += 1
            f.write(f"{i} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
            cv2.imshow("res", img)

            if cv2.waitKey(1) == ord("q"):
                break

cv2.destroyAllWindows()
