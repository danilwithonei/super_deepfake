import mediapipe as mp
import cv2


path_to_video = "./back_videos/rock/1.mp4"
path_to_ann = "./back_videos/rock/1.txt"

cap = cv2.VideoCapture(path_to_video)

mp_face_det = mp.solutions.face_detection
with mp_face_det.FaceDetection() as model:
    with open(path_to_ann, "w") as f:
        i = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model.process(img_rgb)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape
                    x_f, y_f, w, h = (
                        int(bboxC.xmin * iw),
                        int(bboxC.ymin * ih),
                        int(bboxC.width * iw),
                        int(bboxC.height * ih),
                    )
            i += 1
            f.write(f"{i} {x_f} {y_f} {w} {h}\n")
            cv2.imshow("res", img)

            if cv2.waitKey(1) == ord("q"):
                break

cv2.destroyAllWindows()
