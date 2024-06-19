import cv2
import mediapipe as mp
import numpy as np

# import scipy


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def find_4_points(x1, y1, x2, y2, length):
    angle = np.arctan2(y2 - y1, x2 - x1)

    x_1 = int(x1 + length * np.cos(angle + np.pi / 2))
    y_1 = int(y1 + length * np.sin(angle + np.pi / 2))

    x_2 = int(x1 - length * np.cos(angle + np.pi / 2))
    y_2 = int(y1 - length * np.sin(angle + np.pi / 2))

    x_3 = int(x2 + length * np.cos(angle + np.pi / 2))
    y_3 = int(y2 + length * np.sin(angle + np.pi / 2))

    x_4 = int(x2 - length * np.cos(angle + np.pi / 2))
    y_4 = int(y2 - length * np.sin(angle + np.pi / 2))

    return [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]


def get_pts(results, landmark1, landmark2, h, w, l=20):
    first_landmark = [
        int(results.pose_landmarks.landmark[landmark1].x * w),
        int(results.pose_landmarks.landmark[landmark1].y * h),
    ]
    second_landmark = [
        int(results.pose_landmarks.landmark[landmark2].x * w),
        int(results.pose_landmarks.landmark[landmark2].y * h),
    ]
    points = find_4_points(*first_landmark, *second_landmark, l)
    return points


def paste_sweater_on_img(
    img,
    piece_of_sweater,
    person_points,
    sweater_points,
):
    pts1 = np.array(person_points)
    pts2 = np.array(sweater_points)

    h, _ = cv2.findHomography(pts2, pts1)
    r = cv2.warpPerspective(piece_of_sweater, h, (img.shape[1], img.shape[0]))
    r_gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(r_gray, 1, 255, cv2.THRESH_BINARY)
    # mask: np.ndarray = scipy.ndimage.binary_fill_holes(mask).astype(np.uint8) * 255

    _person_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    _person_img = cv2.add(_person_img, r)
    return _person_img


path_to_sweater_img = "./14.png"
path_to_video = "./nik.mp4"  # "../web_trainer/1.mp4"

# extract sweater
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

sweater_img = cv2.imread(path_to_sweater_img)
sweater_img_paste = cv2.resize(sweater_img, (100, 200))
landmarks = [
    (
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    ),
    (
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
    ),
    (
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
    ),
    (
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
    ),
    (
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
    ),
    (
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
    ),
    (
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ),
    (
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
    ),
]
pieces_of_sweater = []
sweater_points_list = []
body_lanmarks = [
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
]
body_pts = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    _, results = mediapipe_detection(sweater_img, pose)
    for landmark1, landmark2 in landmarks:
        sweater_points = get_pts(
            results,
            landmark1,
            landmark2,
            *sweater_img.shape[:-1],
        )
        pts_for_poly = np.array(
            [
                sweater_points[0],
                sweater_points[2],
                sweater_points[3],
                sweater_points[1],
            ]
        )
        mask = np.zeros(sweater_img.shape[:-1], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [pts_for_poly], 255)
        piece_of_sweater = cv2.bitwise_and(sweater_img, sweater_img, mask=mask)
        pieces_of_sweater.append(piece_of_sweater)
        sweater_points_list.append(sweater_points)

    for b_l in body_lanmarks:
        x = int(results.pose_landmarks.landmark[b_l].x * sweater_img.shape[1])
        y = int(results.pose_landmarks.landmark[b_l].y * sweater_img.shape[0])
        body_pts.append([x, y])

    pts_for_poly = np.array(
        [
            body_pts[0],
            body_pts[2],
            body_pts[3],
            body_pts[1],
        ]
    )
    mask = np.zeros(sweater_img.shape[:-1], dtype=np.uint8)
    mask = cv2.fillPoly(mask, [pts_for_poly], 255)
    body_sweater_img = cv2.bitwise_and(sweater_img, sweater_img, mask=mask)

cap = cv2.VideoCapture(path_to_video)
# cap.set(1, 500)

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter(
    "output.mp4", fourcc, 30, (600, 800)
)  # Ширина: w, Высота: 900 (800 + 100)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        try:
            _, img = cap.read()
            img = cv2.resize(img, (600, 800))
            _, results = mediapipe_detection(img, pose)
            person_body_pts = []
            for b_l in body_lanmarks:
                x = int(results.pose_landmarks.landmark[b_l].x * img.shape[1])
                y = int(results.pose_landmarks.landmark[b_l].y * img.shape[0])
                person_body_pts.append([x, y])

            img = paste_sweater_on_img(
                img,
                body_sweater_img,
                person_body_pts,
                body_pts,
            )
            for i, (landmark1, landmark2) in enumerate(landmarks):
                person_points = get_pts(
                    results,
                    landmark1,
                    landmark2,
                    *img.shape[:-1],
                )
                img = paste_sweater_on_img(
                    img,
                    pieces_of_sweater[i],
                    person_points,
                    sweater_points_list[i],
                )
            img[-200:, -100:] = sweater_img_paste
            cv2.imshow("ff", img)
            out.write(img)
            if cv2.waitKey(1) == ord("q"):
                break
        except:
            pass
    cap.release()
    out.release()
    cv2.destroyAllWindows()
