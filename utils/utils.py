import cv2
import mediapipe as mp
import numpy as np
import copy
import itertools


def mediapipe_detection(
    img:np.ndarray, model: mp.solutions.pose.Pose
) -> tuple[np.ndarray, dict]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, results


def get_poly_by_line(
    x1: int, y1: int, x2: int, y2: int, length: int
) -> list[tuple[int, int]]:
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


def paste_piece_of_img(
    img: np.ndarray,
    piece_of_img: np.ndarray,
    in_points: tuple[int, int, int, int],
    from_points: tuple[int, int, int, int],
    mask = None
):
    pts1 = np.array(in_points)
    pts2 = np.array(from_points)

    # perspective transform moment
    h, _ = cv2.findHomography(pts2, pts1)
    res = cv2.warpPerspective(piece_of_img, h, (img.shape[1], img.shape[0]))

    # paste piece of image in another by mask
    if mask is None:
        res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(res_gray, 1, 255, cv2.THRESH_BINARY)
    else:
        mask = cv2.warpPerspective(mask, h, (img.shape[1], img.shape[0]))
    result_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    result_img = cv2.add(result_img, res)
    return result_img


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
