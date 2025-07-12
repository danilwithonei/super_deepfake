import numpy as np
import cv2
import mediapipe as mp
from skimage.transform import PiecewiseAffineTransform, warp


class FastPiecewiseAffineTransform(PiecewiseAffineTransform):
    def __call__(self, coords):
        coords = np.asarray(coords)
        simplex = self._tesselation.find_simplex(coords)
        affines = np.array(
            [self.affines[i].params for i in range(len(self._tesselation.simplices))]
        )[simplex]
        pts = np.c_[coords, np.ones((coords.shape[0], 1))]
        result = np.einsum("ij,ikj->ik", pts, affines)
        result[simplex == -1, :] = -1
        return result


def trans(src, dst, img, tform, shape):
    tform.estimate(src, dst)
    out = warp(img, tform, output_shape=shape)
    return out


def detection(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    if results.multi_face_landmarks:
        height, width, _ = image.shape
        points = np.array(
            [
                [int(landmark.x * width), int(landmark.y * height)]
                for i, landmark in enumerate(results.multi_face_landmarks[0].landmark)
                if i in indices
            ]
        )
        return points
    else:
        return None


tform = FastPiecewiseAffineTransform()
mp_face_mesh = mp.solutions.face_mesh
model = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

meshes = [
    mp_face_mesh.FACEMESH_FACE_OVAL,
    mp_face_mesh.FACEMESH_LEFT_EYE,
    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    mp_face_mesh.FACEMESH_RIGHT_EYE,
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
    mp_face_mesh.FACEMESH_NOSE,
    mp_face_mesh.FACEMESH_LIPS,
]
indices = []
for m in meshes:
    for i, ii in m:
        indices.append(i)
        indices.append(ii)

path_to_img = "images/r.png"

orig_img = cv2.imread(path_to_img)
orig_img = cv2.resize(orig_img, (600, 600))
orig_pts = detection(orig_img)

pts = orig_pts.copy()
load_pts = np.load('face_schema.npy')

pts = (orig_pts - load_pts*600).astype(np.int16)
print(pts)
dragging_point = None


def mouse_callback(event, x, y, flags, param):
    global dragging_point
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(pts):
            if abs(pt[0] - x) < 5 and abs(pt[1] - y) < 5:
                dragging_point = i

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            pts[dragging_point] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None  #


cv2.namedWindow("s")
cv2.setMouseCallback("s", mouse_callback)

while True:

    res_img = orig_img.copy()
    res_img = trans(
        pts,
        orig_pts,
        orig_img,
        tform,
        orig_img.shape,
    )

    for pt in pts:
        res_img = cv2.circle(res_img, pt, 2, (0, 0, 255))

    delta = orig_pts - pts
    delta = delta.astype(np.float64) / 600
    # delta[:, 1] /= 600

    # for pt in orig_pts:
    #     res_img = cv2.circle(res_img, pt, 2, (0, 255, 0))

    cv2.imshow("s", res_img)
    if cv2.waitKey(1) == ord("q"):
        np.save("face_schema.npy",delta)
        break
