import numpy as np
import cv2
from effects.base_effect import BaseEffect
import insightface
import onnxruntime as ort


model_path = "/home/bob/non_work/cb6ad443-228e-4cd3-a6fd-90ac85f035e6/models/inswapper_128.onnx"

available_providers = ort.get_available_providers()
print(available_providers)
providers = [available_providers[1]]


class Effect24(BaseEffect):
    def __init__(self) -> None:
        super().__init__()
        self._settings_dict = {"face_path": "images/sasha.png"}
        self.is_ready = False

    def settings(self, settings_dict: dict):
        self.face_detector = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
        self.face_swapper = insightface.model_zoo.get_model(model_path, providers=providers)
        self.target_img = cv2.imread(settings_dict["face_path"])
        self.target_face = self.face_detector.get(self.target_img)[0]
        self.is_ready = True

    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return img
        source_faces = self.face_detector.get(img)
        if source_faces:
            source_face = source_faces[0]
            img = self.face_swapper.get(img, source_face, self.target_face, paste_back=True)
        return img
