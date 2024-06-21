import numpy as np
from abc import ABC, abstractmethod


class BaseEffect(ABC):
    @abstractmethod
    def set_prikol_on_img(img: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def settings(settings_dict: dict):
        pass
