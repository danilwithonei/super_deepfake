import numpy as np
from abc import ABC, abstractmethod


class BaseEffect(ABC):
    _settings_dict: dict[str, str] = {}

    @abstractmethod
    def set_prikol_on_img(self, img: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def settings(self, settings_dict: dict[str, str]):
        pass
