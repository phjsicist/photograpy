from numpy.typing import NDArray
import numpy as np

from ..mask import Mask


class OpacityMask(Mask):
    def __init__(self, opacity: float):
        super().__init__()
        self.opacity = opacity

    def update(self) -> None:
        self.content = np.full(self.shape, self.opacity)
