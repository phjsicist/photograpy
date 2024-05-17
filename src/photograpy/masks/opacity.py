from numpy.typing import NDArray
import numpy as np

from ..base import update_func
from ..mask import Mask


class OpacityMask(Mask):
    def __init__(self, opacity: float):
        super().__init__()
        self.opacity = opacity

    @update_func(50)
    def _update_content(self) -> None:
        self.content = np.full(self.shape, self.opacity)
