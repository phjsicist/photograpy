from numpy.typing import NDArray
import numpy as np

from ..mask import Mask


class CustomMask(Mask):
    def __init__(self, img: NDArray[np.float_]):
        super().__init__()
        self.img = img

    def update(self) -> None:
        self._content = np.atleast_2d(self.img)
