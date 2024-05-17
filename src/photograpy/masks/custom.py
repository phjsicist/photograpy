from numpy.typing import NDArray
import numpy as np

from ..base import update_func
from ..mask import Mask


class CustomMask(Mask):
    def __init__(self, img: NDArray[np.float_]):
        super().__init__()
        self.img = img

    @update_func(50)
    def _update_content(self) -> None:
        self.content = np.atleast_2d(self.img)

        if self.content.ndim == 2:
            self.content = np.stack([self.content]*3, axis=-1)
