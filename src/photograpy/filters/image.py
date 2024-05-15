from numpy.typing import NDArray
import numpy as np

from ..layer import Layer, update_func


class ImageLayer(Layer):
    def __init__(self, img: NDArray):
        super().__init__()
        self.img = img

    @update_func(50)
    def update_layer(self) -> None:
        self._content = np.atleast_2d(self.img).astype(int)

        if self._content.ndim == 2:
            self._content = np.stack([self._content]*3, axis=-1)
