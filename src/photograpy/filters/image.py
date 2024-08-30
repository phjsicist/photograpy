from numpy.typing import NDArray
import numpy as np

from ..layer import Layer, update_func


class ImageLayer(Layer):
    def __init__(self, img: NDArray):
        super().__init__()
        self.img = img

    @update_func(50)
    def _update_layer(self) -> None:
        self.content = np.atleast_2d(self.img)
        if np.issubdtype(self.content.dtype, np.integer):
            self.content = (self.content / 255).astype(np.float_)

        if self.content.ndim == 2:
            self.content = np.stack([self.content]*3, axis=-1)
