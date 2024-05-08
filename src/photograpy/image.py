from numpy.typing import NDArray
import numpy as np

from .layer import Layer


class ImageLayer(Layer):
    def __init__(self, img: NDArray, interpolation_method='linear'):
        self.content = np.atleast_2d(img).astype(int)

        if self.content.ndim == 2:
            self.content = np.stack([self.content]*3, axis=-1)
