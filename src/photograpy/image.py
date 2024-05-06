from typing import Optional

from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .layer import Layer


class ImageLayer(Layer):
    def __init__(self, img: NDArray, interpolation_method='linear'):
        self._content = np.atleast_2d(img).astype(int)

        if self._content.ndim == 2:
            self._content = np.stack([self._content]*3, axis=-1)

        self._shape = self._content.shape[:2]
        self._original_shape = self._shape
        self._height, self._width = self._shape
        
        self._default_interpolation_method = interpolation_method
        self._interpolator = self._get_interpolator()

    def _get_interpolator(self, method: Optional[str]=None) -> RegularGridInterpolator:
        if method is None:
            method = self._default_interpolation_method

        h_axis = np.arange(self._height)
        w_axis = np.arange(self._width)
        return RegularGridInterpolator((h_axis, w_axis), self._content, method=method)

    @property
    def shape(self) -> tuple:
        return self._shape

    @shape.setter
    def shape(self, new_shape: tuple[int, int]) -> None:
        self._shape = new_shape
        self._height, self._width = new_shape

    def set_shape(self, new_shape: tuple[int, int], interpolation_method=None) -> None:
        new_height, new_width = new_shape

        new_h_axis = np.linspace(0, self._original_shape[0]-1, new_height)
        new_w_axis = np.linspace(0, self._original_shape[1]-1, new_width)

        hh, ww = np.meshgrid(new_h_axis, new_w_axis, indexing='ij')

        self._content = self._interpolator((hh, ww)).astype(int)
        self.shape = new_shape
