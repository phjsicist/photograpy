from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class Image:
    def __init__(self, img: np.ndarray, interpolation_method='linear'):
        self._img = np.array(img, ndmin=3) # TODO: ndmin prepends channel dimension which is wrong

        self._shape = self._img.shape[:2]
        self._original_height, self._original_width = self._shape
        self._height, self._width = self._shape
        self._number_of_channels = self._img.shape[2]
        
        self._default_interpolation_method = interpolation_method
        self._interpolator = self._get_interpolator()

    def _get_interpolator(self, method: Optional[str]=None) -> RegularGridInterpolator:
        if method is None:
            method = self._default_interpolation_method

        h_axis = np.arange(self._height)
        w_axis = np.arange(self._width)
        return RegularGridInterpolator((h_axis, w_axis), self._img, method=method)

    @property
    def shape(self) -> tuple:
        return self._shape

    @shape.setter
    def shape(self, new_shape: tuple[int, int]) -> None:
        self._shape = new_shape
        self._height, self._width = new_shape

    def set_shape(self, new_shape: tuple[int, int], interpolation_method=None) -> None:
        new_height, new_width = new_shape

        new_h_axis = np.linspace(0, self._original_height-1, new_height)
        new_w_axis = np.linspace(0, self._original_width-1, new_width)

        hh, ww = np.meshgrid(new_h_axis, new_w_axis, indexing='ij')

        self._img = self._interpolator((hh, ww)).astype(int)
        self.shape = new_shape

# for test purposes only
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    arr = plt.imread('examples\\image.jpg')
    
    fig, ax = plt.subplots(4, 1)

    img = Image(arr)
    ax[0].imshow(img._img)

    img.set_shape((216, 800), interpolation_method='linear')
    ax[1].imshow(img._img)

    img.set_shape((216, 800), interpolation_method='cubic')
    ax[2].imshow(img._img)

    img.set_shape((216, 800), interpolation_method='quintic')
    ax[3].imshow(img._img)

    plt.show()
