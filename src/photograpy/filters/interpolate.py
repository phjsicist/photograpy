import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..filter import Filter
from ..layer import Layer

class InterpolationFilter(Filter):
    def get_interpolator(self) -> RegularGridInterpolator:
        h_axis = np.arange(self._parent.shape()[0])
        w_axis = np.arange(self._parent.shape()[1])
        return RegularGridInterpolator((h_axis, w_axis), self._parent._content)
    
class ReshapeFilter(InterpolationFilter):
    def _apply(self, parent: Layer, new_shape: tuple[int, int]) -> None:
        new_height, new_width = new_shape

        new_h_axis = np.linspace(0, parent.shape()[0]-1, new_height)
        new_w_axis = np.linspace(0, parent.shape()[1]-1, new_width)

        hh, ww = np.meshgrid(new_h_axis, new_w_axis, indexing='ij')

        self._content = self.get_interpolator()((hh, ww)).astype(int)
