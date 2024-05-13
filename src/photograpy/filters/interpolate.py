import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..layer import Layer

class InterpolationFilter(Layer):
    def get_interpolator(self) -> RegularGridInterpolator:
        h_axis = np.arange(self.parent.shape[0])
        w_axis = np.arange(self.parent.shape[1])
        return RegularGridInterpolator((h_axis, w_axis), self.parent.content)
    
class ReshapeFilter(InterpolationFilter):
    def __init__(self, new_shape: tuple[int, int]) -> None:
        super().__init__()
        self.new_shape = new_shape

    def update(self) -> None:
        new_height, new_width = self.new_shape

        new_h_axis = np.linspace(0, self.parent.shape[0]-1, new_height)
        new_w_axis = np.linspace(0, self.parent.shape[1]-1, new_width)

        hh, ww = np.meshgrid(new_h_axis, new_w_axis, indexing='ij')

        self._content = self.get_interpolator()((hh, ww)).astype(int)
        super().update()
