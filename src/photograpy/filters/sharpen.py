import numpy as np

from .fourier import FftFilter, IfftFilter
from ..layer import LayerGroup
from ..mask import Mask
from ..masks.opacity import OpacityMask

class SharpenFilter(LayerGroup):
    def __init__(self, radius: float, amount: float) -> None:
        layers = (FftFilter(),
                  IfftFilter())
        layers[0].add_mask(HiPassMask, radius=radius)
        super().__init__(layers)
        self.add_mask(OpacityMask, amount)

class HiPassMask(Mask):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = radius

    def update(self) -> None:
        self.content = np.zeros(self.parent.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.parent.h_freq[i]**2 + self.parent.w_freq[j]**2 > 1/self.radius**2:
                    self.content[i, j] = 1.0
        self.content = np.stack([self.content]*3, axis=-1)
