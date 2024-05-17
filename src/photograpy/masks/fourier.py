import numpy as np

from .. import Mask, update_func

class HiPassMask(Mask):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = radius

    @update_func(50)
    def _update_content(self) -> None:
        self.content = np.zeros(self.parent.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.parent.h_freq[i]**2 + self.parent.w_freq[j]**2 > 1/self.radius**2:
                    self.content[i, j] = 1.0
        self.content = np.stack([self.content]*3, axis=-1)
