from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft2, irfft2

from ..layer import Layer


class FourierFilter(Layer):
    def __init__(self, *args, **kwargs) -> None:
        self.fcontent: Optional[NDArray[np.complex_]] = None
        super().__init__(*args, **kwargs)

    @property
    def content(self) -> Optional[NDArray[np.int_]]:
        c = np.log10(np.abs(self.fcontent))
        content = (c - c.min()) / (c.max() - c.min()) * 255
        return content.astype(int)
    
    @content.setter
    def content(self, _):
        pass


class FftFilter(FourierFilter):
    def update(self):
        self.fcontent = rfft2(self.parent.content, axes=(0, 1))


class IfftFilter(Layer):
    def update(self):
        if hasattr(self.parent, 'fcontent'):
            self.content = irfft2(self.parent.fcontent, axes=(0, 1)).real.astype(int)
        else:
            raise ValueError('IfftFilter can only be applied to fourier space layers.')
