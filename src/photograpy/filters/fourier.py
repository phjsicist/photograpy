from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft2, irfft2

from ..filter import Filter
from ..layer import Layer


class FourierFilter(Filter):
    def __init__(self, *args, **kwargs) -> None:
        self._fcontent: Optional[NDArray[np.complex_]] = None
        super().__init__(*args, **kwargs)

    @property
    def _content(self) -> Optional[NDArray[np.int_]]:
        c = np.log10(np.abs(self._fcontent))
        content = (c - c.min()) / (c.max() - c.min()) * 255
        return content.astype(int)
    
    @_content.setter
    def _content(self, _):
        pass


class FftFilter(FourierFilter):
    def _apply(self, parent: Layer):
        self._fcontent = rfft2(parent._content, axes=(0, 1))


class IfftFilter(Filter):
    def _apply(self, parent: FourierFilter):
        if hasattr(parent, '_fcontent'):
            self._content = irfft2(parent._fcontent, axes=(0, 1)).real.astype(int)
        else:
            raise ValueError('IfftFilter can only be applied to fourier space layers.')
