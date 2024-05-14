from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft2, irfft2, rfftfreq, fftfreq

from ..layer import Layer


class FourierFilter(Layer):
    def __init__(self) -> None:
        self._fcontent: Optional[NDArray[np.complex_]] = None
        self.h_freq: Optional[NDArray[np.float_]] = None
        self.w_freq: Optional[NDArray[np.float_]] = None
        super().__init__()

    @property 
    def shape(self) -> tuple[int, int]:
        if self._fcontent is None:
            return None
        else:
            return self._fcontent.shape[:2]

    @property
    def fcontent(self) -> Optional[NDArray[np.complex_]]:
        if self.mask is None or self._fcontent is None:
            return self._fcontent
        else:
            return self.mask.content * self._fcontent

    @property
    def content(self) -> Optional[NDArray[np.int_]]:
        if self.fcontent is None:
            return None
        c = np.log10(np.abs(self.fcontent))
        content = (c - c.min()) / (c.max() - c.min()) * 255
        return content.astype(int)
    
    @content.setter
    def content(self, _):
        pass


class FftFilter(FourierFilter):
    def update(self):
        self._fcontent = rfft2(self.parent.content, axes=(0, 1), norm='ortho')
        self.h_freq = fftfreq(self.parent.shape[0])
        self.w_freq = rfftfreq(self.parent.shape[1])
        super().update()


class IfftFilter(Layer):
    def __init__(self, cast_method='clip') -> None:
        super().__init__()
        self.cast_method = cast_method

    def update(self):
        if hasattr(self.parent, '_fcontent'):
            c: NDArray[np.int_] = irfft2(self.parent.fcontent, axes=(0, 1), norm='ortho').real
            if self.cast_method == 'clip':
                self._content = c.clip(0, 255).astype(int)
            elif self.cast_method == 'squeeze':
                self._content = ((c - c.min()) / (c.max() - c.min()) * 255).astype(int)
            else:
                raise ValueError(f'Unexpected value for cast_method: {self.cast_method}. Should be "clip" or "squeeze".')
        else:
            raise ValueError('IfftFilter can only be applied to fourier space layers.')
        super().update()
