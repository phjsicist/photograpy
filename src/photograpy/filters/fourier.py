from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft2, irfft2, rfftfreq, fftfreq

from ..layer import Layer, update_func


class FourierFilter(Layer):
    def __init__(self) -> None:
        self.fcontent: Optional[NDArray[np.complex_]] = None
        self.h_freq: Optional[NDArray[np.float_]] = None
        self.w_freq: Optional[NDArray[np.float_]] = None
        super().__init__()

    @property 
    def shape(self) -> tuple[int, int]:
        if self.fcontent is None:
            return None
        else:
            return self.fcontent.shape[:2]

    @update_func(70, overwrite=True)
    def _apply_mask(self) -> None:
        if self.mask is not None and self.fcontent is not None:
            self.fcontent = self.mask.content * self.fcontent

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
    @update_func(50)
    def update_filter(self):
        self.fcontent = rfft2(self.parent.content, axes=(0, 1), norm='ortho')
        self.h_freq = fftfreq(self.parent.shape[0])
        self.w_freq = rfftfreq(self.parent.shape[1])


class IfftFilter(Layer):
    def __init__(self, cast_method='clip') -> None:
        super().__init__()
        self.cast_method = cast_method

    @update_func(50)
    def _update_filter(self):
        if hasattr(self.parent, 'fcontent'):
            c: NDArray[np.int_] = irfft2(self.parent.fcontent, axes=(0, 1), norm='ortho').real
            if self.cast_method == 'clip':
                self.content = c.clip(0, 255).astype(int)
            elif self.cast_method == 'squeeze':
                self.content = ((c - c.min()) / (c.max() - c.min()) * 255).astype(int)
            else:
                raise ValueError(f'Unexpected value for cast_method: {self.cast_method}. Should be "clip" or "squeeze".')
        else:
            raise ValueError('IfftFilter can only be applied to fourier space layers.')
