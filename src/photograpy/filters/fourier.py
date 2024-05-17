from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft2, irfft2, rfftfreq, fftfreq

from ..layer import Layer, update_func


class FourierFilter(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.blend_mode = 'none'
        self.content: Optional[NDArray[np.complex_]] = None
        self.h_freq: Optional[NDArray[np.float_]] = None
        self.w_freq: Optional[NDArray[np.float_]] = None


class FftFilter(FourierFilter):
    @update_func(50)
    def update_filter(self):
        self.content = rfft2(self.parent.content, axes=(0, 1), norm='ortho')
        self.h_freq = fftfreq(self.parent.shape[0])
        self.w_freq = rfftfreq(self.parent.shape[1])


class IfftFilter(Layer):
    def __init__(self, cast_method='clip') -> None:
        super().__init__()
        self.cast_method = cast_method

    @update_func(50)
    def _update_filter(self):
        if np.iscomplexobj(self.parent.content):
            c: NDArray[np.int_] = irfft2(self.parent.content, axes=(0, 1), norm='ortho').real
            if self.cast_method == 'clip':
                self.content = c.clip(0, 255).astype(int)
            elif self.cast_method == 'squeeze':
                self.content = ((c - c.min()) / (c.max() - c.min()) * 255).astype(int)
            else:
                raise ValueError(f'Unexpected value for cast_method: {self.cast_method}. Should be "clip" or "squeeze".')
        else:
            raise ValueError('IfftFilter can only be applied to fourier space layers.')
