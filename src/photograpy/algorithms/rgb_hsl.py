import numpy as np
from numpy.typing import NDArray

def rgb2hsl(img: NDArray[np.float_]) -> NDArray[np.float_]:
    """Converts an array from RGB to the HSL color space.
    
    See https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB_alternative.
    """
    v = np.max(img, axis=2)
    c = v - np.min(img, axis=2)

    l = v - c/2

    sl = np.zeros_like(v)
    lmin = np.min((l, 1-l), axis=0)
    sl = np.divide(v-l, lmin, out=sl, where=(lmin!=0))

    r, g, b = np.squeeze(np.split(img, 3, axis=2))

    h = np.zeros_like(v)
    h = np.divide(g-b, c, out=h, where=(c!=0)&(v==r))
    h = np.mod(h, 6, out=h, where=(c!=0)&(v==r))
    h = np.divide(b-r, c, out=h, where=(c!=0)&(v==g))
    h = np.add(h, 2, out=h, where=(c!=0)&(v==g))
    h = np.divide(r-g, c, out=h, where=(c!=0)&(v==b))
    h = np.add(h, 4, out=h, where=(c!=0)&(v==b))

    return np.stack((60*h, sl, l), axis=-1)

def hsl2rgb(img: NDArray[np.float_]) -> NDArray[np.float_]:
    """Converts an array from HSL to RGB color space.
    
    See https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB.
    """
    h, sl, l = np.squeeze(np.split(img.copy(), 3, axis=2))

    a = sl * np.min((l, 1-l), axis=0)

    def f(n):
        k = (n + h/30) % 12
        kmin = np.min((k-3, 9-k, np.ones_like(k)), axis=0)
        kmax = np.max((-np.ones_like(k), kmin), axis=0)
        return l - a*kmax
    
    r = f(0)
    g = f(8)
    b = f(4)

    return np.stack((r, g, b), axis=-1)
