from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .layer import Layer

class Mask:
    def __init__(self) -> None:
        self.content: Optional[NDArray[np.float_]] = None
        self.fcontent: Optional[NDArray[np.complex_]] = None
        self.parent: Optional[Layer] = None

    def apply(self, parent: Layer) -> None:
        self.parent = parent
        parent.mask = self

    def update(self):
        raise NotImplementedError
