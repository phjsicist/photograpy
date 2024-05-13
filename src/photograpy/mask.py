from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .layer import Layer

class Mask:
    def __init__(self) -> None:
        self.content: Optional[NDArray[np.float_]] = None
        self.parent: Optional[Layer] = None

    @property
    def shape(self) -> tuple[int, int]:
        if self.content is None:
            return None
        else:
            return self.content.shape[:2]

    def apply(self, parent: Layer) -> None:
        self.parent = parent
        parent.mask = self

    def update(self):
        raise NotImplementedError
