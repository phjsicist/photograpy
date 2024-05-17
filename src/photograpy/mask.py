from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import LayerBase
from .layer import Layer

class Mask(LayerBase):
    def __init__(self) -> None:
        super().__init__()
        self.content: Optional[NDArray[np.float_]] = None
        self.parent: Optional[Layer] = None

    def apply(self, parent: Layer) -> None:
        self.parent = parent
        parent.mask = self
