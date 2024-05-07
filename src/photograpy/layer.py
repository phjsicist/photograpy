from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .filter import Filter

class Layer:
    def __init__(self) -> None:
        self._content: Optional[NDArray[np.int_]] = None
        self._parent: Optional[Layer] = None
        self._child: Optional[Layer] = None

    def shape(self) -> tuple[int, int]:
        return self._content.shape[:2]

    def get_content(self) -> Optional[NDArray[np.int_]]:
        if self._content is None:
            return None
        return self._content.copy()

    def add_filter(self, filter: Filter | type[Filter], *args, **kwargs) -> None:
        if isinstance(filter, type):
            filter(*args, **kwargs).apply(self)
        else:
            filter.apply(self)
