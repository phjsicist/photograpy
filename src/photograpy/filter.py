from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from .layer import Layer

class Filter(Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    def apply(self, parent: Layer) -> None:
        self._parent = parent
        parent._child = self
        self._apply(parent, *self._args, **self._kwargs)

    def _apply(self, parent: Layer) -> None:
        raise NotImplementedError
    
class FilterGroup(Filter):
    def __init__(self, filters: Iterable[Filter]=()) -> None:
        super().__init__((), {})
        self._filters = []
        for f in filters:
            self.append_filter(f)

    @property
    def _content(self) -> Optional[NDArray[np.int_]]:
        if self._filters:
            return self._filters[-1]._content
        else:
            return None
        
    @_content.setter
    def _content(self, value: Any) -> None:
        pass

    def append_filter(self, filter: Filter | type[Filter], *args, **kwargs) -> None:
        if self._filters:
            self._filters[-1].add_filter(filter, args, kwargs)
            self._filters.append(filter)
        else:
            self._filters.append(filter)
            self._parent.add_filter(self)

    def _apply(self, parent: Layer, args, kwargs) -> None:
        if self._filters:
            self._filters[0]._parent = parent
            self._filters[0]._apply(parent, *args, **kwargs)
        else:
            pass
