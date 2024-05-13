from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .mask import Mask


class Layer:
    def __init__(self, *args, **kwargs) -> None:
        self._content: Optional[NDArray[np.int_]] = None
        self.child: Optional[Layer] = None
        self.mask: Optional[Mask] = None
        self.parent: Optional[Layer] = None

    @property
    def content(self) -> Optional[NDArray[np.int_]]:
        if self.mask is None:
            return self._content
        else:
            return (self.parent.content + self.mask.content * (self._content - self.parent.content)).astype(int)
        
    def shape(self) -> tuple[int, int]:
        if self.content is None:
            return None
        else:
            return self.content.shape[:2]

    def get_content(self) -> Optional[NDArray[np.int_]]:
        if self.content is None:
            return None
        return self.content.copy()

    def add_layer(self, layer: Layer | type[Layer], *args, **kwargs) -> None:
        if isinstance(layer, type):
            layer(*args, **kwargs).apply(self)
        else:
            layer.apply(self)

    def apply(self, parent: Layer) -> None:
        self.parent = parent
        parent.child = self
        self.update()

    def update(self) -> None:
        raise NotImplementedError
    
    def add_mask(self, mask: Mask | type[Mask], *args, **kwargs) -> None:
        if isinstance(mask, type):
            mask(*args, **kwargs).apply(self)
        else:
            mask.apply(self)


class LayerGroup(Layer):
    def __init__(self, layers: Iterable[Layer]=()) -> None:
        super().__init__()
        self.layers: list[Layer] = []
        for f in layers:
            self.append_layer(f)

    @property
    def _content(self) -> Optional[NDArray[np.int_]]:
        if self.layers:
            return self.layers[-1].content
        else:
            return None
        
    @_content.setter
    def _content(self, _) -> None:
        pass

    def append_layer(self, layer: Layer | type[Layer], *args, **kwargs) -> None:
        if self.layers:
            self.layers[-1].add_layer(layer, args, kwargs)
            self.layers.append(layer)
        else:
            self.layers.append(layer)
            self.parent.add_layer(self)

    def update(self) -> None:
        if self.layers:
            self.layers[0].parent = self.parent
            self.layers[0].update()
        else:
            pass
