from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import LayerBase, update_func

if TYPE_CHECKING:
    from .mask import Mask


class Layer(LayerBase):
    def __init__(self) -> None:
        super().__init__()
        self.content: Optional[NDArray[np.int_]] = None
        self.child: Optional[Layer] = None
        self.mask: Optional[Mask] = None
        self.parent: Optional[Layer] = None

    def get_content(self) -> Optional[NDArray[np.int_]]:
        if self.content is None:
            return None
        return self.content.copy()

    def add_layer(self, layer: Layer | type[Layer], *args, **kwargs) -> None:
        if isinstance(layer, type):
            layer(*args, **kwargs).apply(self)
        else:
            layer.apply(self)

    def add_mask(self, mask: Mask | type[Mask], *args, **kwargs) -> None:
        if isinstance(mask, type):
            mask(*args, **kwargs).apply(self)
        else:
            mask.apply(self)

    def apply(self, parent: Layer) -> None:
        self.parent = parent
        parent.child = self

    @update_func(60, timer=False)
    def _update_mask(self) -> None:
        if self.mask is not None:
            self.mask.update()

    @update_func(70, timer=False)
    def _apply_mask(self) -> None:
        if self.content is not None and self.mask is not None and self.parent is not None and self.shape == self.parent.shape:
            self.content = (self.parent.content + self.mask.content * (self.content - self.parent.content)).astype(int)

    @update_func(90, timer=False)
    def _update_child(self) -> None:
        if self.child is not None:
            self.child.update()


class LayerGroup(Layer):
    def __init__(self, layers: Iterable[Layer]=()) -> None:
        super().__init__()
        self.layers: list[Layer] = []
        for f in layers:
            self.append_layer(f)

    @property
    def content(self) -> Optional[NDArray[np.int_]]:
        if self.layers:
            return self.layers[-1].content
        else:
            return None
        
    @content.setter
    def content(self, _) -> None:
        pass

    def append_layer(self, layer: Layer | type[Layer], *args, **kwargs) -> None:
        if self.layers:
            self.layers[-1].add_layer(layer, args, kwargs)
        elif self.parent is not None:
            self.parent.add_layer(layer, args, kwargs)
            self.parent.child = self
        self.layers.append(layer)

    def apply(self, parent: Layer) -> None:
        super().apply(parent)
        if self.layers:
            if isinstance(self.layers[0], LayerGroup):
                self.layers[0].apply(parent)
                parent.child = self
            else:
                self.layers[0].parent = parent

    @update_func(40)
    def _update_layers(self) -> None:
        if self.layers:
            self.layers[0].update()
