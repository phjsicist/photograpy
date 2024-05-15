from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .mask import Mask

def update_func(position: int):
    def decorator(func):
        func._position = position
        return func
    return decorator

class LayerMetaclass(type):
    def __init__(cls, name: str, bases: tuple[type], attrs: dict[str, Any]):
        if bases:
            cls.update_functions: dict[str, Callable[[Any], None]] = bases[0].update_functions.copy()
        for attr in attrs.values():
            position = getattr(attr, "_position", None)
            if position is not None:
                if str(position) in cls.update_functions.keys():
                    raise ValueError(f'Update position {position} on class {name} is already used by ' +
                                     f'function {cls.update_functions[str(position)].__qualname__}.')
                cls.update_functions[str(position)] = attr

class Layer(metaclass=LayerMetaclass):
    update_functions: dict[str, Callable[[Any], None]] = {}

    def __init__(self) -> None:
        self._content: Optional[NDArray[np.int_]] = None
        self.child: Optional[Layer] = None
        self.mask: Optional[Mask] = None
        self.parent: Optional[Layer] = None

    def update(self):
        update_funcs = sorted(self.__class__.update_functions.items())
        print(update_funcs)
        for _, func in update_funcs:
            func(self)

    @property
    def content(self) -> Optional[NDArray[np.int_]]:
        if self.mask is None or self.parent is None:
            return self._content
        else:
            return (self.parent.content + self.mask.content * (self._content - self.parent.content)).astype(int)

    @property 
    def shape(self) -> tuple[int, int]:
        if self._content is None:
            return None
        else:
            return self._content.shape[:2]

    def get_content(self) -> Optional[NDArray[np.int_]]:
        if self._content is None:
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

    @update_func(90)
    def update_dependents(self) -> None:
        if self.mask is not None:
            self.mask.update()
        if self.child is not None:
            self.child.update()
    
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
        elif self.parent is not None:
            self.parent.add_layer(layer, args, kwargs)
            self.parent.child = self
        self.layers.append(layer)

    def apply(self, parent: Layer) -> None:
        super().apply(parent)
        if self.layers:
            if isinstance(self.layers[0], LayerGroup):
                self.layers[0].apply(parent)
            else:
                self.layers[0].parent = parent

    def update(self) -> None:
        if self.layers:
            self.layers[0].update()
        super().update()
