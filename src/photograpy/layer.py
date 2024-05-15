from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .mask import Mask

def update_func(position: int, overwrite=False):
    def decorator(func):
        func._position = position
        func._overwrite = overwrite
        return func
    return decorator

class LayerMetaclass(type):
    def __init__(cls, name: str, bases: tuple[type], attrs: dict[str, Any]):
        # initialise update function registry and populate it with update functions from bases classes
        cls.update_functions: dict[str, Callable[[Any], None]] = {}
        for base in bases:
            if hasattr(base, 'update_functions'):
                cls.update_functions = {**cls.update_functions, **base.update_functions}

        # register new update functions
        for attr in attrs.values():
            position = getattr(attr, "_position", None)
            if position is not None:
                if str(position) in cls.update_functions.keys() and not attr._overwrite:
                    raise ValueError(f'Update position {position} on class {name} is already used by ' +
                                     f'function {cls.update_functions[str(position)].__qualname__}.')
                cls.update_functions[str(position)] = attr

        # create update method for calling the registered update functions
        def update(self: cls) -> None:
            update_funcs = sorted(self.__class__.update_functions.items())
            for _, func in update_funcs:
                func(self)
        setattr(cls, update.__name__, update)

class Layer(metaclass=LayerMetaclass):
    update_functions: dict[str, Callable[[Any], None]] = {}

    def __init__(self) -> None:
        self.content: Optional[NDArray[np.int_]] = None
        self.child: Optional[Layer] = None
        self.mask: Optional[Mask] = None
        self.parent: Optional[Layer] = None

    @property 
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

    def add_mask(self, mask: Mask | type[Mask], *args, **kwargs) -> None:
        if isinstance(mask, type):
            mask(*args, **kwargs).apply(self)
        else:
            mask.apply(self)

    def apply(self, parent: Layer) -> None:
        self.parent = parent
        parent.child = self

    @update_func(60)
    def _update_mask(self) -> None:
        if self.mask is not None:
            self.mask.update()

    @update_func(70)
    def _apply_mask(self) -> None:
        if self.content is not None and self.mask is not None and self.parent is not None and self.shape == self.parent.shape:
            self.content = (self.parent.content + self.mask.content * (self.content - self.parent.content)).astype(int)

    @update_func(90)
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
            else:
                self.layers[0].parent = parent

    @update_func(40)
    def _update_layers(self) -> None:
        if self.layers:
            self.layers[0].update()
