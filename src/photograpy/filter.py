import numpy as np

from .layer import Layer

class Filter(Layer):
    def __init__(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        super().__init__()

    def apply(self, parent: Layer) -> None:
        self._parent = parent
        parent._child = self
        self._apply(parent, *self._args, **self._kwargs)

    def _apply(self, parent: Layer) -> None:
        raise NotImplementedError
