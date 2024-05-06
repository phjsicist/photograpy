from typing import Optional

from ..filter import Filter
from ..layer import Layer

class InvertFilter(Filter):
    def _apply(self, parent: Layer, channels: Optional[tuple[int, ...]]=None) -> None:
        self._content = parent.get_content()
        if channels is None:
            channels = range(self._content.shape[2])
        for chn in channels:
            self._content[:, :, chn] = 255 - self._content[:, :, chn]
