from typing import Optional

from ..layer import Layer, update_func

class InvertFilter(Layer):
    def __init__(self, channels: Optional[tuple[int, ...]]=None) -> None:
        super().__init__()
        if channels is None:
            self.channels = range(3)
        else:
            self.channels = channels

    @update_func(50)
    def update(self) -> None:
        self._content = self.parent.get_content()   
        for chn in self.channels:
            self._content[:, :, chn] = 255 - self._content[:, :, chn]
        super().update()
