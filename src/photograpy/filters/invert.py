from typing import Optional

from ..layer import Layer

class InvertFilter(Layer):
    def __init__(self, channels: Optional[tuple[int, ...]]=None) -> None:
        super().__init__()
        if channels is None:
            self.channels = range(3)
        else:
            self.channels = channels

    def _apply(self) -> None:
        self.content = self.parent.get_content()   
        for chn in self.channels:
            self.content[:, :, chn] = 255 - self.content[:, :, chn]
