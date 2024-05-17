from . import FftFilter, IfftFilter
from .. import LayerGroup
from ..masks import HiPassMask, OpacityMask

class SharpenFilter(LayerGroup):
    def __init__(self, radius: float, amount: float) -> None:
        layers = (FftFilter(),
                  IfftFilter())
        layers[0].add_mask(HiPassMask, radius=radius)
        super().__init__(layers)
        self.blend_mode = 'add'
        self.add_mask(OpacityMask, amount)
