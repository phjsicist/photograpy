from typing import Optional

from numpy.typing import NDArray

class BlendMode:
    def __call__(self, base: NDArray, content: NDArray, mask: Optional[NDArray]) -> NDArray:
        dtype = content.dtype
        self.base = base
        self.content = content
        self.mask = mask
        return self.blend().astype(dtype)

class BlendNone(BlendMode):
    def blend(self) -> NDArray:
        if self.mask is not None:
            return self.mask * self.content
        return self.content

class BlendNormal(BlendMode):
    def blend(self) -> NDArray:
        if self.mask is not None and self.content.shape[:2] == self.base.shape[:2]:
                return self.base + self.mask * (self.content - self.base)
        return self.content

class BlendAdd(BlendMode):
    def blend(self) -> NDArray:
        if self.mask is not None:
                return (self.base + self.mask * self.content).clip(0, 255)
        return (self.base + self.content).clip(0, 255)
    
class BlendMultiply(BlendMode):
     def blend(self) -> NDArray:
        if self.mask is not None:
             return self.base * self.mask * self.content / 255
        return self.base * self.content / 255

BLEND_MODES: dict[str, BlendMode] = {
     'none': BlendNone,
     'normal': BlendNormal,
     'add': BlendAdd,
     'multiply': BlendMultiply
     }
