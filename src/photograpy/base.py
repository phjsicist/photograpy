from datetime import timedelta
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Optional

from numpy.typing import NDArray

def update_func(position: int, timer=True, overwrite=False):
    def decorator(func: Callable):
        func._position = position
        func._overwrite = overwrite

        @wraps(func)
        def timer_wrapper(*args, **kwargs):
            start = perf_counter()
            result = func(*args, **kwargs)
            end = perf_counter()
            timer_wrapper._runtime = timedelta(seconds=end - start)
            return result

        if timer:
            return timer_wrapper
        return func
    return decorator


class LayerMeta(type):
    def __init__(cls, name: str, bases: tuple[type], attrs: dict[str, Any]) -> None:
        # initialise update function registry and populate it with update functions from bases classes
        cls.update_functions: dict[str, Callable[[Any], None]] = {}
        for base in bases:
            if hasattr(base, 'update_functions'):
                cls.update_functions = {**cls.update_functions, **base.update_functions}

        # register new update functions
        for attr in attrs.values():
            position = getattr(attr, '_position', None)
            if position is not None:
                if str(position) in cls.update_functions.keys() and not attr._overwrite:
                    raise ValueError(f'Update position {position} on class {name} is already used by ' +
                                     f'function {cls.update_functions[str(position)].__qualname__}.')
                cls.update_functions[str(position)] = attr

        # create update method for calling the registered update functions
        def update(self: cls) -> None:
            self.update_runtime = timedelta(0)
            update_funcs = sorted(self.__class__.update_functions.items())
            for position, func in update_funcs:
                func(self)
                if hasattr(func, '_runtime'):
                    self.update_runtime += func._runtime

        setattr(cls, update.__name__, update)


class LayerBase(metaclass=LayerMeta):
    def __init__(self) -> None:
        self.content: Optional[NDArray[Any]] = None

    @property 
    def shape(self) -> Optional[tuple[int, int]]:
        if self.content is None:
            return None
        else:
            return self.content.shape[:2]
