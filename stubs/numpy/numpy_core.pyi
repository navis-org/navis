from typing import *
from typing_extensions import Literal

import pandas as pd

pi: float


def array(x: Union[Sequence[Any],
                   Iterable[Any],
                   ndarray], **kwargs) -> ndarray: ...


class int32: ...


class float32: ...


class ndarray(List):
    shape: Sequence[int]
    T: 'ndarray'
    ndim: int

    def __setitem__(self, idx, value) -> None: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator: ...

    def __invert__(self) -> 'ndarray': ...

    def __eq__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __ne__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __lt__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __le__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __gt__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __ge__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __div__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __truediv__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __mul__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __imul__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __sub__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __rsub__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __add__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __pow__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __or__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __xor__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    def __and__(self, other: Any) -> ndarray[bool]: ...  # type: ignore  # already defined in typeshed - does not like being overriden

    # can't really overload this as depends on dimensions of array
    def __getitem__(self,  # type: ignore  # already defined in typeshed - does not like being overriden
                    key: Union[int, slice,
                               Sequence[Union[int, bool]],
                               Iterable[Union[int, bool]],
                               Tuple[Union[slice, int,
                                           Iterable[Union[int, bool]]],
                                     Union[slice, int,
                                           Iterable[Union[int, bool]]]],
                               ]) -> Union['ndarray', str, int, bool]: ...

    def tolist(self) -> List: ...

    @overload
    def sum(self, axis: Literal[None] = ..., **kwargs) -> float: ...

    @overload
    def sum(self, axis: int, **kwargs) -> ndarray: ...

    @overload
    def min(self, axis: Literal[None] = ..., **kwargs) -> float: ...

    @overload
    def min(self, axis: int, **kwargs) -> ndarray: ...

    @overload
    def max(self, axis: Literal[None] = ..., **kwargs) -> float: ...

    @overload
    def max(self, axis: int, **kwargs) -> ndarray: ...

    @overload
    def mean(self, axis: Literal[None] = ..., **kwargs) -> float: ...

    @overload
    def mean(self, axis: int, **kwargs) -> ndarray: ...

    @overload
    def any(self, axis: Literal[None] = ..., **kwargs) -> float: ...

    @overload
    def any(self, axis: int, **kwargs) -> ndarray: ...

    @overload
    def astype(self, ty: Literal[str], **kwargs) -> 'ndarray[str]': ...

    @overload
    def astype(self, ty: Literal[int], **kwargs) -> 'ndarray[int]': ...

    @overload
    def astype(self, ty: Literal[float], **kwargs) -> 'ndarray[float]': ...

    @overload
    def astype(self, ty: Literal[bool], **kwargs) -> 'ndarray[bool]': ...

    def reshape(self, shape: Tuple, **kwargs) -> 'ndarray': ...


def linspace(*args, **kwargs) -> ndarray: ...


def insert(*args, **kwargs) -> ndarray: ...


def diff(*args, **kwargs) -> ndarray: ...


def cumsum(*args, **kwargs) -> ndarray: ...


def sum(*args, **kwargs) -> ndarray: ...


@overload
def nansum(*args, axis: Literal[None], **kwargs) -> float: ...


@overload
def nansum(*args, axis: int, **kwargs) -> ndarray: ...


def min(*args, **kwargs) -> ndarray: ...


def max(*args, **kwargs) -> ndarray: ...


def mean(*args, **kwargs) -> ndarray: ...


def exp(*args, **kwargs) -> ndarray: ...


def sqrt(*args, **kwargs) -> ndarray: ...


def vstack(*args, **kwargs) -> ndarray: ...


def zeros(*args, **kwargs) -> ndarray: ...


def round(*args, **kwargs) -> ndarray: ...


def unique(*args, **kwargs) -> ndarray: ...


def any(*args, **kwargs) -> ndarray: ...


def append(*args, **kwargs) -> ndarray: ...


def empty(*args, **kwargs) -> ndarray: ...


def isnan(*args, **kwargs) -> ndarray: ...


def arange(*args, **kwargs) -> ndarray: ...


def argmin(*args, **kwargs) -> ndarray: ...


def remainder(*args, **kwargs) -> ndarray: ...


def split(*args, **kwargs) -> ndarray: ...


def sin(*args, **kwargs) -> ndarray: ...


def cos(*args, **kwargs) -> ndarray: ...


def ones(*args, **kwargs) -> ndarray: ...


def outer(*args, **kwargs) -> ndarray: ...


def size(*args, **kwargs) -> ndarray: ...


def where(*args, **kwargs) -> ndarray: ...
